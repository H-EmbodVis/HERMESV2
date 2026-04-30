# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import sys
import json
import random
import logging
import warnings
import traceback

from copy import deepcopy
from functools import partial
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import torch
import torch.distributed as dist
import numpy as np
import transformers
import torch.nn as nn

from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer,
    HfArgumentParser, Trainer, TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import enable_default_handler, enable_explicit_format, set_verbosity

from ...internvl.dist_utils import init_dist
from ...internvl.model.internvl_chat import (
    InternVisionConfig, InternVisionModel,
    InternVLChatConfig, InternVLChatModel,
)
from ...internvl.patch import (
    concat_pad_data_collator,
    replace_train_dataloader,
    replace_qwen3_attention_class,
    replace_gpt_oss_with_flash_sink_attn,
)
IMG_CONTEXT_TOKEN = '<|reserved_special_token_3|>'
IMG_START_TOKEN = '<|reserved_special_token_4|>'
IMG_END_TOKEN = '<|reserved_special_token_5|>'
from ...internvl.train.dataset import (
    ConcatDataset, TCSLoader,
    build_transform, dynamic_preprocess,
    preprocess_pretrain, preprocess_internvl2_5, preprocess_internvl3_5_gpt_oss, preprocess_llama3
)
from ...internvl.train.dataset_packed import PackedDataset, packed_collate_fn
from mmdet.models import HEADS

from ipdb import set_trace
import warnings
from ...internvl.model.internvl_chat.conversation import get_conv_template

use_tcs_loader = bool(os.environ.get("USE_TCS_LOADER", "0") == "1")

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def wrap_llm_lora(llm, r=128, lora_alpha=256, lora_dropout=0.05, target_modules=None):
    # Determine the target modules based on the architecture of the language model
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type='CAUSAL_LM'
    )
    llm = get_peft_model(llm, lora_config)
    llm.enable_input_require_grads()
    llm.print_trainable_parameters()
    return llm


def _freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False


# class MLPV2(nn.Module):
#     def __init__(self, input_channel, output_channel):
#         super(MLPV2, self).__init__()
#         self.fc1 = nn.Sequential(
#             nn.LayerNorm(input_channel),
#             nn.Linear(input_channel, output_channel),
#             nn.GELU(),
#             nn.Linear(output_channel, output_channel),
#         )
#
#     def forward(self, x):
#         with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#             x = self.fc1(x)
#         return x
class MLPV2(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(MLPV2, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_channel, output_channel),
            nn.GELU(),
            nn.Linear(output_channel, output_channel),
        )

    def forward(self, x):
        if not self.training:
            x_dtype = x.dtype
            weight_dtype = self.fc1[0].weight.dtype
            x = x.to(weight_dtype)
        x = self.fc1(x)
        if not self.training:
            x = x.to(x_dtype)
        return x


@HEADS.register_module()
class HERMESLlama(nn.Module):
    """
    Provides: model loading, LoRA config, data preprocess, learnable queries, and loss computation.
    """

    def __init__(
            self,
            model_path: str = None,
            load_weight: bool = True,
            set_lora: bool = True,
            is_pretraining: bool = False,
            chat_cfg: Optional[dict] = None,
            input_dim: int = 1024,
            attention_type: str = 'flash_attention_2',
            img_length: int = 60,
            num_learnable_query: int = 0,
            use_query_loss: bool = False,
            use_text_emb_fusion: bool = False,
            llm_path: str = None,
            unfreeze_lm_head: bool = False,
    ):
        super().__init__()

        assert model_path is not None, 'model_path must be provided.'
        self.model_path = model_path
        self.tokenizer_path = model_path
        self.unfreeze_lm_head = unfreeze_lm_head

        self.template_name = 'hermes_llama'
        self.preprocess_func = preprocess_llama3

        # Default LoRA target modules
        self.target_modules = [
            'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
            'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
        ]

        self.ds_name = None
        self.load_weight = load_weight
        self.img_length = img_length
        self.chat_cfg = chat_cfg or {}
        self.use_query_loss = use_query_loss
        self.use_text_emb_fusion = use_text_emb_fusion

        # runtime dtype: use bf16 when LoRA to save memory
        self.lora = set_lora
        self.is_pretraining = is_pretraining
        self.torch_dtype = torch.bfloat16 if set_lora else torch.float32

        self.num_learnable_query = num_learnable_query
        self.learnable_query_token_id = -10  # sentinel id for query slots (not part of vocab)

        # will be initialized in create_llm()
        self.llm = None
        self.tokenizer = None
        self.img_context_token_id = None
        self.in_mlp = None
        self.out_mlp = None
        self.query_mlp = None

        # store requested input feature dim
        self._input_dim = input_dim

    def create_llm(self):
        """
        Build Llama LLM.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            add_eos_token=False,
            trust_remote_code=True,
            use_fast=False,
        )
        self.tokenizer.tokenizer_path = self.tokenizer_path
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.model_max_length = 8192
        token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN]
        self.num_new_tokens = self.tokenizer.add_tokens(token_list, special_tokens=True)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        )
        if self.num_new_tokens > 0:
            self.llm.resize_token_embeddings(len(self.tokenizer))
            input_embeddings = self.llm.get_input_embeddings().weight.data
            input_embeddings_avg = input_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)
            input_embeddings[-self.num_new_tokens:] = input_embeddings_avg
            output_embeddings = self.llm.get_output_embeddings().weight.data
            output_embeddings_avg = output_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-self.num_new_tokens:] = output_embeddings_avg
            self.llm.config.vocab_size = len(self.tokenizer)

        self.llm.config.use_cache = False
        # self.llm.config._attn_implementation = 'flash_attention_2'

        # build projection heads according to LLM hidden size
        llm_hidden = getattr(self.llm.config, 'hidden_size', None)
        if llm_hidden is None:
            # try common alternatives
            llm_hidden = getattr(self.llm.config, 'n_embd', None)
        assert llm_hidden is not None, 'Cannot infer LLM hidden size.'

        self.in_mlp = MLPV2(self._input_dim, llm_hidden)
        self.out_mlp = MLPV2(llm_hidden, self._input_dim)
        if self.use_query_loss:
            self.query_mlp = MLPV2(llm_hidden, llm_hidden)
        # LoRA / freezing
        if self.is_pretraining:
            _freeze_params(self.llm)
            if self.unfreeze_lm_head:
                for param in self.llm.lm_head.parameters():
                    param.requires_grad = True
            if self.lora:
                warnings.warn('Using LoRA during pretraining is not recommended; enabling as requested.')
                self.set_lora(unfreeze_lm_head=self.unfreeze_lm_head)
        else:
            if self.lora:
                self.set_lora(unfreeze_lm_head=self.unfreeze_lm_head)
            else:
                # fully finetune
                self.set_lora(freeze_llm=False, use_llm_lora=0)

        torch.cuda.empty_cache()

    def set_lora(self,
                 freeze_llm=True,
                 unfreeze_lm_head=False,
                 use_llm_lora=128,
                 ):
        if freeze_llm:
            self.llm = self.llm.eval()
            _freeze_params(self.llm)
        if unfreeze_lm_head:
            for name, param in self.llm.lm_head.named_parameters():
                param.requires_grad = True
        if use_llm_lora:
            wrap_llm_lora(self.llm, r=use_llm_lora, lora_alpha=2 * use_llm_lora, target_modules=self.target_modules)

    def preprocess(self, conversations=None, img_length: int = 3600):
        if conversations is None:
            conversations = [
                {"from": "user", "value": "<image>\n请描述这张图片。"},
                {"from": "assistant", "value": "这是一张图片。"},
            ]

        ret = self.preprocess_func(self.template_name, [deepcopy(conversations)],
                                   self.tokenizer, [img_length],
                                   group_by_length=True, ds_name=self.ds_name)
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
        )
        return ret

    def forward(
            self,
            bev=None,
            text=None,
            learn_query=None,
            device=torch.device('cuda:0'),
            position_ids=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=True,
    ):
        if bev is None:
            bev = torch.rand(1, self.img_length, self.img_length, self._input_dim, dtype=torch.float32, device=device)
        else:
            device = bev.device

        try:
            input_ids = text['input_ids'].to(device)
            attention_mask = text['attention_mask'].to(device)
            labels = text['labels'].to(device)
        except:
            processed_text = self.preprocess(text, img_length=self.img_length)
            input_ids = processed_text['input_ids'].unsqueeze(0).to(device)
            attention_mask = processed_text['attention_mask'].unsqueeze(0).to(device)
            labels = processed_text['labels'].unsqueeze(0).to(device)

        input_embeds = self.llm.get_input_embeddings()(input_ids).clone()

        B, N, C = input_embeds.shape
        _, H, W, _ = bev.shape
        input_embeds = input_embeds.reshape(B * N, C)
        bev = bev.reshape(B, H * W, -1)

        bev_embeds = self.in_mlp(bev)
        input_ids = input_ids.reshape(B * N)

        # optional learnable queries
        if self.num_learnable_query > 0 and learn_query is not None:
            if learn_query.size(1) != self.num_learnable_query:
                self.num_learnable_query = learn_query.size(1)
            if learn_query.shape[-1] != C:
                learn_query = self.in_mlp(learn_query)
            query_ids = torch.zeros(self.num_learnable_query, device=input_ids.device) + self.learnable_query_token_id
            input_ids = torch.cat((input_ids, query_ids), dim=0)

            selected_query = (input_ids == self.learnable_query_token_id)
            for_query = torch.zeros((self.num_learnable_query, input_embeds.size(1)), device=input_embeds.device,
                                    dtype=input_embeds.dtype)
            input_embeds = torch.cat((input_embeds, for_query), dim=0)

            try:
                input_embeds[selected_query] = learn_query.reshape(-1, C).to(input_embeds.dtype)
                ignore_flag = False
            except Exception as e:
                print(f'warning (learn_query inject): {e}')
                input_embeds[selected_query] = 0.0
                assert False, 'learn_query must be provided correctly when num_learnable_query > 0'
        else:
            selected_query = torch.zeros_like(input_ids, dtype=torch.bool)

        # replace image context token positions with bev embeddings
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = bev_embeds.reshape(-1, C).to(input_embeds.dtype)
            ignore_flag = False
        except Exception as e:
            vit_embeds = bev_embeds.reshape(-1, C)
            print(f'warning (img inject): {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = vit_embeds[:n_token].to(input_embeds.dtype)
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, -1, C).to(torch.bfloat16)

        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.hidden_states
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[-1]
        hidden_states = hidden_states.reshape(B, -1, C)

        hidden_bev = hidden_states[:, selected, :].to(torch.float32)
        text_emb = hidden_states[:, ~selected, :].to(torch.float32)
        if self.num_learnable_query > 0 and learn_query is not None:
            hidden_query = hidden_states[:, selected_query, :].to(torch.float32)
            logits = outputs.logits[:, ~selected_query, :]
            # ensure we align with text tokens length
            text_emb = text_emb[:, :logits.size(1), :]
        else:
            hidden_query = None
            logits = outputs.logits

        # chat loss (CE) only for pretraining/fine-tuning when labels provided
        loss_chat = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.llm.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss_chat = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss_chat = loss_chat * 0.0

        # auxiliary query regression loss
        loss_query = None
        if self.num_learnable_query > 0 and learn_query is not None and self.use_query_loss:
            hidden_query = hidden_query[:, :-self.num_learnable_query, :].contiguous()
            hidden_query = self.query_mlp(hidden_query)
            hidden_query_label = learn_query[:, self.num_learnable_query:, :].contiguous()
            loss_fnt = torch.nn.L1Loss()
            loss_query = loss_fnt(hidden_query, hidden_query_label)

        if self.num_learnable_query > 0 and learn_query is not None:
            hidden_query = self.out_mlp(hidden_query)
        else:
            hidden_query = None

        output_bev = self.out_mlp(hidden_bev).reshape(B, H, W, -1)
        if self.use_text_emb_fusion:
            text_emb = self.out_mlp(text_emb)

        return {
            'out_bev': output_bev,
            'logits': logits,
            'chat_loss': loss_chat,
            'out_query': hidden_query,
            'text_emb': text_emb,
            'query_loss': loss_query,
        }

    @torch.no_grad()
    def chat(self, tokenizer, pixel_values, question, conv=None, generation_config=None, history=None,
             return_history=False, num_patches_list=None,
             IMG_CONTEXT_TOKEN=IMG_CONTEXT_TOKEN, IMG_START_TOKEN=IMG_START_TOKEN, IMG_END_TOKEN=IMG_END_TOKEN,
             verbose=False):
        generation_config = generation_config or self.chat_cfg
        tokenizer = tokenizer or self.tokenizer
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template_name)
        sep = template.sep.strip() if template.sep2 is None else template.sep2.strip()
        eos_token_id = tokenizer.convert_tokens_to_ids(sep)
        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            # print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            self.num_image_token = self.img_length
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            if conv is not None:
                output = self(bev=pixel_values.clone().to(torch.float32), text=conv)
                return response, output
            else:
                return response, None

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None

        if pixel_values is not None:
            vit_embeds = pixel_values
            input_embeds = self.llm.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            _, H, W, _ = vit_embeds.shape
            vit_embeds = vit_embeds.reshape(B, H * W, -1)
            vit_embeds = self.in_mlp(vit_embeds).to(self.torch_dtype)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)
            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.llm.get_input_embeddings()(input_ids)
        outputs = self.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )
        return outputs

