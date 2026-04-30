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
from ...internvl.train.constants import (
    BOX_END_TOKEN, BOX_START_TOKEN,
    IMG_END_TOKEN, IMG_START_TOKEN, IMG_CONTEXT_TOKEN,
    QUAD_END_TOKEN, QUAD_START_TOKEN,
    REF_END_TOKEN, REF_START_TOKEN,
)
from ...internvl.train.dataset import (
    ConcatDataset, TCSLoader,
    build_transform, dynamic_preprocess,
    preprocess_pretrain, preprocess_internvl2_5, preprocess_internvl3_5_gpt_oss,
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


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={'help': 'Path to a pretrained model (local or from huggingface.co/models).'}
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the ViT. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP. Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'},
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'}
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'}
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of LLM. Default is False.'},
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={'help': 'Set to True to use gradient checkpointing. Default is True.'},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},
    )
    ps_version: Literal['v1', 'v2'] = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is v2.'}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'}
    )
    use_liger: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the liger kernel.'}
    )
    use_custom_flash_attn: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the custom flash attn.'}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """
    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True. Default is False.'},
    )
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    dynamic_image_size: bool = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic high resolution strategy. Default is False.'},
    )
    use_thumbnail: bool = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image. Default is False.'},
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video data. Default is 8.'},
    )
    max_num_frame: int = field(
        default=32,
        metadata={'help': 'The maximum number of frames for video data. Default is 32.'},
    )
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(
        default='imagenet',
        metadata={'help': 'The normalization type for the image. Default is imagenet.'},
    )
    use_packed_ds: bool = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for efficient training. Default is False.'},
    )
    num_images_expected: int = field(
        default=40,
        metadata={'help': 'The maximum number of images per packed sample. Default is 40.'},
    )
    max_packed_tokens: int = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: int = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: int = field(
        default=1000,
        metadata={'help': 'The log frequency of the packed dataset. Default is 1000.'},
    )
    strict_mode: bool = field(
        default=True,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: bool = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: bool = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    loss_reduction: str = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is token.'},
    )
    split_annotations: bool = field(
        default=False,
        metadata={'help': 'Whether to split annotations to save memory usage. Default is False.'},
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self,
            template_name,
            meta,
            tokenizer,
            tcs_loader,
            ds_name,
            num_image_token,
            image_size=448,
            is_train=True,
            pad2square=False,
            group_by_length=False,
            dynamic_image_size=False,
            use_thumbnail=False,
            min_dynamic_patch=1,
            max_dynamic_patch=12,
            min_num_frame=8,  # for video data
            max_num_frame=32,  # for video data
            sampling_method='rand',  # for video data
            repeat_time=1,
            normalize_type='imagenet',
            split_annotations=False,
            # hyperparameters for packed training
            use_packed_ds=False,
            data_rank=0,
            data_world_size=1,
            distributed_mode=False,
            force_shuffle=False,
            random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        # logger.info(f'[Dataset] num_image_token: {num_image_token}')
        # logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        # logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        # logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        # TODO: quick resume
        self._state_dict = {}

        # logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        self.split_annotations = split_annotations

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                repeat_time = int(repeat_time)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        if self.split_annotations:
            total_lines = len(self.raw_data)
            # logger.info(f'world_size: {self.world_size}, rank: {self.rank}, total_lines: {total_lines}')
            lines_per_rank = total_lines // self.world_size  # Number of lines each rank should process
            lines_per_rank = max(1, lines_per_rank)

            # Calculate the start and end line numbers for the current rank
            start_line = lines_per_rank * self.rank  # Starting line for the current rank
            end_line = start_line + lines_per_rank  # Ending line for the current rank

            # Assign the appropriate lines to the current rank
            self.raw_data = self.raw_data[start_line:end_line]

        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        self.num_fake_dump = 0

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self, use_pretrain=False):
        # Select the appropriate preprocessing function based on the template name
        if use_pretrain:
            return preprocess_pretrain

        if self.template_name == 'internvl2_5':
            return preprocess_internvl2_5

        if self.template_name == 'internvl3_5_gpt_oss':
            return preprocess_internvl3_5_gpt_oss

        raise NotImplementedError(f'Unsupported template: {self.template_name}')

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            return self.root + image_path
        return os.path.join(self.root, image_path)

    def get_transform(self):
        # Build transformation function
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )
        return transform

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        first_turn_idx = 1 if data_item['conversations'][0]['value'] == 'system' else 0
        if '<image>' not in data_item['conversations'][first_turn_idx]['value']:
            data_item['conversations'][first_turn_idx]['value'] = '<image>\n' + \
                                                                  data_item['conversations'][first_turn_idx]['value']

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        use_pretrain = (data_item['conversations'][first_turn_idx]['from'] == 'pretrain')
        preprocess_function = self.get_preprocess_function(use_pretrain=use_pretrain)

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][
                    0] == image_end_token_id).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        first_turn_idx = 1 if data_item['conversations'][0]['value'] == 'system' else 0
        if '<image>' not in data_item['conversations'][first_turn_idx]['value']:
            data_item['conversations'][first_turn_idx]['value'] = '<image>\n' * len(data_item['image']) + \
                                                                  data_item['conversations'][first_turn_idx]['value']

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=max(1, self.max_dynamic_patch // num_image),
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        use_pretrain = (data_item['conversations'][first_turn_idx]['from'] == 'pretrain')
        preprocess_function = self.get_preprocess_function(use_pretrain=use_pretrain)

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  ds_name=self.ds_name, num_image=num_image)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][
                    0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains a video placeholder
        first_turn_idx = 1 if data_item['conversations'][0]['value'] == 'system' else 0
        if '<video>' not in data_item['conversations'][first_turn_idx]['value']:
            data_item['conversations'][first_turn_idx]['value'] = '<video>\n' + \
                                                                  data_item['conversations'][first_turn_idx]['value']

        # Get the video file path
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)

        # Load the video frames using tcs_loader
        # TODO: Load videos without using tcsloader.
        image_list = self.tcs_loader(
            video_path,
            image_type='video',
            max_num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=data_item.get('clip', None))

        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][first_turn_idx]['value'] = data_item['conversations'][first_turn_idx][
            'value'].replace(
            '<video>\n', special_tokens + '\n')

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        use_pretrain = (data_item['conversations'][first_turn_idx]['from'] == 'pretrain')
        preprocess_function = self.get_preprocess_function(use_pretrain=use_pretrain)

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  ds_name=self.ds_name, num_image=num_patches)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        use_pretrain = (data_item['conversations'][0]['from'] == 'pretrain')
        preprocess_function = self.get_preprocess_function(use_pretrain=use_pretrain)

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def fake_data_get_item(self):
        # Build transformation function
        self.num_fake_dump += 1
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function(use_pretrain=True)

        conversations = [
            {"from": "pretrain",
             "value": '我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'},
        ]

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(conversations)],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=torch.ones_like(ret['input_ids'][0]) * -100,
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        logger.warning(f'Dumping a fake data, the dataset is: {self.ds_name} ({self.num_fake_dump})')
        return ret

    def _enable_worker_distributed(self):
        if (
                self.distributed_mode
                and not self.worker_distributed
                and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                if self.use_packed_ds:
                    raise StopIteration
                return self.fake_data_get_item()
            try:
                data_item = json.loads(self.raw_data[i])
                # conversations = data_item['conversations']
                # check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10)
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                # print(e, self.ds_name, flush=True)
                # if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                #     traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=} (total={len(self)})'
            )

        for i in range(start_idx, len(self)):
            yield self[i]


def build_datasets(
        data_args,
        tokenizer,
        tcs_loader,
        model,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        min_num_frame=8,
        max_num_frame=32,
        normalize_type='imagenet',
        split_annotations=False,
):
    datasets = []
    lengths = []
    data_rank = 0 if split_annotations else dist.get_rank()
    data_world_size = 1 if split_annotations else dist.get_world_size()
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style, ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name].get('data_augment', False),
            pad2square=data_args.pad2square,
            group_by_length=group_by_length and not data_args.use_packed_ds,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            min_num_frame=min_num_frame,
            max_num_frame=max_num_frame,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            split_annotations=split_annotations,
            # hyperparameters for packed training
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)} ({data_rank=}, {data_world_size=})')
        datasets.append(dataset)
        lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
        )
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x ** 0.5)
    raise NotImplementedError(loss_reduction)


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
class HERMESLLMV2(nn.Module):
    """
    HERMESLLM adapted for InternVL3.5 (GPT-OSS backbone).
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
    ):
        super().__init__()

        assert model_path is not None, 'model_path must be provided for InternVL3.5 model.'
        self.model_path = model_path
        self.tokenizer_path = model_path

        # InternVL3.5 template and preprocess
        self.template_name = 'internvl2_5'

        if self.template_name == 'internvl2_5':
            self.preprocess_func = preprocess_internvl2_5

        if self.template_name == 'internvl3_5_gpt_oss':  # do not support for HERMESLLM
            self.preprocess_func = preprocess_internvl3_5_gpt_oss

        # Default LoRA target modules for Qwen/GPT family
        self.target_modules = [
            'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
            'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
        ]

        self.ds_name = 'internvl3_5'
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
        """Load InternVL3.5 chat model and extract its LLM; build projection MLPs and configure LoRA."""
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            add_eos_token=False,
            trust_remote_code=True,
            use_fast=False,
        )
        self.tokenizer.tokenizer_path = self.tokenizer_path
        self.tokenizer.model_max_length = 8192
        token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                      QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                      REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
        self.num_new_tokens = self.tokenizer.add_tokens(token_list, special_tokens=True)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        # config
        config = InternVLChatConfig.from_pretrained(self.model_path)
        config.vision_config.drop_path_rate = 0.0
        config.template = self.template_name
        config.select_layer = -1
        config.dynamic_image_size = True
        config.use_thumbnail = True
        config.ps_version = 'v2'
        config.min_dynamic_patch = 1
        config.max_dynamic_patch = 6

        # GPT-OSS specifics
        if getattr(config, 'llm_config', None) is not None and getattr(config.llm_config, 'model_type',
                                                                       '') == 'gpt_oss':
            config.llm_config.output_router_logits = True
            # keep default attention; runtime patches can swap to flash/sink attention if needed
        else:
            # non gpt_oss fallback
            try:
                config.llm_config._attn_implementation = 'flash_attention_2'
            except Exception:
                pass

        # build model, keep only language model
        if self.load_weight:
            model = InternVLChatModel.from_pretrained(self.model_path, torch_dtype=self.torch_dtype, config=config)
        else:
            model = InternVLChatModel._from_config(config).to(self.torch_dtype)

        model.language_model.config.use_cache = False
        self.llm = model.language_model

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
            if self.lora:
                warnings.warn('Using LoRA during pretraining is not recommended; enabling as requested.')
                self.set_lora()
        else:
            if self.lora:
                self.set_lora()

            else:
                # fully finetune
                self.set_lora(freeze_llm=False, use_llm_lora=0)

        # free vision/backbone wrappers we don't use here
        del model
        torch.cuda.empty_cache()

    def set_lora(self,
                 freeze_llm=True,
                 unfreeze_lm_head=False,
                 use_llm_lora=128,
                 freeze_mlp=True,
                 ):
        if freeze_llm:
            self.llm = self.llm.eval()
            _freeze_params(self.llm)
        if unfreeze_lm_head:
            self.llm.lm_head.requires_grad = True
        if use_llm_lora:
            self.llm = wrap_llm_lora(self.llm, r=use_llm_lora, lora_alpha=2 * use_llm_lora, target_modules=self.target_modules)

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
        except Exception:
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
        if labels is not None and self.is_pretraining:
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
             IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False):
        generation_config = generation_config or self.chat_cfg
        tokenizer = tokenizer or self.tokenizer
        history = [] if history is None else history
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


def main():
    replace_train_dataloader()

    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.use_packed_ds = data_args.use_packed_ds

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training/evaluation parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=model_args.use_fast_tokenizer,
    )
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = TCSLoader('petreloss.conf') if use_tcs_loader else None

    if model_args.use_liger:
        raise NotImplementedError

    if model_args.model_name_or_path is not None:
        logger.info('Loading InternVLChatModel...')
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate

        if config.llm_config.model_type == 'gpt_oss':
            config.llm_config.output_router_logits = True
            logger.info('Using vanilla attention for GptOssForCausalLM')

            # config.llm_config._attn_implementation = 'kernels-community/vllm-flash-attn3'
            # logger.info('Using kernels-community/vllm-flash-attn3 for GptOssForCausalLM')
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'
            logger.info('Using flash_attention_2')

        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        model = InternVLChatModel.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16,
                                                  config=config)
    else:
        logger.info('Loading ViT-6B...')
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        vision_model = InternVisionModel.from_pretrained(model_args.vision_path, torch_dtype=torch.bfloat16,
                                                         config=vision_config)

        logger.info('Loading LLM...')
        llm_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)

        if llm_config.model_type == 'gpt_oss':
            llm_config.output_router_logits = True
            logger.info('Using vanilla attention for GptOssForCausalLM')

            # llm_config._attn_implementation = 'kernels-community/vllm-flash-attn3'
            # logger.info('Using kernels-community/vllm-flash-attn3 for GptOssForCausalLM')
        else:
            llm_config._attn_implementation = 'flash_attention_2'
            logger.info('Using flash_attention_2 for LLaMA')

        llm = AutoModelForCausalLM.from_pretrained(model_args.llm_path, torch_dtype=torch.bfloat16, config=llm_config,
                                                   trust_remote_code=True)

        logger.info('Building InternVLChatConfig...')
        internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(),
            llm_config.to_dict(),
            downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square,
            template=data_args.conv_style,
            select_layer=model_args.vision_select_layer,
            dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail,
            ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch,
            max_dynamic_patch=data_args.max_dynamic_patch,
        )
        internvl_chat_config.force_image_size = data_args.force_image_size

        logger.info('Building InternVLChatModel...')
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)

    model.img_context_token_id = img_context_token_id
    model.tokenizer = tokenizer

    if model_args.use_custom_flash_attn:
        replace_gpt_oss_with_flash_sink_attn(model.language_model, use_varlen=data_args.use_packed_ds)
    elif data_args.use_packed_ds:
        replace_qwen3_attention_class(model.language_model)

    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info('Finished')

    patch_size = model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {model.config.vision_config.image_size}')
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from '
                    f'{model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size,
                                                 patch_size=patch_size)
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    if model_args.grad_checkpoint:
        model.vision_model.gradient_checkpointing = True
        model.vision_model.encoder.gradient_checkpointing = True
        model.language_model._set_gradient_checkpointing()
        logger.info('gradient_checkpointing is enabled')

    train_dataset = build_datasets(
        data_args,
        tokenizer,
        tcs_loader,
        model,
        group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size,
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type,
        min_num_frame=data_args.min_num_frame,
        max_num_frame=data_args.max_num_frame,
        split_annotations=data_args.split_annotations,
    )

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    if data_args.use_packed_ds:
        collator = partial(
            packed_collate_fn,
            data_collator=concat_pad_data_collator,
            max_item_length=data_args.max_packed_tokens if data_args.strict_mode else 0,
            micro_num=training_args.train_batch_size,
            len2weight=partial(len2weight, loss_reduction=data_args.loss_reduction),
        )
    else:
        collator = concat_pad_data_collator

    training_args.split_annotations = data_args.split_annotations
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        logger.info(f'[Memory Usage before training] {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f}GB')
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()


if __name__ == '__main__':
    main()
