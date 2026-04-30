import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .down_util.down_sample_cross import DownSampleCross

from mmdet.models import HEADS
from mmcv.runner import auto_fp16
from mmcv.cnn import xavier_init
from .render_utils.rays import RayBundle
from .render_head_bevformer import BEVFormerRenderHead
import numpy as np
import torch.nn.functional as F
from typing import Optional

try:
    from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func  # type: ignore
except Exception:
    try:
        from flash_attn import flash_attn_func as _flash_attn_func  # type: ignore
    except Exception:
        _flash_attn_func = None

def _cfg_get(cfg, *keys, default=None):
    for key in keys:
        if key is None:
            continue
        value = cfg.get(key, None) if hasattr(cfg, "get") else getattr(cfg, key, None)
        if value is not None:
            return value
    return default


@HEADS.register_module()
class HERMESFutureRenderHead(BEVFormerRenderHead):
    def __init__(self,
                 *args,
                 loss_weight=None,
                 use_can_bus=False,
                 llm_cfg=None,
                 use_atten_pool=False,
                 use_max_pool=False,
                 downsample_scale=2,
                 num_ego_layers=3,
                 future_generate=True,
                 pred_can_bus=False,
                 lidar_second_alignment=False,
                 enable_gram_global_loss=False,
                 gram_global_mode='avg_zw_zh_wh',
                 gram_token_normalize=True,
                 **kwargs):
        super(HERMESFutureRenderHead, self).__init__(*args, **kwargs)

        self.loss_weight = loss_weight
        self.use_atten_pool = use_atten_pool
        self.use_max_pool = use_max_pool
        self.future_generate = future_generate
        self.bev_channels = self.embed_dims * (2 ** downsample_scale)
        self.llm_cfg = llm_cfg
        self.chat_cfg = llm_cfg.chat_config
        self.use_curr_query = self.llm_cfg.get("use_curr_query", False)
        self.lidar_second_alignment = lidar_second_alignment
        self.enable_gram_global_loss = enable_gram_global_loss
        assert gram_global_mode == "avg_zw_zh_wh", "Only avg_zw_zh_wh is kept in the open-source version."
        self.gram_global_mode = gram_global_mode
        self.gram_token_normalize = gram_token_normalize

        self.init_llm()
        starting_index = 0 if self.use_curr_query else 1
        if hasattr(self.llm_cfg, "num_learnable_query") and self.llm_cfg.num_learnable_query > 0:
            self.num_learnable_query = self.llm_cfg.num_learnable_query
            self.pool = nn.AdaptiveMaxPool1d(self.num_learnable_query)

            self.query_pos = nn.Parameter(
                torch.randn(self.num_learnable_query * (len(self.frames) - starting_index), self.bev_channels) / (
                    self.bev_channels) ** 0.5) if len(self.frames) > 1 else nn.Parameter(
                torch.randn(self.num_learnable_query * 3, self.bev_channels) / (self.bev_channels) ** 0.5)
        else:
            self.num_learnable_query = 0
            self.query_embedding = None
        self.frame_embedding = nn.Parameter(
            torch.randn([len(self.frames), self.bev_channels]) / (self.bev_channels) ** 0.5) if len(
            self.frames) > 1 else None

        self.use_can_bus = use_can_bus
        self.pred_can_bus = pred_can_bus
        if self.use_can_bus:
            self.can_bus_mlp = nn.Sequential(
                nn.Linear(18, self.embed_dims * 2),
                nn.LayerNorm(self.embed_dims * 2),
                nn.LeakyReLU(inplace=True),
                nn.Linear(self.embed_dims * 2, self.bev_channels),
                nn.LayerNorm(self.bev_channels),
                nn.LeakyReLU(inplace=True),
            )
            xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

        curr2future_link_cfg = self._build_curr2future_link_cfg()
        self.use_curr2future_link = curr2future_link_cfg is not None
        self.use_ego_attention_v2 = self.use_curr2future_link
        if self.use_curr2future_link:
            self.curr2future_link = Curr2FutureLink(
                embed_dim=self.bev_channels,
                num_heads=curr2future_link_cfg["num_heads"],
                num_layers=curr2future_link_cfg["num_layers"],
                text_dim=curr2future_link_cfg["text_dim"],
                text_pool_k=curr2future_link_cfg["text_pool_k"],
                use_flash_attn=curr2future_link_cfg["use_flash_attn"],
                ffn_ratio=curr2future_link_cfg["ffn_ratio"],
                use_text=curr2future_link_cfg["use_text"],
                use_global_query=curr2future_link_cfg["use_global_query"],
                use_canbus_adaln=curr2future_link_cfg["use_canbus_adaln"],
                cross_first=curr2future_link_cfg["cross_first"],
                canbus_dropout_p=curr2future_link_cfg["canbus_dropout_p"],
                text_dropout_p=curr2future_link_cfg["text_dropout_p"],
            )
            self.ego_attention_v2 = self.curr2future_link
        else:
            self.curr2future_link = None

        if self.use_downsample:
            use_ego_attention = False if num_ego_layers == 0 or self.num_learnable_query == 0 or self.use_curr2future_link else True
            self.down_sample = DownSampleCross(self.in_channels, num_layers=downsample_scale,
                                               use_ego_attention=use_ego_attention, num_ego_layers=num_ego_layers)

    def _build_curr2future_link_cfg(self):
        curr2future_link_cfg = _cfg_get(self.llm_cfg, "curr2future_link_cfg", default=None)
        enabled = _cfg_get(
            self.llm_cfg,
            "use_curr2future_link",
            "use_ego_attention_v2",
            default=curr2future_link_cfg is not None,
        )
        if not enabled:
            return None

        curr2future_link_cfg = curr2future_link_cfg or {}
        return dict(
            num_layers=int(_cfg_get(curr2future_link_cfg, "num_layers", default=_cfg_get(self.llm_cfg, "curr2future_link_layers", "ego_attn_v2_layers", default=2))),
            num_heads=int(_cfg_get(curr2future_link_cfg, "num_heads", default=_cfg_get(self.llm_cfg, "curr2future_link_num_heads", "ego_attn_v2_num_heads", default=16))),
            text_pool_k=int(_cfg_get(curr2future_link_cfg, "text_pool_k", default=_cfg_get(self.llm_cfg, "curr2future_link_text_pool_k", "ego_attn_v2_text_pool_k", default=8))),
            text_dim=int(_cfg_get(curr2future_link_cfg, "text_dim", default=_cfg_get(self.llm_cfg, "curr2future_link_text_dim", "ego_attn_v2_text_dim", default=2048))),
            use_flash_attn=bool(_cfg_get(curr2future_link_cfg, "use_flash_attn", default=_cfg_get(self.llm_cfg, "curr2future_link_use_flash_attn", "ego_attn_v2_use_flash_attn", default=True))),
            ffn_ratio=float(_cfg_get(curr2future_link_cfg, "mlp_ratio", default=_cfg_get(self.llm_cfg, "curr2future_link_mlp_ratio", "ego_attn_v2_mlp_ratio", default=4.0))),
            cross_first=bool(_cfg_get(curr2future_link_cfg, "cross_first", default=_cfg_get(self.llm_cfg, "curr2future_link_cross_first", "ego_attn_v2_cross_first", default=False))),
            use_text=bool(_cfg_get(curr2future_link_cfg, "use_text", default=_cfg_get(self.llm_cfg, "curr2future_link_use_text", "ego_attn_v2_use_text", default=True))),
            use_global_query=bool(_cfg_get(curr2future_link_cfg, "use_global_query", default=_cfg_get(self.llm_cfg, "curr2future_link_use_global_query", "ego_attn_v2_use_global_query", default=True))),
            use_canbus_adaln=bool(_cfg_get(curr2future_link_cfg, "use_canbus_adaln", "use_can_bus_adaln", default=_cfg_get(self.llm_cfg, "curr2future_link_use_canbus_adaln", "ego_attn_v2_use_canbus_adaln", default=False))),
            canbus_dropout_p=float(_cfg_get(curr2future_link_cfg, "canbus_dropout_p", default=_cfg_get(self.llm_cfg, "curr2future_link_canbus_dropout_p", "ego_attn_v2_canbus_dropout_p", default=0.0))),
            text_dropout_p=float(_cfg_get(curr2future_link_cfg, "text_dropout_p", default=_cfg_get(self.llm_cfg, "curr2future_link_text_dropout_p", "ego_attn_v2_text_dropout_p", default=0.0))),
        )

    def loss(self, preds_dict, targets):
        lidar_targets, _ = targets
        batch_size = len(lidar_targets)
        loss_dict = {}
        for bs_idx in range(batch_size):
            if hasattr(self, "cur_iter"):
                self.render_model.cur_iter = self.cur_iter
            i_loss_dict = self.render_model.loss(preds_dict[bs_idx], lidar_targets[bs_idx], None)
            if "lidar_sim_loss" in preds_dict[bs_idx]:
                if preds_dict[bs_idx]["lidar_sim_loss"] is not None:
                    i_loss_dict['lidar_sim_loss'] = preds_dict[bs_idx]["lidar_sim_loss"]
            if "gram_global_loss" in preds_dict[bs_idx]:
                if preds_dict[bs_idx]["gram_global_loss"] is not None:
                    i_loss_dict['gram_global_loss'] = preds_dict[bs_idx]["gram_global_loss"]
            for k, v in i_loss_dict.items():
                if k not in loss_dict:
                    loss_dict[k] = []
                loss_dict[k].append(v)
        for k, v in loss_dict.items():
            loss_dict[k] = torch.stack(v, dim=0).mean()
        return loss_dict

    def init_llm(self):
        from ..internvl_chat.internvl.train.internlm2_chat import HERMESLLM

        model_map = {
            'HERMESLLM': HERMESLLM,
        }
        model_type_name = self.llm_cfg.get("model_type", "HERMESLLM")
        model_type = model_map.get(model_type_name)
        assert model_type is not None, f"Model type {model_type_name} is not supported. " \
                                       f"Supported types are: {list(model_map.keys())}"

        llm_kwargs = {
            'model_path': self.llm_cfg.llm_path,
            'load_weight': self.llm_cfg.load_internvl_weight,
            'set_lora': self.llm_cfg.set_lora,
            'is_pretraining': self.llm_cfg.is_pretraining,
            'chat_cfg': self.chat_cfg,
            'attention_type': self.llm_cfg.attention_type,
            'img_length': self.llm_cfg.img_length,
            'num_learnable_query': self.llm_cfg.get("num_learnable_query", 0),
            'input_dim': self.bev_channels,
            'use_query_loss': self.llm_cfg.get("use_curr_query", False),
            'use_text_emb_fusion': self.llm_cfg.get("use_text_emb_fusion", False),
        }
        self.llm = model_type(**llm_kwargs)
        self.llm.create_llm()

    def _init_layers(self):
        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)

    def init_weights(self):
        self.transformer.init_weights()

    @staticmethod
    def _gram_matrix(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = x.float()
        G = torch.bmm(x, x.transpose(1, 2))
        return G

    def _compute_gram_global_loss(self, uni_feats: torch.Tensor, pts_feats: torch.Tensor) -> torch.Tensor:
        token_norm = self.gram_token_normalize

        if uni_feats.size(-1) != pts_feats.size(-1):
            uni_feats = F.interpolate(uni_feats, size=pts_feats.shape[2:], mode='trilinear',
                                      align_corners=False)

        def _gram_loss_from_tokens(u_tokens: torch.Tensor, p_tokens: torch.Tensor) -> torch.Tensor:
            u_tokens = u_tokens.float()
            p_tokens = p_tokens.float()
            if token_norm:
                u_tokens = F.normalize(u_tokens, dim=-1, eps=1e-6)
                p_tokens = F.normalize(p_tokens, dim=-1, eps=1e-6)
            G_u = self._gram_matrix(u_tokens)
            G_p = self._gram_matrix(p_tokens)
            return ((G_u - G_p) ** 2).mean(dim=(1, 2))

        u_h = uni_feats.mean(dim=(2, 4)).permute(0, 2, 1)
        p_h = pts_feats.mean(dim=(2, 4)).permute(0, 2, 1)
        l_h = _gram_loss_from_tokens(u_h, p_h)

        u_w = uni_feats.mean(dim=(2, 3)).permute(0, 2, 1)
        p_w = pts_feats.mean(dim=(2, 3)).permute(0, 2, 1)
        l_w = _gram_loss_from_tokens(u_w, p_w)

        u_z = uni_feats.mean(dim=(3, 4)).permute(0, 2, 1)
        p_z = pts_feats.mean(dim=(3, 4)).permute(0, 2, 1)
        l_z = _gram_loss_from_tokens(u_z, p_z)

        return l_h + l_w + l_z

    @auto_fp16(apply_to=("pts_feats", "img_feats", "img_depth"))
    def forward(self, pts_feats, rays, img_metas, img_depth, input_bev_embed=None, prev_bev_down=None,
                prev_bev=None, gt_bev=None,
                only_bev=False,
                fusion_prev=False, prev_img_metas=None,
                conv=None, llm_pred_bev=None, img=None):
        bs = 1
        dtype = torch.float16
        B, C, H, W = input_bev_embed.shape
        assert len(self.frames) == len(rays), "frame number should be equal to rays number"
        len_future = len(self.frames) if len(self.frames) > 1 else 3

        can_bus = None
        if self.use_can_bus:
            can_bus = torch.as_tensor(
                np.stack([img_metas[self.frames[i]]["can_bus"] for i in range(len_future)], axis=0),
                device=input_bev_embed.device, dtype=torch.float32,
            )
            canbus_drop_p = float(self.llm_cfg.get("ego_attn_v2_canbus_dropout_p", 0.0))
            if self.training and canbus_drop_p > 0.0:
                if torch.rand(1, device=input_bev_embed.device).item() < canbus_drop_p:
                    can_bus.zero_()
                    for i in range(len_future):
                        z = np.zeros_like(img_metas[self.frames[i]]["can_bus"],
                                          dtype=img_metas[self.frames[i]]["can_bus"].dtype)
                        img_metas[self.frames[i]]["can_bus"] = z

        if self.use_downsample or hasattr(self, 'llm'):
            input_bev_embed = input_bev_embed.permute(0, 2, 3, 1).to(self.llm.torch_dtype)
            query_embedding = None
            starting_index = 0 if self.use_curr_query else 1
            if self.num_learnable_query > 0:
                with torch.cuda.amp.autocast():
                    q = self.pool(input_bev_embed.view(B, -1, C).permute(0, 2, 1)).permute(0, 2, 1)
                    query_embedding = q.repeat(1, len_future, 1).contiguous()

                    if self.use_can_bus:
                        num_sel = len_future - starting_index
                        frame_emb = self.frame_embedding[starting_index:len_future].view(1, num_sel, 1, C)
                        can_bus = torch.as_tensor(
                            np.stack(
                                [img_metas[self.frames[i]]["can_bus"] for i in range(len_future)],
                                axis=0
                            ),
                            device=input_bev_embed.device,
                            dtype=torch.float32
                        )
                        can_bus_emb = self.can_bus_mlp(can_bus[starting_index:]
                                                       ).view(1, num_sel, 1, C)
                        query_embedding = query_embedding.view(B, len_future, self.num_learnable_query, C)[:,
                        starting_index:, :, :] + frame_emb + can_bus_emb
                        query_embedding = query_embedding.reshape(B, num_sel * self.num_learnable_query, C).to(
                            self.llm.torch_dtype)

                query_embedding = (query_embedding + self.query_pos).to(self.llm.torch_dtype)

            llm_out = checkpoint(self.llm, input_bev_embed, conv, query_embedding, use_reentrant=False)

            llm_bev_embed = llm_out['out_bev']
            if self.training:
                self.loss_chat = llm_out['chat_loss']
            out_query = llm_out['out_query']

            if out_query is not None:
                with ((torch.cuda.amp.autocast())):
                    out_query = out_query.view(-1, self.num_learnable_query, self.bev_channels)
                    if self.future_generate:
                        assert len(self.frames) > 1
                        bev_embed_down = llm_bev_embed.expand(len(self.frames), -1, -1, -1)
                        if self.use_can_bus:
                            bev_embed_down = bev_embed_down + self.frame_embedding[:, None, None, :] \
                                             + self.can_bus_mlp(can_bus)[:, None, None, :]
                        if self.use_curr2future_link:
                            bev_embed_down_ = self.curr2future_link(
                                bev_embed_down[-out_query.shape[0]:, ...],
                                out_query,
                                llm_out.get('text_emb', None),
                                can_bus[starting_index:] if can_bus is not None else None,
                            )
                        else:
                            bev_embed_down_ = self.down_sample.ego_attention(
                                bev_embed_down[-out_query.shape[0]:, ...].permute(0, 3, 1, 2), out_query)
                        bev_embed_down = torch.cat([bev_embed_down[:1], bev_embed_down_], dim=0)
                    else:
                        bev_embed_down = llm_bev_embed
            else:
                if len(self.frames) > 1:
                    with torch.cuda.amp.autocast():
                        bev_embed_down = llm_bev_embed.expand(len(self.frames), -1, -1, -1)
                        if self.use_can_bus:
                            bev_embed_down = bev_embed_down + self.frame_embedding[:, None, None, :] + self.can_bus_mlp(
                                can_bus)[:, None, None, :]
                        else:
                            bev_embed_down = bev_embed_down + self.frame_embedding[:, None, None, :]
                else:
                    bev_embed_down = llm_bev_embed

            if hasattr(self, 'down_sample'):
                uni_feats = checkpoint(self.down_sample.up_sample, bev_embed_down.permute(0, 3, 1, 2))
                if uni_feats.size(-1) != self.bev_w:
                    uni_feats = F.interpolate(
                        uni_feats, (self.bev_h, self.bev_w),
                        mode="bilinear", align_corners=False
                    )
            else:
                uni_feats = bev_embed_down.permute(0, 3, 1, 2)
            uni_feats = checkpoint(self.bev_upsample, uni_feats)

            if self.training and self.lidar_second_alignment:
                if uni_feats.shape[-1] != pts_feats.shape[-1]:
                    sim_loss = 1 - F.cosine_similarity(
                        F.interpolate(uni_feats, size=pts_feats.shape[2:], mode='trilinear',
                                      align_corners=False).float(),
                        pts_feats.float(), dim=1).reshape(len(self.frames), -1).mean(dim=1)
                else:
                    sim_loss = 1 - F.cosine_similarity(uni_feats.float(), pts_feats.float(), dim=1).reshape(
                        len(self.frames), -1).mean(dim=1)
            else:
                sim_loss = None

            if self.training and self.enable_gram_global_loss:
                gram_global_loss = checkpoint(self._compute_gram_global_loss, uni_feats,
                                              pts_feats)
            else:
                gram_global_loss = None

            uni_feats = checkpoint(self.render_conv, uni_feats)
            uni_feats_dict = {}
            for i in range(len(self.frames)):
                uni_feats_dict[self.frames[i]] = uni_feats[i]

        else:
            uni_feats = self.get_uni_feats(img_metas, input_bev_embed, only_bev, bs, dtype, fusion_prev=fusion_prev)
            if only_bev:
                return uni_feats

        batch_ret = []
        for bs_idx in range(bs):
            for idx, frame_idx in enumerate(self.frames):
                lidar_rays, cam_rays = rays[frame_idx]
                i_ray_o, i_ray_d, i_ray_depth = (
                    lidar_rays[bs_idx]["ray_o"],
                    lidar_rays[bs_idx]["ray_d"],
                    lidar_rays[bs_idx].get("depth", None),
                )
                if self.training:
                    scaled_points = lidar_rays[bs_idx]["scaled_points"]
                    lidar_ray_bundle = RayBundle(
                        origins=i_ray_o, directions=i_ray_d, depths=i_ray_depth
                    )
                    preds_dict = self.render_model(
                        lidar_ray_bundle, None, uni_feats_dict[frame_idx], points=scaled_points
                    )
                    preds_dict['lidar_sim_loss'] = sim_loss[idx] if sim_loss is not None else None
                    preds_dict['gram_global_loss'] = gram_global_loss[idx] if gram_global_loss is not None else None
                else:
                    lidar_ray_bundle = RayBundle(
                        origins=i_ray_o, directions=i_ray_d, depths=None
                    )
                    preds_dict = self.render_model(
                        lidar_ray_bundle, None, uni_feats_dict[frame_idx]
                    )

                batch_ret.append(preds_dict)

        return batch_ret

    def sample_rays(self, pts, imgs, img_metas):

        lidar_ret = self.sample_lidar_rays(pts, img_metas)
        return lidar_ret, None

    def sample_rays_test(self, pts, imgs, img_metas):
        lidar_ret = self.sample_lidar_rays(pts, img_metas, test=True)
        return lidar_ret, None

    def sample_lidar_rays(self, pts, img_metas, test=False):
        lidar_ret = []
        for i in range(len(pts)):
            lidar_pc = pts[i]
            dis = torch.norm(lidar_pc[:, :3], p=2, dim=-1)
            dis_mask = (dis > self.ray_sampler_cfg.close_radius) & (
                    dis < self.ray_sampler_cfg.get("far_radius", 100.0)
            )
            lidar_pc = lidar_pc[dis_mask]
            lidar_points = lidar_pc[:, :3]
            lidar_origins = torch.zeros_like(lidar_points)
            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = torch.norm(lidar_directions, dim=-1, keepdim=True)
            lidar_directions = lidar_directions / lidar_ranges
            lidar_ret.append(
                {
                    "ray_o": lidar_origins * self.render_model.scale_factor,
                    "ray_d": lidar_directions,
                    "depth": lidar_ranges * self.render_model.scale_factor,
                    "scaled_points": lidar_points * self.render_model.scale_factor,
                    "scale_factor": self.render_model.scale_factor,
                }
            )
        return lidar_ret

    def get_uni_feats(self, img_feats, img_metas, prev_bev, only_bev, bs, dtype, fusion_prev=False):
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_embed = self.transformer.get_bev_features(
            img_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h,
                         self.real_w / self.bev_w),
            bev_pos=bev_pos,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )

        if only_bev is not None and only_bev:
            return bev_embed

        bev_embed = bev_embed.permute(0, 2, 1).contiguous().view(bs, -1, self.bev_h,
                                                                 self.bev_w)

        if self.use_downsample:
            bev_embed_ori, bev_embed_recon = self.down_sample(bev_embed)
            bev_embed = bev_embed_recon.clone()

        bev_embed = self.bev_upsample(bev_embed)

        uni_feats = self.render_conv(bev_embed)
        if self.use_downsample:
            return uni_feats, bev_embed_ori, bev_embed_recon
        else:
            return uni_feats


class SDPAAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0,
                 qkv_bias: bool = True, use_flash_attn: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_flash_attn = use_flash_attn
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def _split_heads(self, x: torch.Tensor):
        B, N, C = x.shape
        return x.view(B, N, self.num_heads, self.head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, Nq, C = q.shape
        q_lin = self.q_proj(q)
        k_lin = self.k_proj(k)
        v_lin = self.v_proj(v)
        use_flash = (
                self.use_flash_attn and _flash_attn_func is not None and q_lin.is_cuda and
                q_lin.dtype in (torch.float16, torch.bfloat16) and (self.head_dim % 8 == 0)
        )
        if use_flash:
            q_fa = self._split_heads(q_lin).contiguous()
            k_fa = self._split_heads(k_lin).contiguous()
            v_fa = self._split_heads(v_lin).contiguous()
            p = self.attn_drop if self.training else 0.0
            try:
                x = _flash_attn_func(q_fa, k_fa, v_fa, dropout_p=p, softmax_scale=None, causal=False)
            except Exception:
                use_flash = False
        if not use_flash:
            q_sp = self._split_heads(q_lin).transpose(1, 2)
            k_sp = self._split_heads(k_lin).transpose(1, 2)
            v_sp = self._split_heads(v_lin).transpose(1, 2)
            x = F.scaled_dot_product_attention(q_sp, k_sp, v_sp, dropout_p=self.attn_drop if self.training else 0.0)
            x = x.transpose(1, 2)

        x = x.reshape(B, Nq, C)
        x = self.proj_drop(self.proj(x))
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim: int, ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Curr2FutureLinkBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_ratio: float = 4.0, attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 use_flash_attn: bool = True, use_canbus_adaln: bool = False, cross_first: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = SDPAAttention(embed_dim, num_heads, attn_drop, proj_drop, use_flash_attn=use_flash_attn)

        self.norm2_q = nn.LayerNorm(embed_dim)
        self.norm2_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = SDPAAttention(embed_dim, num_heads, attn_drop, proj_drop, use_flash_attn=use_flash_attn)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = MLP(embed_dim, ratio=ffn_ratio, drop=proj_drop)
        self.use_canbus_adaln = use_canbus_adaln
        self.cross_first = cross_first
        if self.use_canbus_adaln:
            self.modulation = nn.Parameter(torch.randn(1, 4, embed_dim) / embed_dim ** 0.5)

    def _self_attn_step(self, x: torch.Tensor, mem: torch.Tensor, style_params):
        norm_x = self.norm1(x)
        if self.use_canbus_adaln and style_params is not None:
            assert style_params[0].dtype == torch.float32
            with torch.amp.autocast('cuda', dtype=torch.float32):
                norm_x = norm_x.float() * (1 + style_params[0]) + style_params[1]
        x = x + self.self_attn(norm_x, norm_x, norm_x)
        return x

    def _cross_attn_step(self, x: torch.Tensor, mem: torch.Tensor):
        mem = self.norm2_kv(mem)
        return x + self.cross_attn(self.norm2_q(x), mem, mem)

    def _ffn_step(self, x: torch.Tensor, style_params):
        if self.use_canbus_adaln and style_params is not None:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + self.ffn(
                    self.norm3(x).float() * (1 + style_params[2]) + style_params[3])
        else:
            x = x + self.ffn(self.norm3(x))
        return x

    def forward(self, x: torch.Tensor, mem: torch.Tensor, style_params=None) -> torch.Tensor:

        if self.use_canbus_adaln and style_params is not None:
            assert style_params.dtype == torch.float32
            with torch.amp.autocast('cuda', dtype=torch.float32):
                style_params = (self.modulation + style_params).chunk(4, dim=1)
                assert style_params[0].dtype == torch.float32

        if not self.cross_first:
            x = self._self_attn_step(x, mem, style_params)
            x = self._cross_attn_step(x, mem)
        else:
            x = self._cross_attn_step(x, mem)
            x = self._self_attn_step(x, mem, style_params)
        x = self._ffn_step(x, style_params)
        return x


class TextProjector(nn.Module):
    def __init__(self, text_dim: int, embed_dim: int, k: int):
        super().__init__()
        self.k = k
        self.ln = nn.LayerNorm(text_dim)
        self.fc1 = nn.Linear(text_dim, embed_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(self.k)

    def forward(self, text_emb: torch.Tensor, target_B: int) -> torch.Tensor:
        with torch.amp.autocast('cuda', dtype=torch.float32):
            t = self.ln(text_emb)
            t = self.pool(self.act(self.fc1(t)).transpose(1, 2)).transpose(1, 2)
            t = self.fc2(t)
        return t.expand(target_B, -1, -1).contiguous()


class Curr2FutureLink(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, text_dim: int, text_pool_k: int,
                 use_flash_attn: bool = True, ffn_ratio: float = 4.0, use_text: bool = True,
                 use_global_query: bool = True, use_canbus_adaln: bool = False, cross_first: bool = False,
                 canbus_dropout_p: float = 0.0, text_dropout_p: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_text = use_text
        self.use_global_query = use_global_query
        self.use_canbus_adaln = use_canbus_adaln
        self.cross_first = cross_first
        self.canbus_dropout_p = float(canbus_dropout_p)
        self.text_dropout_p = float(text_dropout_p)
        if self.use_text:
            self.text_proj = TextProjector(text_dim=text_dim, embed_dim=embed_dim, k=text_pool_k)
        else:
            self.text_proj = None
        self.layers = nn.ModuleList([
            Curr2FutureLinkBlock(embed_dim, num_heads, ffn_ratio,
                                 attn_drop=0.0, proj_drop=0.0, use_flash_attn=use_flash_attn,
                                 use_canbus_adaln=use_canbus_adaln, cross_first=cross_first)
            for _ in range(num_layers)
        ])
        if self.use_canbus_adaln:
            self.canbus_adaln = nn.Sequential(
                nn.Linear(18, embed_dim),
                nn.SiLU(),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim * 4),
                nn.Tanh()
            )
            nn.init.zeros_(self.canbus_adaln[-2].weight)
            nn.init.zeros_(self.canbus_adaln[-2].bias)

    def _build_style(self, can_bus: torch.Tensor):
        style = self.canbus_adaln(can_bus)
        B = style.shape[0]
        style = style.view(B, 4, self.embed_dim)
        return style

    def forward(self, bev_bchw: torch.Tensor, out_query: torch.Tensor,
                text_emb: Optional[torch.Tensor], can_bus: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, H, W, C = bev_bchw.shape
        style_params = None
        if self.use_canbus_adaln and (can_bus is not None):
            if self.training and self.canbus_dropout_p > 0.0:
                if torch.rand(1, device=bev_bchw.device).item() < self.canbus_dropout_p:
                    can_bus = torch.zeros_like(can_bus)
        if self.use_canbus_adaln and can_bus is not None:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                style_params = self._build_style(can_bus)
                assert style_params.dtype == torch.float32, "style_params should be float32 for stability"

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            x = bev_bchw.reshape(B, H * W, C)
            if self.use_global_query:
                x = torch.cat([x, out_query], dim=1)
            local_text_emb = text_emb
            if self.use_text and (local_text_emb is not None) and self.training and self.text_dropout_p > 0.0:
                if torch.rand(1, device=bev_bchw.device).item() < self.text_dropout_p:
                    local_text_emb = torch.zeros_like(local_text_emb)
            if self.use_text and (local_text_emb is not None):
                text_tokens = self.text_proj(local_text_emb, target_B=B)
                mem = torch.cat([out_query, text_tokens], dim=1)
            else:
                mem = out_query
            for blk in self.layers:
                x = blk(x, mem, style_params)
            bev_tokens = x[:, :H * W, :].reshape(B, H, W, C)
        return bev_tokens.to(bev_bchw.dtype)


EgoAttnV2Block = Curr2FutureLinkBlock
EgoAttentionV2 = Curr2FutureLink


if __name__ == '__main__':
    from ipdb import set_trace

    set_trace()
