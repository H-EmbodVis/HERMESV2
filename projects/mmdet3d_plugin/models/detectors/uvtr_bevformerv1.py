from re import I
from collections import OrderedDict
import torch
import torch.nn as nn

from mmcv.cnn import Conv2d
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.core import multi_apply
import numpy as np
from ..utils.uni3d_voxelpooldepth import DepthNet
from .uvtr_ssl_base import UVTRSSLBase
import copy

count = 0


@DETECTORS.register_module()
class UVTRBEVFormerV1(UVTRSSLBase):
    """UVTRBEVFormerV1."""

    def __init__(
            self,
            num_levels=4,
            pts_voxel_layer=None,
            pts_voxel_encoder=None,
            pts_middle_encoder=None,
            pts_fusion_layer=None,
            img_backbone=None,
            pts_backbone=None,
            img_neck=None,
            depth_head=None,
            pts_neck=None,
            pts_bbox_head=None,
            img_roi_head=None,
            img_rpn_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            frames=(0,),
    ):
        super(UVTRBEVFormerV1, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        if self.with_img_backbone:
            in_channels = self.img_neck.out_channels if img_neck is not None else self.pts_bbox_head.in_channels
            out_channels = self.pts_bbox_head.in_channels
            if isinstance(in_channels, list):
                in_channels = in_channels[0]
            self.input_proj = Conv2d(in_channels, out_channels, kernel_size=1)
            if depth_head is not None:
                depth_dim = self.pts_bbox_head.view_trans.depth_dim
                dhead_type = depth_head.pop("type", "SimpleDepth")
                if dhead_type == "SimpleDepth":
                    self.depth_net = Conv2d(out_channels, depth_dim, kernel_size=1)
                else:
                    self.depth_net = DepthNet(
                        out_channels, out_channels, depth_dim, **depth_head
                    )
            self.depth_head = depth_head
            self.num_levels = num_levels

        if pts_middle_encoder:
            self.pts_fp16 = (
                True if hasattr(self.pts_middle_encoder, "fp16_enabled") else False
            )
        self.frames = frames

    @force_fp32(apply_to=("pts_feats", "img_feats"))
    def forward_pts_train(
            self, pts_feats, img_feats, points, img, img_metas, img_depth, prev_bev=None
    ):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        batch_rays = self.pts_bbox_head.sample_rays(points, img, img_metas)
        out_dict = self.pts_bbox_head(
            pts_feats, img_feats, batch_rays, img_metas, img_depth, prev_bev=prev_bev,
            fusion_prev= False
        )

        losses = self.pts_bbox_head.loss(out_dict, batch_rays)

        if getattr(self.pts_bbox_head, "sim_loss", None) is not None:
            sim_loss = self.pts_bbox_head.sim_loss
            if sim_loss is not None:
                losses.update({'sim_loss': self.pts_bbox_head.sim_loss})
        return losses

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        # # Modify: roll back to previous version for single frame
        is_training = self.training
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_metas = [meta for meta in img_metas_list]

            img_feats_list = self.extract_img_feat(
                img=imgs_queue,
                len_queue=len_queue,
                img_metas=img_metas)

            for t in range(len_queue):
                img_metas = [each[t - len_queue] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each_scale[:, t] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head.get_uni_feats(img_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev,
                                                       only_bev=True, bs=1, dtype=img_feats[0].dtype)
        if is_training:
            self.train()

        return prev_bev

    def forward_train(self, points=None, img_metas=None, img=None, **kwargs):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        ego_mask = self.pts_bbox_head.ray_sampler_cfg.get("ego_mask", None)
        if ego_mask is not None and points is not None:
            for indx, points_ in enumerate(points):
                ego_mask_ = torch.logical_and(
                    torch.logical_and(ego_mask[0] <= points_[:, 0],
                                      ego_mask[2] >= points_[:, 0]),
                    torch.logical_and(ego_mask[1] <= points_[:, 1],
                                      ego_mask[3] >= points_[:, 1]),
                )
                ego_mask_ = torch.logical_not(ego_mask_)
                points[indx] = points_[ego_mask_]
        prev_bev = None
        if len(self.frames) > 1:
            # Obtain history BEV features
            img_metas = OrderedDict(sorted(img_metas[0].items()))
            img_dict = {}
            for ind, t in enumerate(img_metas.keys()):
                img_dict[t] = img[:, ind, ...]
            img = img_dict[0]
            for i in range(len(self.frames)):
                # 1. if the future frame is not in the img_dict, copy the current frame to the future frame
                if self.frames[i] not in img_dict.keys() and self.frames[i] > 0:
                    assert 0 in img_dict.keys(), "The first frame is not in the img_dict."
                    img_dict[self.frames[i]] = copy.deepcopy(img_dict[self.frames[i - 1]])
                    img_metas[self.frames[i]] = copy.deepcopy(img_metas[self.frames[i - 1]])

                # 2. if the future scene is not the same as the current scene, copy the current scene to the future scene
                if "prev_bev_exists" in img_metas[self.frames[i]]:
                    if self.frames[i] != 0 and img_metas[self.frames[i]]['prev_bev_exists'] == False:
                        assert 0 in img_dict.keys(), "The first frame is not in the img_dict."
                        img_dict[self.frames[i]] = copy.deepcopy(img_dict[self.frames[i - 1]])
                        img_metas[self.frames[i]] = copy.deepcopy(img_metas[self.frames[i - 1]])

            img_dict.pop(self.frames[-1])
            prev_img_metas = copy.deepcopy(img_metas)
            prev_img_metas.pop(self.frames[-1])
            prev_imgs = []
            for v in img_dict.values():
                prev_imgs.append(v.unsqueeze(1))
            prev_imgs = torch.cat(prev_imgs, dim=1)

            prev_bev = self.obtain_history_bev(prev_imgs, [prev_img_metas])

        curr_img_meta = [copy.deepcopy(img_metas)[0], ]

        pts_feats, img_feats, img_depth = self.extract_feat(
            points=points, img=img, img_metas=curr_img_meta
        )
        losses = dict()
        losses_pts = self.forward_pts_train(
            pts_feats,
            img_feats[:getattr(self, 'num_levels', None)] if img_feats is not None else img_feats,
            points, img, curr_img_meta, img_depth, prev_bev=prev_bev
        )
        losses.update(losses_pts)

        if img_feats is not None:
            if hasattr(self.img_backbone, 'similarity_losses'):
                losses.update(self.img_backbone.get_similarity_losses())

        return losses

    def train_step(self, data, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        if "_iter" in kwargs:
            self.cur_iter = kwargs["_iter"]
            self.pts_bbox_head.cur_iter = kwargs["_iter"]
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
