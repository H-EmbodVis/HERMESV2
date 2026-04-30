import copy
import os
from codecs import ignore_errors
from re import I
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import json

from mmcv.cnn import Conv2d
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.core import multi_apply
import numpy as np
from ..utils.uni3d_voxelpooldepth import DepthNet
from .uvtr_ssl_base import UVTRSSLBase
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple
import mmcv
from matplotlib import pyplot as plt
from chamferdist import ChamferDistance
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont
import textwrap

chamfer_distance = ChamferDistance()

count = 0
chamfer_dis_list = []
chamfer_dis_list_gt = []
chamfer_dis_list_input_pred = []
depth_l1_list = []


def format_number(n, decimal_places=1):
    if abs(round(n, decimal_places)) <= 1e-2:
        return 0.0
    else:
        format_string = f"{{n:+.{decimal_places}f}}"
        return format_string.format(n=n)


@DETECTORS.register_module()
class HERMES(UVTRSSLBase):
    """HERMES Detector."""

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
            pre_bev_checkpoint=None,
            frame_loss_weight=None,
            text_loss_weight=None,
            world_query_loss_weight=None,
            traj_pred_loss_weight=None,

    ):
        super(HERMES, self).__init__(
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
        self.pre_bev_checkpoint = pre_bev_checkpoint
        self.frame_loss_weight = frame_loss_weight
        self.text_loss_weight = text_loss_weight
        self.world_query_loss_weight = world_query_loss_weight
        self.traj_pred_loss_weight = traj_pred_loss_weight
        if frame_loss_weight is not None:
            assert len(frame_loss_weight) == len(frames)

    @force_fp32(apply_to=("pts_feats", "img_feats"))
    def forward_pts_train(
            self, points, img, img_metas, img_depth, input_bev_embed, prev_bev_down=None, prev_bev=None,
            gt_bev_emb=None,
            prev_img_metas=None,
            conv=None
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
            batch_rays, img_metas, img_depth, input_bev_embed, prev_bev_down, prev_bev, gt_bev_emb,
            fusion_prev=False,
            prev_img_metas=prev_img_metas,
            conv=conv
        )

        losses = self.pts_bbox_head.loss(out_dict, batch_rays)
        if hasattr(self.pts_bbox_head, "loss_chat") and self.pts_bbox_head.loss_chat:
            losses.update({'pretrain_chat_loss': self.pts_bbox_head.loss_chat})
        return losses

    def forward_train(self, points=None, img_metas=None, img=None, **mono_input_dict):
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
        img_metas = OrderedDict(sorted(img_metas[0].items()))
        img_dict = {}
        for ind, t in enumerate(img_metas.keys()):
            img_dict[t] = img[:, ind, ...]
        target_points = {}
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
            # 3. replace the target point with future point
            target_points[self.frames[i]] = img_metas[self.frames[i]]['points'].cuda()

        img = img_dict[0]
        img_dict.pop(self.frames[-1])

        prev_img_metas = copy.deepcopy(img_metas)
        prev_img_metas.pop(self.frames[-1])

        curr_img_meta = [copy.deepcopy(img_metas)[0], ]

        img = img.squeeze(1)
        pts_feats, curr_img_feats, _ = self.extract_feat(
            points=[point for _, point in target_points.items()], img=img.cuda(), img_metas=curr_img_meta
        )
        if os.environ.get("SAVE_POINT_FEAT", "false") == "true":
            sample_idx = curr_img_meta[0]['sample_idx']
            save_dir = os.path.join("./point_feat/")
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)
            np.savez_compressed(
                os.path.join(save_dir, f"{sample_idx}.npz"),
                pts_feat=pts_feats.cpu().numpy()
            )
        # input_bev_emb = self.pts_bbox_head.get_uni_feats(img_feats=curr_img_feats, img_metas=curr_img_meta,
        #                                                  prev_bev=None, only_bev=True, bs=1, dtype=torch.float16)
        input_bev_emb = checkpoint(self.pts_bbox_head.get_uni_feats, curr_img_feats, curr_img_meta, None, True, 1,
                                   torch.float16, use_reentrant=False)
        input_bev_emb = input_bev_emb.permute(0, 2, 1).reshape(1, -1, self.pts_bbox_head.bev_h,
                                                               self.pts_bbox_head.bev_w)

        # if not use down_sample
        if hasattr(self.pts_bbox_head, 'down_sample'):
            # input_bev_emb = self.pts_bbox_head.down_sample.down_sample(input_bev_emb)
            input_bev_emb = checkpoint(self.pts_bbox_head.down_sample.down_sample, input_bev_emb)

        losses = dict()
        img_depth = None
        batch_rays = {}
        for index in range(len(self.frames)):
            batch_rays[self.frames[index]] = self.pts_bbox_head.sample_rays([target_points[self.frames[index]], ],
                                                                            img, curr_img_meta)

        out_dict = self.pts_bbox_head(
            pts_feats, batch_rays, img_metas, img_depth, input_bev_emb, None, None, None,
            fusion_prev=False, prev_img_metas=None, conv=curr_img_meta[0]["conv"], img=img
        )
        for index, frame in enumerate(self.frames):
            # for index, frame in enumerate(range(len(out_dict))):
            losses_pts = self.pts_bbox_head.loss([out_dict[index], ], batch_rays[frame])
            for key in losses_pts.keys():
                loss_weight = self.frame_loss_weight[index] if self.frame_loss_weight is not None else 1.0
                if loss_weight > 0:
                    losses[key + "_frame{}".format(frame)] = losses_pts[key] * loss_weight
        if hasattr(self.pts_bbox_head, "loss_chat") and self.pts_bbox_head.loss_chat:
            loss_weight = self.text_loss_weight if self.text_loss_weight is not None else 1.0
            if loss_weight > 0:
                losses.update({'pretrain_chat_loss': self.pts_bbox_head.loss_chat * loss_weight})
        if hasattr(self.pts_bbox_head, "loss_world_query") and self.pts_bbox_head.loss_world_query:
            loss_weight = self.world_query_loss_weight if self.world_query_loss_weight is not None else 0.1
            if loss_weight > 0:
                losses.update({'world_query_loss': self.pts_bbox_head.loss_world_query * loss_weight})
        if hasattr(self.img_backbone, 'similarity_losses'):
            losses.update(self.img_backbone.get_similarity_losses())
        if hasattr(self.pts_bbox_head, "loss_can_bus") and self.pts_bbox_head.pred_can_bus:
            loss_weight = self.traj_pred_loss_weight if self.traj_pred_loss_weight is not None else 1.0
            losses.update({'traj_pred_loss': self.pts_bbox_head.loss_can_bus * loss_weight})

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

    def forward_test(self, img_metas, points=None, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        if isinstance(img[0], torch.Tensor):
            results = self.simple_test(img_metas[0], points[0], img[0])
        else:
            results = self.simple_test(img_metas[0].data[0], points[0].data[0], img[0].data[0])
        return results

    def simple_test(self, img_metas, points=None, img=None):
        """Test function without augmentaiton."""
        img_metas = OrderedDict(sorted(img_metas[0].items()))
        img_dict = {}
        for ind, t in enumerate(img_metas.keys()):
            img_dict[t] = img[:, ind, ...]
        ignore_flag = False
        target_points = {}

        # if i > 0 not in img_dict.keys(), copy 0 to i
        for i in range(len(self.frames)):
            # 1. if the future frame is not in the img_dict, copy the current frame to the future frame
            if self.frames[i] not in img_dict.keys() and self.frames[i] > 0:
                assert 0 in img_dict.keys(), "The first frame is not in the img_dict."
                img_dict[self.frames[i]] = copy.deepcopy(img_dict[self.frames[i - 1]])
                img_metas[self.frames[i]] = copy.deepcopy(img_metas[self.frames[i - 1]])
                ignore_flag = True
            # 2. if the future scene is not the same as the current scene, copy the current scene to the future scene
            if self.frames[i] > 0 and img_metas[self.frames[i]]['prev_bev_exists'] == False:
                assert 0 in img_dict.keys(), "The first frame is not in the img_dict."
                img_dict[self.frames[i]] = copy.deepcopy(img_dict[self.frames[i - 1]])
                img_metas[self.frames[i]] = copy.deepcopy(img_metas[self.frames[i - 1]])
                ignore_flag = True
            # 3. replace the target point with future point
            target_points[self.frames[i]] = img_metas[self.frames[i]]['points'].cuda()

        img = img_dict[0]
        img_dict.pop(self.frames[-1])

        prev_img_metas = copy.deepcopy(img_metas)
        prev_img_metas.pop(self.frames[-1])

        curr_img_meta = [copy.deepcopy(img_metas)[self.frames[0]], ]

        img = img.squeeze(1)
        _, curr_img_feats, _ = self.extract_feat(
            points=points, img=img.cuda(), img_metas=curr_img_meta
        )
        with torch.cuda.amp.autocast():
            input_bev_emb = self.pts_bbox_head.get_uni_feats(img_feats=curr_img_feats, img_metas=curr_img_meta,
                                                             prev_bev=None, only_bev=True, bs=1, dtype=torch.float16)
            input_bev_emb = input_bev_emb.permute(0, 2, 1).reshape(1, -1, self.pts_bbox_head.bev_h,
                                                                   self.pts_bbox_head.bev_w)
            if hasattr(self.pts_bbox_head, 'down_sample'):
                input_bev_emb = self.pts_bbox_head.down_sample.down_sample(input_bev_emb)
        conv_dict = {}
        batch_rays = {}
        for index in range(len(self.frames)):
            batch_rays[self.frames[index]] = self.pts_bbox_head.sample_rays([target_points[self.frames[index]], ],
                                                                            img, curr_img_meta)

        curr_img_meta = [copy.deepcopy(img_metas)[self.frames[0]], ]

        # set the target point
        token = curr_img_meta[0]['sample_idx']
        num_nusc_qa = 0
        conv_save = []
        question_list = []
        gt_answer_list = []
        conv_path = "./data/data_nusc/conv/val/"
        conv_path = os.path.join(conv_path, token + ".json")
        desc_path = "./data/data_nusc/desc/val/"
        desc_path = os.path.join(desc_path, token + ".json")
        with open(conv_path, 'r') as f:
            conv_gt = json.load(f)
            # add the questions to the question_list and the gt_answer to the gt_answer_list
            for i in range(len(conv_gt)):
                question_list.append(conv_gt[i]['question'])
                gt_answer_list.append(conv_gt[i]['answer'])
        with open(desc_path, 'r') as f:
            desc_gt = json.load(f)
            desc = desc_gt['description']
            action = desc_gt['action']
            question_list = question_list + ["What action should be taken in the current driving scenario?"]
            gt_answer_list = gt_answer_list + [action]
        question_list = ["Can you provide a summary of the current driving scenario based on the input images?"] \
                        + question_list
        gt_answer_list = [desc] + gt_answer_list

        # drivelm_path = "./data/drivelm/val/"
        # drivelm_path = os.path.join(drivelm_path, token + ".json")
        # with open(drivelm_path, 'r') as f:
        #     drivelm_data = json.load(f)
        #     tasks = ["perception", "prediction", "planning", "behavior"]
        #     for task in tasks:
        #         if task in drivelm_data and drivelm_data[task]:
        #             for item in drivelm_data[task]:
        #                 question = item.get("Q", "")
        #                 answer = item.get("A", "")
        #                 question_list.append(question)
        #                 gt_answer_list.append(answer)

        question_list = question_list + ["Please try to imagine the future scene."]
        gt_answer_list = gt_answer_list + [
            "Sure, I can try to envision the future scene based on the understanding of the driving scenario."]

        # question_list = ["Please try to imagine the future scene."]
        # gt_answer_list = [
        #     "Sure, I can try to envision the future scene based on the understanding of the driving scenario."]
        # nusc_qa_path = "/data/NuScenes_val_questions_split/"
        # nusc_qa_path = os.path.join(nusc_qa_path, token + ".json")
        # if os.path.exists(nusc_qa_path):
        #     with open(nusc_qa_path, 'r') as f:
        #         qa = json.load(f)["questions"]
        #         num_nusc_qa = len(qa)
        #         for i in range(len(qa)):
        #             question_list.append(qa[i]['question'] + "Use one or two words to answer the question.")
        #             gt_answer_list.append(qa[i]['answer'])

        assert len(question_list) == len(gt_answer_list)
        question_list[0] = "<image>\n" + question_list[0]

        history_list = []
        conv = []
        for i, question in enumerate(question_list):
            response, output = self.pts_bbox_head.llm.chat(
                tokenizer=None,
                pixel_values=input_bev_emb.permute(0, 2, 3, 1).to(self.pts_bbox_head.llm.torch_dtype),
                question=question,
                conv=None,
                generation_config=self.pts_bbox_head.chat_cfg,
                verbose=True,
                # history=history_list
                history=None
            )
            conv.append({'from': 'human', 'value': question})
            conv.append({'from': 'gpt', 'value': response})
            conv_save.append({
                "id": "{}".format(token),
                "question": question,
                "answer": response,
                "gt_answer": gt_answer_list[i]
            })

        results = self.pts_bbox_head(
            None, batch_rays, img_metas, None, input_bev_emb, fusion_prev=False,
            llm_pred_bev=None, conv=conv, img=img.cuda()
        )

        traj_pred = "None"
        if hasattr(self.pts_bbox_head, "pred_can_bus") and self.pts_bbox_head.pred_can_bus and 'traj_pred' in results[
            0].keys():
            assert 'gt_planning' in curr_img_meta[0].keys()
            try:
                planning_traj = curr_img_meta[0]['gt_planning'][0, :, :2]
            except:
                planning_traj = curr_img_meta[0]['gt_planning'][:, :2]
            try:
                mask = curr_img_meta[0]['gt_planning_mask'][0].any(axis=1)
            except:
                mask = img_metas[self.frames[0]]['gt_planning_mask'].astype(bool)
            planning_traj = planning_traj[mask]
            if len(planning_traj) == 6:
                traj_pred = results[0]['traj_pred']
                # traj_pred = torch.zeros_like(traj_pred_)
                # for i in range(0, len(traj_pred)):
                #     if i == 0:
                #         traj_pred[i] = traj_pred_[i]
                #     else:
                #         traj_pred[i] = traj_pred_[i] + traj_pred[i - 1]
                traj_pred = traj_pred.cpu().numpy().tolist()
        H, W = curr_img_meta[0]["ori_shape"][0], curr_img_meta[0]["ori_shape"][1]
        num_cam = len(curr_img_meta[0]["img_shape"])

        mean = [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
        std = [0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]
        img_rgb = (img.squeeze(0)[:, :, :H, :W].cpu().permute(0, 2, 3, 1).numpy() * std + mean).astype(np.uint8)

        output_dict = {}
        for index, frame in enumerate(self.frames):
            lidar_rays, _ = batch_rays[frame]
            scale_factor = lidar_rays[0]["scale_factor"].astype(np.float32)
            llm_render_depth = results[index]["depth"] / scale_factor
            lidar_ray_o = lidar_rays[0]["ray_o"].detach().cpu().numpy()
            lidar_ray_d = lidar_rays[0]["ray_d"].detach().cpu().numpy()
            lidar_pred = lidar_ray_d * llm_render_depth.detach().cpu().numpy() + lidar_ray_o
            lidar_gt = lidar_ray_d * lidar_rays[0][
                "depth"].detach().cpu().numpy() / scale_factor + lidar_ray_o
            depth_l1 = torch.abs(lidar_rays[0]["depth"] - results[index]["depth"]).mean()
            lidar_error = np.abs((lidar_pred - lidar_gt) / lidar_gt).mean()
            chamfer_dis = self.compute_chamfer_distance(torch.from_numpy(lidar_pred).float(),
                                                        torch.from_numpy(lidar_gt).float(), 'cuda')

            lidar_gt_vis = self.visualize_lidar(lidar_gt)
            lidar_gt_vis = cv2.cvtColor(lidar_gt_vis, cv2.COLOR_BGR2RGB)
            lidar_pred_vis = self.visualize_lidar(lidar_pred)
            lidar_pred_vis = cv2.cvtColor(lidar_pred_vis, cv2.COLOR_BGR2RGB)

            output_dict[frame] = {
                frame: {
                    "lidar_pred": lidar_pred,
                    "lidar_gt": lidar_gt,
                    "depth_l1": depth_l1,
                    "lidar_error": lidar_error,
                    "chamfer_dis": chamfer_dis,
                    "lidar_gt_vis": lidar_gt_vis,
                    "lidar_pred_vis": lidar_pred_vis,
                }
            }

        config_path = os.environ.get("IMG_SAVE_CONFIG_PATH", "stage3.py")
        formatted_time = os.environ.get("TIME_STR", "hermes_eval")
        BGR_FORMAT = int(os.environ.get("BGR_FORMAT", "1"))
        SAVE_IMAGE = int(os.environ.get("SAVE_IMAGE", "0"))
        if config_path is None or formatted_time is None:
            raise ValueError("IMG_SAVE_CONFIG_PATH or TIME_STR is not set")
        dir_name = config_path.split('/')[-1].split('.py')[0]
        dir_name = os.path.join(dir_name, formatted_time)
        save_dir = os.path.join("./outputs/", dir_name)
        os.makedirs(save_dir, exist_ok=True)

        if SAVE_IMAGE:
            gt_image = []
            for i in range(num_cam):
                rgb_i = img_rgb[i]
                if BGR_FORMAT:
                    rgb_i = cv2.cvtColor(rgb_i, cv2.COLOR_BGR2RGB)
                # resize the image to (H//2, W//2)
                rgb_i = cv2.resize(rgb_i, (W // 2, H // 2))
                gt_image.append(rgb_i)
            gt_image = [gt_image[i] for i in [2, 0, 1, 4, 3, 5]]
            gt_image_1 = cv2.hconcat(gt_image[:len(gt_image) // 2])
            gt_image_2 = cv2.hconcat(gt_image[len(gt_image) // 2:])
            save_image = cv2.vconcat([gt_image_1, gt_image_2])

            # cv2.imwrite(f"{save_dir}/{img_metas[0]['sample_idx']}_gt.png", save_image)
            # # copy a blank image with the same size as save_image and put gpt_reply on it, auto change line
            # text_image_pred = self.text_to_image_cv2_auto_split("token: {}\n".format(token) + conv,
            #                                                     (gt_image_2.shape[0] // 2, gt_image_2.shape[1]))
            # save_image = cv2.vconcat([save_image, text_image_pred])

            num_pred = len(self.frames)
            lidar_gt_vis_list = []
            lidar_pred_vis_list = []
            for index, frame in enumerate(self.frames):
                lidar_gt_vis_list.append(output_dict[frame][frame]["lidar_gt_vis"])
            lidar_gt_vis = cv2.hconcat(lidar_gt_vis_list)
            lidar_gt_vis[:3, :, :] = lidar_gt_vis[-3:, :, :] = lidar_gt_vis[:, :3, :] = lidar_gt_vis[:, -3:, :] = [234,
                                                                                                                   234,
                                                                                                                   234]
            lidar_gt_vis = cv2.resize(lidar_gt_vis, (save_image.shape[1], save_image.shape[1] // num_pred))
            save_image = cv2.vconcat([save_image, lidar_gt_vis])
            for index, frame in enumerate(self.frames):
                lidar_pred_vis_list.append(output_dict[frame][frame]["lidar_pred_vis"])

            lidar_pred_vis = cv2.hconcat(lidar_pred_vis_list)
            lidar_pred_vis[:3, :, :] = lidar_pred_vis[-3:, :, :] = lidar_pred_vis[:, :3, :] = lidar_pred_vis[:, -3:,
            :] = [234, 234, 234]
            lidar_pred_vis = cv2.resize(lidar_pred_vis, (save_image.shape[1], save_image.shape[1] // num_pred))
            save_image = cv2.vconcat([save_image, lidar_pred_vis])
            # qa_pairs = [
            #     (conv['question'], conv['answer']) for i, conv in enumerate(conv_save[:-1])
            # ]
            # conversation_image = ConversationImage(qa_pairs, image_width=int(save_image.shape[0] * 2 / 3),
            #                                        image_height=save_image.shape[0], font_size=30)
            # image_array = conversation_image.generate_image()
            #
            # image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            # save_image = cv2.hconcat([save_image, image_array])

            chamfer_lines = [
                f"frame {frame}: {output_dict[frame][frame]['chamfer_dis'].item():.4f}"
                for frame in self.frames
            ]
            chamfer_text = "Chamfer distance (predicted pcd):\n" + "\n".join(chamfer_lines)
            chamfer_img_height = 80 + 40 * len(chamfer_lines)
            chamfer_image = self.text_to_image_cv2_auto_split(
                chamfer_text,
                (chamfer_img_height, save_image.shape[1]),
                font_scale=1,
                font_color=(0, 0, 0),
                line_type=2,
            )
            save_image = cv2.vconcat([save_image, chamfer_image])
            qa_pairs = [
                (conv['question'], conv['answer']) for i, conv in enumerate(conv_save[:-1])
            ]
            conversation_image = ConversationImage(qa_pairs, image_width=int(save_image.shape[0] * 2 / 3),
                                                   image_height=save_image.shape[0], font_size=30)
            image_array = conversation_image.generate_image()

            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            save_image = cv2.hconcat([save_image, image_array])

            sample_idx = img_metas[0].get('sample_idx', 'unknown')
            footer_image = self.text_to_image_cv2_auto_split(
                f"sample_idx: {sample_idx}",
                (80, save_image.shape[1])
            )
            save_image = cv2.vconcat([save_image, footer_image])
            cv2.imwrite(os.path.join(save_dir, f"{img_metas[0]['sample_idx']}_{output_dict[0][0]['chamfer_dis'].item():.2f}_{output_dict[2][2]['chamfer_dis'].item():.2f}_{output_dict[4][4]['chamfer_dis'].item():.2f}_{output_dict[6][6]['chamfer_dis'].item():.2f}.png"), save_image)

            print(f"save to {save_dir}/{img_metas[0]['sample_idx']}.png")
        else:
            print("save image is not set")

        # save results to a json file
        results = {}
        for index, frame in enumerate(self.frames):
            results["chamfer_dis_frame_{}".format(frame)] = output_dict[frame][frame]["chamfer_dis"].item()
        results.update({
            "ignore_flag": ignore_flag,
            "desc": conv_save,
            "num_nusc_qa": num_nusc_qa,
            "traj_pred": traj_pred,
        })
        os.makedirs(f"{save_dir}/results", exist_ok=True)
        with open(f"{save_dir}/results/{img_metas[0]['sample_idx']}.json", 'w') as f:
            json.dump(results, f, indent=1)
        print(f"save results to {save_dir}/results/{img_metas[0]['sample_idx']}.json")

        if ignore_flag:
            print("ignore this frame")
        for index, frame in enumerate(self.frames):
            print("chamfer_dis_frame_{}:".format(frame), output_dict[frame][frame]["chamfer_dis"])

        print("#" * 10)

        results = [None, ]
        return results

    def compute_chamfer_distance(self, pred_pcd, gt_pcd, device):
        cd_forward, cd_backward, CD_info = chamfer_distance(
            pred_pcd[None, ...].to(device),
            gt_pcd[None, ...].to(device),
            bidirectional=True,
            reduction='sum')

        chamfer_dist_value = (cd_forward / pred_pcd.shape[0]) + (cd_backward / gt_pcd.shape[0])
        return chamfer_dist_value / 2.0

    def visualize_lidar(
            self,
            lidar: Optional[np.ndarray] = None,
            *,
            xlim: Tuple[float, float] = (-52, 52),
            ylim: Tuple[float, float] = (-52, 52),
            color: Optional[Tuple[int, int, int]] = None,
            radius: float = 0.5,
            thickness: float = 25,
    ):
        fig = plt.figure(figsize=(7.68, 7.68))

        ax = plt.gca()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect(1)
        ax.set_axis_off()
        # make the scatter plot's color align with the lidar's height
        if color is None:
            color = lidar[:, 2].clip(-3, 5)
            # color = (color - color.min()) / (color.max() - color.min())
            color = (color + 3) / 8

        if lidar is not None:
            plt.scatter(
                lidar[:, 0],
                lidar[:, 1],
                s=3.14 * radius ** 2,
                c=color,
                cmap="viridis",
            )
        # plt.savefig("temp.pdf", bbox_inches='tight', pad_inches=0)
        # set_trace()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return data

    def text_to_image_cv2(self, text, image_size, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=2, font_color=(0, 0, 0),
                          line_type=3, ):
        # Create a blank image with white background
        img = np.ones_like(image_size, np.uint8) * 255
        # Initialize starting position
        y0, dy = 100, 50  # Start at y=y0, and move down by dy pixels for each new line

        # Split text into lines to fit within the image width
        lines = text.split('\n')

        for i, line in enumerate(lines):
            y = y0 + i * dy
            cv2.putText(img, line, (400, y), font, font_scale, font_color, line_type)
        return img

    def text_to_image_cv2_auto_split(self, text, image_size, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1,
                                     font_color=(0, 0, 0), line_type=3):
        # Create a blank image with white background
        img = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255

        # Initialize starting position
        y0, dy = 50, 40  # Initial Y position and line height
        max_width = image_size[1] - 20  # Account for padding on sides

        # Split text into words
        words = text.split(' ')
        lines = []
        current_line = ""

        for word in words:
            # Test line width by adding one word at a time
            test_line = f"{current_line} {word}".strip()
            (line_width, _) = cv2.getTextSize(test_line, font, font_scale, line_type)[0]
            if line_width <= max_width and word != "\n":
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word

        # Append the last line if there's any remaining text
        if current_line:
            lines.append(current_line)

        # Draw each line on the image
        for i, line in enumerate(lines):
            y = y0 + i * dy  # Calculate y position for each line
            cv2.putText(img, line, (10, y), font, font_scale, font_color, line_type)

        return img

    def put_text(self, img, text, position=(15, 30), font_size=50, font_color=(0, 0, 0)):
        font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
        font = ImageFont.truetype(font_path, font_size)
        pil_image = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_image)
        draw.text(position, text, font=font, fill=font_color)
        return np.array(pil_image)


class ConversationImage:
    def __init__(self, qa_pairs, image_width=1400, image_height=2100, font_path=None, font_size=40):
        self.qa_pairs = qa_pairs
        self.image_width = image_width
        self.image_height = image_height
        self.font_path = font_path or "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
        self.font_size = font_size

        # 设置字体
        self.font = ImageFont.truetype(self.font_path, self.font_size)

        # 创建图像和画布
        self.background_color = (234, 234, 234)
        self.image = Image.new("RGB", (self.image_width, self.image_height), self.background_color)
        self.draw = ImageDraw.Draw(self.image)

        # 设置图标
        self.user_icon = Image.new("RGB", (50, 50), color=(0, 123, 255))  # 用户图标，蓝色 50x50像素
        # self.user_icon = Image.open("user_icon.png").resize((50, 50))  # 用户图标，蓝色 50x50像素
        self.robot_icon = Image.new("RGB", (50, 50), color=(34, 139, 34))  # 机器人图标，绿色 50x50像素
        # self.robot_icon = Image.open("logo.jpg").resize((50, 50))  # 机器人图标，绿色 50x50像素

        # 设置气泡和文本样式
        self.question_bubble_color = (255, 255, 255)
        self.answer_bubble_color = (169, 234, 122)  # 浅绿色
        self.bubble_padding = 25  # 气泡内边距
        self.bubble_corner_radius = 15  # 气泡圆角半径
        self.triangle_size = 40  # 尖角的大小

        # 设置文本的最大宽度（根据气泡的宽度调整）
        self.max_question_width = self.image_width * 1.5  # 问题部分的最大宽度
        self.max_answer_width = self.image_width * 1.5  # 回答部分的最大宽度

    def get_text_width(self, text):
        bbox = self.draw.textbbox((0, 0), text, font=self.font)
        return bbox[2] - bbox[0]

    def justify_text(self, text, max_width):
        lines = text.split('\n')
        justified_lines = []

        for line in lines:
            words_in_line = line.split()

            if len(words_in_line) == 1:
                justified_lines.append(line)
                continue

            line_width = sum([self.get_text_width(word) for word in words_in_line])
            total_spaces = max_width - line_width
            space_between_words = total_spaces // (len(words_in_line) - 1)
            extra_spaces = total_spaces % (len(words_in_line) - 1)

            aligned_line = words_in_line[0]
            for i in range(1, len(words_in_line)):
                spaces_to_add = space_between_words + (1 if i <= extra_spaces else 0)
                aligned_line += ' ' * spaces_to_add + words_in_line[i]

            justified_lines.append(aligned_line)

        return '\n'.join(justified_lines)

    def draw_bubble(self, x, y, width, height, color, draw_triangle=False):
        # 绘制气泡的矩形部分
        self.draw.rounded_rectangle([x, y, x + width, y + height], self.bubble_corner_radius, fill=color)

        if draw_triangle:
            # 如果需要绘制尖角，绘制三角形
            triangle = [
                (x + width - self.triangle_size, y + height - self.triangle_size),
                (x + width, y + height),
                (x + width - self.triangle_size, y + height)
            ]
            self.draw.polygon(triangle, fill=color)

    def generate_image(self):
        y_position = 50  # 初始y坐标
        draw_triangle = False  # 设置是否绘制尖角

        for question, answer in self.qa_pairs:
            # 计算每行文本的宽度限制
            question_max_chars_per_line = self.max_question_width // self.get_text_width('A')  # 每行可以容纳的最大字符数
            wrapped_question = textwrap.fill(question, width=question_max_chars_per_line)  # 自动换行

            # 使用 getbbox() 获取问题文本边界框 (left, top, right, bottom)
            question_bbox = self.draw.textbbox((0, 0), wrapped_question, font=self.font)
            question_width = question_bbox[2] - question_bbox[0]
            question_height = question_bbox[3] - question_bbox[1]
            question_bubble_width = min(question_width + 2 * self.bubble_padding, self.max_question_width)
            question_bubble_height = question_height + self.bubble_padding

            # 绘制问题气泡，确保图标位于气泡左侧并且垂直居中
            self.draw_bubble(50 + self.user_icon.width + self.bubble_padding, y_position, question_bubble_width,
                             question_bubble_height,
                             self.question_bubble_color, draw_triangle)
            self.draw.text((50 + self.user_icon.width + 2 * self.bubble_padding, y_position + self.bubble_padding // 2),
                           wrapped_question,
                           fill=(0, 0, 0), font=self.font)

            # # 将用户图标绘制到气泡左侧
            # self.image.paste(self.user_icon, (50, y_position + (question_bubble_height - self.user_icon.height) // 2),
            #                  self.user_icon)

            # 更新 y_position：由于气泡和图标已经占用了空间，调整 y_position 为下一个回答部分
            y_position += question_bubble_height + self.bubble_padding

            # 计算回答气泡的位置和大小
            answer_max_chars_per_line = self.max_answer_width // self.get_text_width('A')  # 每行可以容纳的最大字符数
            wrapped_answer = textwrap.fill(answer, width=answer_max_chars_per_line)  # 自动换行

            # 使用 getbbox() 获取回答文本边界框 (left, top, right, bottom)
            answer_bbox = self.draw.textbbox((0, 0), wrapped_answer, font=self.font)
            answer_width = answer_bbox[2] - answer_bbox[0]
            answer_height = answer_bbox[3] - answer_bbox[1]
            answer_bubble_width = min(answer_width + 2 * self.bubble_padding, self.max_answer_width)
            answer_bubble_height = answer_height + self.bubble_padding

            # 绘制回答气泡（矩形 + 可选尖角）
            self.draw_bubble(self.image_width - 70 - answer_bubble_width - self.robot_icon.width, y_position,
                             answer_bubble_width,
                             answer_bubble_height, self.answer_bubble_color, draw_triangle)

            # # 计算机器人图标的位置，确保它与文本的第一行对齐
            # icon_y_position = y_position + self.bubble_padding // 2  # 图标放置在气泡的顶部，即第一行文本的位置
            # self.image.paste(self.robot_icon, (self.image_width - 50 - self.robot_icon.width, icon_y_position),
            #                  self.robot_icon)

            # 绘制回答文本
            self.draw.text((
                self.image_width - 50 - answer_bubble_width + self.bubble_padding // 2 - self.robot_icon.width,
                y_position + self.bubble_padding // 2),
                wrapped_answer, fill=(0, 0, 0), font=self.font)

            # 更新y坐标，准备绘制下一部分
            y_position += answer_bubble_height + self.bubble_padding

        # 将生成的图片转换为NumPy数组并返回
        return np.array(self.image)
