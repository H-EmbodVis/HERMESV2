import numpy as np
import torch
import mmcv
from mmdet.datasets.builder import PIPELINES
from PIL import Image
import random


@PIPELINES.register_module()
class CropResizeFlipImage(object):
    """Fixed Crop and then randim resize and flip the image. Note the flip requires to flip the feature in the network   
    ida_aug_conf = {
        "reisze": [576, 608, 640, 672, 704]  # stride of 32 based on 640 (0.9, 1.1)
        "reisze": [512, 544, 576, 608, 640, 672, 704, 736, 768]  #  (0.8, 1.2)
        "reisze": [448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832]  #  (0.7, 1.3)
        "crop": (0, 260, 1600, 900), 
        "H": 900,
        "W": 1600,
        "rand_flip": True,
}
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, data_aug_conf=None, training=True, debug=False):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.debug = debug

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        if not 'aug_param' in results.keys():
            results['aug_param'] = {}
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip = self._sample_augmentation(results)

        if self.debug:
            # unique id per img
            from uuid import uuid4
            uid = uuid4()
            # lidar is RFU in nuscenes
            lidar_pts = np.array([
                [10, 30, -2, 1],
                [-10, 30, -2, 1],
                [5, 15, -2, 1],
                [-5, 15, -2, 1],
                [30, 0, -2, 1],
                [-30, 0, -2, 1],
                [10, -30, -2, 1],
                [-10, -30, -2, 1]
            ], dtype=np.float32).T

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))

            if self.debug:
                pts_to_img_pre_aug = results['lidar2img'][i] @ lidar_pts
                pts_to_img_pre_aug = pts_to_img_pre_aug / pts_to_img_pre_aug[2:3,
                :]  # div by the depth component in homogenous vector

                img_copy = Image.fromarray(np.uint8(imgs[i]))
                for j in range(pts_to_img_pre_aug.shape[1]):
                    x, y = int(pts_to_img_pre_aug[0, j]), int(pts_to_img_pre_aug[1, j])
                    if (0 < x < img_copy.width) and (0 < y < img_copy.height):
                        img_copy.putpixel((x - 1, y - 1), (255, 0, 0))
                        img_copy.putpixel((x - 1, y), (255, 0, 0))
                        img_copy.putpixel((x - 1, y + 1), (255, 0, 0))
                        img_copy.putpixel((x, y - 1), (0, 255, 0))
                        img_copy.putpixel((x, y), (0, 255, 0))
                        img_copy.putpixel((x, y + 1), (0, 255, 0))
                        img_copy.putpixel((x + 1, y - 1), (0, 0, 255))
                        img_copy.putpixel((x + 1, y), (0, 0, 255))
                        img_copy.putpixel((x + 1, y + 1), (0, 0, 255))
                img_copy.save(f'pre_aug_{uid}_{i}.png')

            # augmentation (resize, crop, horizontal flip, rotate)
            # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()  ###different view use different aug (BEV Det)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results['cam2img'][i][:3, :3] = np.matmul(ida_mat, results['cam2img'][i][:3, :3])

            if self.debug:
                pts_to_img_post_aug = np.matmul(results['cam2img'][i], results['lidar2cam'][i]) @ lidar_pts
                pts_to_img_post_aug = pts_to_img_post_aug / pts_to_img_post_aug[2:3,
                :]  # div by the depth component in homogenous vector
                for j in range(pts_to_img_post_aug.shape[1]):
                    x, y = int(pts_to_img_post_aug[0, j]), int(pts_to_img_post_aug[1, j])
                    if (0 < x < img.width) and (0 < y < img.height):
                        img.putpixel((x - 1, y - 1), (255, 0, 0))
                        img.putpixel((x - 1, y), (255, 0, 0))
                        img.putpixel((x - 1, y + 1), (255, 0, 0))
                        img.putpixel((x, y - 1), (0, 255, 0))
                        img.putpixel((x, y), (0, 255, 0))
                        img.putpixel((x, y + 1), (0, 255, 0))
                        img.putpixel((x + 1, y - 1), (0, 0, 255))
                        img.putpixel((x + 1, y), (0, 0, 255))
                        img.putpixel((x + 1, y + 1), (0, 0, 255))
                img.save(f'post_aug_{uid}_{i}.png')

            if 'mono_ann_idx' in results.keys():
                # apply transform to dd3d intrinsics
                if i in results['mono_ann_idx'].data:
                    mono_index = results['mono_ann_idx'].data.index(i)
                    intrinsics = results['mono_input_dict'][mono_index]['intrinsics']
                    if torch.is_tensor(intrinsics):
                        intrinsics = intrinsics.numpy().reshape(3, 3).astype(np.float32)
                    elif isinstance(intrinsics, np.ndarray):
                        intrinsics = intrinsics.reshape(3, 3).astype(np.float32)
                    else:
                        intrinsics = np.array(intrinsics, dtype=np.float32).reshape(3, 3)
                    results['mono_input_dict'][mono_index]['intrinsics'] = np.matmul(ida_mat, intrinsics)
                    results['mono_input_dict'][mono_index]['height'] = img.size[1]
                    results['mono_input_dict'][mono_index]['width'] = img.size[0]

                    # apply transform to dd3d box
                    for ann in results['mono_input_dict'][mono_index]['annotations']:
                        # bbox_mode = BoxMode.XYXY_ABS
                        box = self._box_transform(ann['bbox'], resize, crop, flip, img.size[0])[0]
                        box = box.clip(min=0)
                        box = np.minimum(box, list(img.size + img.size))
                        ann["bbox"] = box

        results["img"] = new_imgs
        results['lidar2img'] = [np.matmul(results['cam2img'][i], results['lidar2cam'][i]) for i in
                                range(len(results['lidar2cam']))]

        return results

    def _box_transform(self, box, resize, crop, flip, img_width):
        box = np.array([box])
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)

        # crop
        coords[:, 0] -= crop[0]
        coords[:, 1] -= crop[1]

        # resize
        coords[:, 0] = coords[:, 0] * resize
        coords[:, 1] = coords[:, 1] * resize

        coords = coords.reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_box = np.concatenate((minxy, maxxy), axis=1)

        return trans_box

    def _img_transform(self, img, resize, resize_dims, crop, flip):
        ida_rot = np.eye(2)
        ida_tran = np.zeros(2)
        # adjust image
        img = img.crop(crop)
        img = img.resize(resize_dims)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= np.array(crop[:2]) * resize
        ida_mat = np.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self, results):
        if 'CropResizeFlipImage_param' in results['aug_param'].keys():
            return results['aug_param']['CropResizeFlipImage_param']
        crop = self.data_aug_conf["crop"]

        if self.training:
            resized_h = random.choice(self.data_aug_conf["reisze"])
            resized_w = resized_h / (crop[3] - crop[1]) * (crop[2] - crop[0])
            resize = resized_h / (crop[3] - crop[1])
            resize_dims = (int(resized_w), int(resized_h))
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
        else:
            resized_h = random.choice(self.data_aug_conf["reisze"])
            assert len(self.data_aug_conf["reisze"]) == 1
            resized_w = resized_h / (crop[3] - crop[1]) * (crop[2] - crop[0])
            resize = resized_h / (crop[3] - crop[1])
            resize_dims = (int(resized_w), int(resized_h))
            flip = False
        results['aug_param']['CropResizeFlipImage_param'] = (resize, resize_dims, crop, flip)

        return resize, resize_dims, crop, flip


@PIPELINES.register_module()
class ResizeCropFlipRotImage():
    def __init__(self, data_aug_conf=None, with_2d=True, filter_invisible=True, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.min_size = 2.0
        self.with_2d = with_2d
        self.filter_invisible = filter_invisible

    def __call__(self, results):

        imgs = results['img']
        N = len(imgs)
        new_imgs = []
        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []
        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"

        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()

        if 'cam2img' not in results:
            if 'cam_intrinsic' in results:
                results['cam2img'] = [x.copy() for x in results['cam_intrinsic']]
            else:
                raise ValueError("Neither 'cam2img' nor 'cam_intrinsic' found in results.")

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            if self.training and self.with_2d:  # sync_2d bbox labels
                gt_bboxes = results['gt_bboxes'][i]
                centers2d = results['centers2d'][i]
                gt_labels = results['gt_labels'][i]
                depths = results['depths'][i]
                if len(gt_bboxes) != 0:
                    gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                        gt_bboxes,
                        centers2d,
                        gt_labels,
                        depths,
                        resize=resize,
                        crop=crop,
                        flip=flip,
                    )
                if len(gt_bboxes) != 0 and self.filter_invisible:
                    gt_bboxes, centers2d, gt_labels, depths = self._filter_invisible(gt_bboxes, centers2d, gt_labels,
                                                                                     depths)

                new_gt_bboxes.append(gt_bboxes)
                new_centers2d.append(centers2d)
                new_gt_labels.append(gt_labels)
                new_depths.append(depths)

            new_imgs.append(np.array(img).astype(np.float32))
            results['cam2img'][i][:3, :3] = np.matmul(ida_mat, results['cam2img'][i][:3, :3])
            if 'cam_intrinsic' in results:
                results['cam_intrinsic'][i] = results['cam2img'][i]

        results['img'] = new_imgs
        results['lidar2img'] = [np.matmul(results['cam2img'][i], results['lidar2cam'][i]) for i in
                                    range(len(results['lidar2cam']))]

        return results

    def _bboxes_transform(self, bboxes, centers2d, gt_labels, depths, resize, crop, flip):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)

        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        bboxes = bboxes[keep]

        centers2d = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH)
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths

    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths):
        # filter invisible 2d bboxes
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        indices_maps = np.zeros((fH, fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers2d, gt_labels, depths

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@PIPELINES.register_module()
class RandomCropResizeFlipImage(object):
    """Randomly Crop/Resize/Flip Images.

    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, data_aug_conf=None, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        if not 'aug_param' in results.keys():
            results['aug_param'] = {}
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip = self._sample_augmentation(results)

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))

            # augmentation (resize, crop, horizontal flip, rotate)
            # resize, resize_dims, crop, flip, rotate = self._sample_augmentation()  ###different view use different aug (BEV Det)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results['cam2img'][i][:3, :3] = np.matmul(ida_mat, results['cam2img'][i][:3, :3])

        results["img"] = new_imgs
        results['lidar2img'] = [np.matmul(results['cam2img'][i], results['lidar2cam'][i]) for i in
                                range(len(results['lidar2cam']))]

        return results

    def _img_transform(self, img, resize, resize_dims, crop, flip):
        ida_rot = np.eye(2)
        ida_tran = np.zeros(2)
        # adjust image
        # First, randomly resize image, then randomly crop image.
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= np.array(crop[:2])
        ida_mat = np.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self, results):
        # Use the same augmentation setting as previous frames.
        if 'CropResizeFlipImage_param' in results['aug_param'].keys():
            return results['aug_param']['CropResizeFlipImage_param']

        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        crop = self.data_aug_conf["crop"]
        fW, fH = int(crop[2] - crop[0]), int(crop[3] - crop[1])

        if self.training:
            resized_h = random.choice(self.data_aug_conf['reisze'])
            resized_w = resized_h / H * W
            resize = resized_h / H
            resize_dims = (int(resized_w), int(resized_h))

            # Then get the crop parameter.
            newW, newH = resize_dims
            crop_h = int(max(0, newH - fH))
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
        else:
            resized_h = random.choice(self.data_aug_conf["reisze"])
            assert len(self.data_aug_conf["reisze"]) == 1
            resized_w = resized_h / (crop[3] - crop[1]) * (crop[2] - crop[0])
            resize = resized_h / (crop[3] - crop[1])
            resize_dims = (int(resized_w), int(resized_h))
            flip = False
        results['aug_param']['CropResizeFlipImage_param'] = (resize, resize_dims, crop, flip)

        return resize, resize_dims, crop, flip


@PIPELINES.register_module()
class GlobalRotScaleTransImage(object):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(
            self,
            rot_range=[-0.3925, 0.3925],
            scale_ratio_range=[0.95, 1.05],
            translation_std=[0, 0, 0],
            reverse_angle=False,
            training=True,
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            only_gt=False,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

        self.flip_dx_ratio = flip_dx_ratio
        self.flip_dy_ratio = flip_dy_ratio
        self.only_gt = only_gt

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        if not 'aug_param' in results.keys():
            results['aug_param'] = {}

        rot_angle, scale_ratio, flip_dx, flip_dy, _, _ = self._sample_augmentation(results)

        # random rotate
        if not self.only_gt:
            self.rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle *= -1
        results["gt_bboxes_3d"].rotate(
            np.array(rot_angle)
        )

        # random scale
        if not self.only_gt:
            self.scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        # random flip
        if flip_dx:
            if not self.only_gt:
                self.flip_along_x(results)
            results["gt_bboxes_3d"].flip(bev_direction='vertical')
        if flip_dy:
            if not self.only_gt:
                self.flip_along_y(results)
            results["gt_bboxes_3d"].flip(bev_direction='horizontal')

        # TODO: support translation
        return results

    def _sample_augmentation(self, results):
        if 'GlobalRotScaleTransImage_param' in results['aug_param'].keys():
            return results['aug_param']['GlobalRotScaleTransImage_param']
        else:
            rot_angle = np.random.uniform(*self.rot_range) / 180 * np.pi
            scale_ratio = np.random.uniform(*self.scale_ratio_range)
            flip_dx = np.random.uniform() < self.flip_dx_ratio
            flip_dy = np.random.uniform() < self.flip_dy_ratio
        # generate bda_mat 

        rot_sin = torch.sin(torch.tensor(rot_angle))
        rot_cos = torch.cos(torch.tensor(rot_angle))
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        bda_mat = flip_mat @ (scale_mat @ rot_mat)
        bda_mat = torch.inverse(bda_mat)
        results['aug_param']['GlobalRotScaleTransImage_param'] = (
            rot_angle, scale_ratio, flip_dx, flip_dy, bda_mat, self.only_gt)

        return rot_angle, scale_ratio, flip_dx, flip_dy, bda_mat, self.only_gt

    def rotate_bev_along_z(self, results, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array([[rot_cos, -rot_sin, 0, 0], [rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.matmul(results["lidar2img"][view], rot_mat_inv)
            results['lidar2cam'][view] = np.matmul(results['lidar2cam'][view], rot_mat_inv)

        return

    def scale_xyz(self, results, scale_ratio):
        scale_mat = np.array(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        scale_mat_inv = np.linalg.inv(scale_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.matmul(results["lidar2img"][view], scale_mat_inv)
            results['lidar2cam'][view] = np.matmul(results['lidar2cam'][view], scale_mat_inv)
        return

    def flip_along_x(self, results):
        flip_mat = np.array(
            [
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).astype(np.float32)

        flip_mat_inv = np.linalg.inv(flip_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.matmul(results["lidar2img"][view], flip_mat_inv)
            results['lidar2cam'][view] = np.matmul(results['lidar2cam'][view], flip_mat_inv)
        return

    def flip_along_y(self, results):
        flip_mat = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ).astype(np.float32)

        flip_mat_inv = np.linalg.inv(flip_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = np.matmul(results["lidar2img"][view], flip_mat_inv)
            results['lidar2cam'][view] = np.matmul(results['lidar2cam'][view], flip_mat_inv)
        return
