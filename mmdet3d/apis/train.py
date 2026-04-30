# Copyright (c) OpenMMLab. All rights reserved.
import torch
import warnings
import random
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, GradientCumulativeFp16OptimizerHook, GradientCumulativeOptimizerHook)
from mmcv.utils import build_from_cfg

from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger

from mmdet.apis import train_detector
from mmseg.apis import train_segmentor

import copy
from collections import defaultdict
from itertools import chain

from torch.nn.utils import clip_grad

from mmcv.utils import TORCH_VERSION, _BatchNorm, digit_version
from mmcv.runner.dist_utils import allreduce_grads
from mmcv.runner.fp16_utils import LossScaler, wrap_fp16_model
from mmcv.runner.hooks.hook import HOOKS, Hook
from mmcv.runner.hooks.optimizer import OptimizerHook

try:
    # If PyTorch version >= 1.6.0, torch.cuda.amp.GradScaler would be imported
    # and used; otherwise, auto fp16 will adopt mmcv's implementation.
    from torch.cuda.amp import GradScaler
except ImportError:
    pass


@HOOKS.register_module()
class Fp16GanOptimizerHook(OptimizerHook):
    """FP16 optimizer hook (using PyTorch's implementation).

    If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
    to take care of the optimization procedure.

    Args:
        loss_scale (float | str | dict): Scale factor configuration.
            If loss_scale is a float, static loss scaling will be used with
            the specified scale. If loss_scale is a string, it must be
            'dynamic', then dynamic loss scaling will be used.
            It can also be a dict containing arguments of GradScalar.
            Defaults to 512. For Pytorch >= 1.6, mmcv uses official
            implementation of GradScaler. If you use a dict version of
            loss_scale to create GradScaler, please refer to:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
            for the parameters.

    Examples:
        >>> loss_scale = dict(
        ...     init_scale=65536.0,
        ...     growth_factor=2.0,
        ...     backoff_factor=0.5,
        ...     growth_interval=2000
        ... )
        >>> optimizer_hook = Fp16OptimizerHook(loss_scale=loss_scale)
    """

    def __init__(self,
                 grad_clip=None,
                 coalesce=True,
                 bucket_size_mb=-1,
                 loss_scale=512.,
                 distributed=True):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.distributed = distributed
        self._scale_update_param = None
        if loss_scale == 'dynamic':
            self.loss_scaler = GradScaler()
        elif isinstance(loss_scale, float):
            self._scale_update_param = loss_scale
            self.loss_scaler = GradScaler(init_scale=loss_scale)
        elif isinstance(loss_scale, dict):
            self.loss_scaler = GradScaler(**loss_scale)
        else:
            raise ValueError('loss_scale must be of type float, dict, or '
                             f'"dynamic", got {loss_scale}')

    def before_run(self, runner):
        """Preparing steps before Mixed Precision Training."""
        # wrap model mode to fp16
        wrap_fp16_model(runner.model)
        # resume from state dict
        if 'fp16' in runner.meta and 'loss_scaler' in runner.meta['fp16']:
            scaler_state_dict = runner.meta['fp16']['loss_scaler']
            self.loss_scaler.load_state_dict(scaler_state_dict)

    def copy_grads_to_fp32(self, fp16_net, fp32_weights):
        """Copy gradients from fp16 model to fp32 weight copy."""
        for fp32_param, fp16_param in zip(fp32_weights,
                                          fp16_net.parameters()):
            if fp16_param.grad is not None:
                if fp32_param.grad is None:
                    fp32_param.grad = fp32_param.data.new(
                        fp32_param.size())
                fp32_param.grad.copy_(fp16_param.grad)

    def copy_params_to_fp16(self, fp16_net, fp32_weights):
        """Copy updated params from fp32 weight copy to fp16 model."""
        for fp16_param, fp32_param in zip(fp16_net.parameters(),
                                          fp32_weights):
            fp16_param.data.copy_(fp32_param.data)

    def after_train_iter(self, runner):
        """Backward optimization steps for Mixed Precision Training. For
        dynamic loss scaling, please refer to
        https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.

        1. Scale the loss by a scale factor.
        2. Backward the loss to obtain the gradients.
        3. Unscale the optimizer’s gradient tensors.
        4. Call optimizer.step() and update scale factor.
        5. Save loss_scaler state_dict for resume purpose.
        """
        # clear grads of last iteration
        runner.model.zero_grad()
        # if runner._iter % 2 == 0 or runner._iter < 30000:
        if runner._iter % 2 == 0:
            runner.optimizer["g"].zero_grad()

            self.loss_scaler.scale(runner.outputs['loss']).backward()
            self.loss_scaler.unscale_(runner.optimizer["g"])
            # grad clip
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            # backward and update scaler
            self.loss_scaler.step(runner.optimizer["g"])
            self.loss_scaler.update(self._scale_update_param)
        elif runner._iter % 2 == 1:
            runner.optimizer["d"].zero_grad()

            self.loss_scaler.scale(runner.outputs['loss']).backward()
            self.loss_scaler.unscale_(runner.optimizer["d"])
            # grad clip
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])
            # backward and update scaler
            self.loss_scaler.step(runner.optimizer["d"])
            self.loss_scaler.update(self._scale_update_param)

        # save state_dict of loss_scaler
        runner.meta.setdefault(
            'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ['EncoderDecoder3D']:
        train_segmentor(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
    elif 'deepspeed' in cfg.keys():
        train_deepspeed_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
    # elif 'VGGTAlignmentBackbone' in cfg.model.img_backbone.get("type", "OpenCLIPConvnext"):
    #     train_detector_vggtalign(
    #         model,
    #         dataset,
    #         cfg,
    #         distributed=distributed,
    #         validate=validate,
    #         timestamp=timestamp,
    #         meta=meta)
    #     train_detector_moe(
    #         model,
    #         dataset,
    #         cfg,
    #         distributed=distributed,
    #         validate=validate,
    #         timestamp=timestamp,
    #         meta=meta)
    elif cfg.get('pts_pretrained_path', None) is not None:
        train_detector_pts_loading(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
    else:
        train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)


def train_deepspeed_detector(model,
                             dataset,
                             cfg,
                             distributed=False,
                             validate=False,
                             timestamp=None,
                             meta=None):
    from mmdet.utils import (build_ddp, build_dp, build_ZeROddp,
                             find_latest_checkpoint, get_root_logger)
    import os
    logger = get_root_logger(cfg.log_level)
    logger.info("Using deepspeed for training")

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    logger.info("Data loaders built")

    # put model on gpus
    from mmdet.core import build_optimizer
    from mmcv.runner.checkpoint import load_checkpoint
    optimizer = build_optimizer(model, cfg.optimizer)
    deepspeed_enabled = 'deepspeed' in cfg.keys() and cfg.get('deepspeed') is True
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        if deepspeed_enabled:
            if cfg.load_from:
                logger.info("Loading checkpoint from {}".format(cfg.load_from))
                load_checkpoint(model, cfg.load_from, map_location="cpu", strict=False)

                import time
                time.sleep(5)

            cfg.optimizer = optimizer
            cfg.model = model
            model, optimizer, _, _ = build_ZeROddp(
                model=model,
                optimizer=optimizer,
                model_parameters=model.parameters(),
                args=cfg,
            )
            logger.info("Build ZeROddp model and optimizer successfully")
            # if cfg.load_from:
            #     message = model.load_state_dict({"module." + k: v for k, v in new_ckpt.items()}, strict=False)
            #     # print(2, torch.distributed.get_rank(), message)
            #     # logger.info(message)
            #     import time
            #     time.sleep(5)
            model.device_ids = [int(os.environ['LOCAL_RANK'])]
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting

    # optimizer_config = GradientCumulativeFp16OptimizerHook(
    #     cumulative_iters=4, loss_scale='dynamic',
    #     grad_clip=dict(max_norm=35, norm_type=2), distributed=distributed)
    optimizer_config = OptimizerHook(**cfg.optimizer_config)

    if deepspeed_enabled:
        assert isinstance(optimizer_config, OptimizerHook), "deepspeed must use OptimizerHook"

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,  # <class 'mmcv.runner.hooks.optimizer.OptimizerHook'>
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        # when distributed training by epoch, using`DistSamplerSeedHook` to set
        # the different seed to distributed sampler for each epoch, it will
        # shuffle dataset at each epoch and avoid overfitting.
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from and not deepspeed_enabled:
        # DeepSpeed load_from has been handled before build_ZeROddp
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def auto_scale_lr(cfg, distributed, logger):
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): Training config.
        distributed (bool): Using distributed or not.
        logger (logging.Logger): Logger.
    """
    # Get flag from config
    if ('auto_scale_lr' not in cfg) or \
            (not cfg.auto_scale_lr.get('enable', False)):
        logger.info('Automatic scaling of learning rate (LR)'
                    ' has been disabled.')
        return

    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get('base_batch_size', None)
    if base_batch_size is None:
        return

    # Get gpu number
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = len(range(world_size))
    else:
        num_gpus = len(cfg.gpu_ids)

    # calculate the batch size
    batch_size = num_gpus * cfg.data.samples_per_gpu
    logger.info(f'You are using {num_gpus} GPU(s) '
                f'and {cfg.data.samples_per_gpu} samples per GPU. '
                f'Total batch size is {batch_size}.')

    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / base_batch_size) * cfg.optimizer.lr
        logger.info('LR has been automatically scaled '
                    f'from {cfg.optimizer.lr} to {scaled_lr}')
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info('The batch size match the '
                    f'base batch size: {base_batch_size}, '
                    f'will not scaling the LR ({cfg.optimizer.lr}).')


def train_detector_moe(model,
                       dataset,
                       cfg,
                       distributed=False,
                       validate=False,
                       timestamp=None,
                       meta=None):
    import torch.distributed as dist
    from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                             Fp16OptimizerHook, OptimizerHook, build_optimizer,
                             build_runner, get_dist_info)

    from mmdet.core import DistEvalHook, EvalHook
    from mmdet.datasets import (build_dataloader, build_dataset,
                                replace_ImageToTensor)
    from mmdet.utils import (build_ddp, build_dp, compat_cfg,
                             find_latest_checkpoint, get_root_logger)
    import os

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    if cfg.load_from:
        state_dict = torch.load(cfg.load_from, map_location='cpu')['state_dict']
        # Filter the state_dict to match the model's keys
        # 1. "pts_bbox_head.llm.llm.model." --> "pts_bbox_head.llm.llm.base_model.model.model."
        # 2. copy the key including "feed_forward." to "feed_forward_vis."
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('pts_bbox_head.llm.llm.model.'):
                new_k = k.replace('pts_bbox_head.llm.llm.model.', 'pts_bbox_head.llm.llm.base_model.model.model.')
                new_state_dict[new_k] = v
                if 'feed_forward.' in new_k:
                    # If the key contains 'feed_forward.', replace it with 'feed_forward_vis.'
                    new_k = new_k.replace('feed_forward.', 'feed_forward_vis.')
                    new_state_dict[new_k] = v
                    logger.info(f"Copying {k} to {new_k} in the state_dict")
            else:
                new_state_dict[k] = v

        message = model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Load checkpoint from {cfg.load_from}, message: {message}")
        import time
        time.sleep(2)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow)


def train_detector_pts_loading(model,
                               dataset,
                               cfg,
                               distributed=False,
                               validate=False,
                               timestamp=None,
                               meta=None):
    import torch.distributed as dist
    from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                             Fp16OptimizerHook, OptimizerHook, build_optimizer,
                             build_runner, get_dist_info)

    from mmdet.core import DistEvalHook, EvalHook
    from mmdet.datasets import (build_dataloader, build_dataset,
                                replace_ImageToTensor)
    from mmdet.utils import (build_ddp, build_dp, compat_cfg,
                             find_latest_checkpoint, get_root_logger)
    import os

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    if cfg.pts_pretrained_path is not None:
        state_dict = torch.load(cfg.pts_pretrained_path, map_location='cpu')['state_dict']
        # Filter the state_dict to match the model's keys
        logger.info(f"Loading pts pretrained weights from {cfg.pts_pretrained_path}")
        new_state_dict = {}
        # 1. the keys including "pts_voxel_encoder", "pts_middle_encoder", "pts_backbone", "pts_neck" are loaded.
        target_keys = ['pts_voxel_encoder', 'pts_middle_encoder', 'pts_backbone', 'pts_neck']
        for k, v in state_dict.items():
            if any(tk in k for tk in target_keys):
                new_state_dict[k] = v
                logger.info(f"Copying {k} to the model")

        message = model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Load checkpoint from {cfg.pts_pretrained_path}, message: {message}")
        import time
        time.sleep(2)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow)


def train_detector_vggtalign(model,
                             dataset,
                             cfg,
                             distributed=False,
                             validate=False,
                             timestamp=None,
                             meta=None):
    import torch.distributed as dist
    from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                             Fp16OptimizerHook, OptimizerHook, build_optimizer,
                             build_runner, get_dist_info)

    from mmdet.core import DistEvalHook, EvalHook
    from mmdet.datasets import (build_dataloader, build_dataset,
                                replace_ImageToTensor)
    from mmdet.utils import (build_ddp, build_dp, compat_cfg,
                             find_latest_checkpoint, get_root_logger)
    import os

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    if cfg.load_from:
        state_dict = torch.load(cfg.load_from, map_location='cpu')['state_dict']
        # Filter the state_dict to match the model's keys
        # 1. "img_backbone.model." --> "img_backbone.semantic_encoder.model."
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('img_backbone.model.'):
                new_k = k.replace('img_backbone.model.', 'img_backbone.semantic_encoder.model.')
                new_state_dict[new_k] = v
                logger.info(f"Copying {k} to {new_k} in the state_dict")
            else:
                new_state_dict[k] = v

        message = model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Load checkpoint from {cfg.load_from}, message: {message}")
        import time
        time.sleep(2)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow)
