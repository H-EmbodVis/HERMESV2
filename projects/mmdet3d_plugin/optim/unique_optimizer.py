import warnings
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
from torch.nn import GroupNorm, LayerNorm
from mmcv.utils import (_BatchNorm, _InstanceNorm,
                        build_from_cfg, is_list_of)
from mmcv.utils.ext_loader import check_ops_exist
from mmcv.runner.optimizer.default_constructor import DefaultOptimizerConstructor
from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS, OPTIMIZERS


@OPTIMIZER_BUILDERS.register_module()
class UniqueOptimizerConstructor(DefaultOptimizerConstructor):
    """
    An enhanced optimizer constructor that:
      1. Skips duplicated Parameters caused by weight tying (e.g. embedding <-> lm_head).
      2. Keeps the first occurrence's hyper-parameters.
      3. Optionally merges groups with identical optimizer hyper-parameter signature.
    Usage: set constructor='UniqueOptimizerConstructor' in optimizer config.
    """

    def __init__(self,
                 optimizer_cfg: Dict,
                 paramwise_cfg: Optional[Dict] = None):
        super().__init__(optimizer_cfg, paramwise_cfg)
        # internal containers
        self._visited_param_ids = set()
        self._duplicate_skipped = 0

    def _register_param(self,
                        params: List[Dict],
                        param_group: Dict,
                        prefix: str,
                        name: str,
                        param: torch.nn.Parameter,
                        bypass_duplicate: bool) -> None:
        """
        Register a single parameter into params list with duplicate protection.
        """
        pid = id(param)
        if pid in self._visited_param_ids and bypass_duplicate:
            # only warn once per real duplication path
            warnings.warn(
                f'[UniqueOptimizer] Skip duplicate tied param: {prefix}.{name}')
            self._duplicate_skipped += 1
            return
        self._visited_param_ids.add(pid)
        params.append(param_group)

    def add_params(self,
                   params: List[Dict],
                   module: nn.Module,
                   prefix: str = '',
                   is_dcn_module: Union[int, float, None] = None) -> None:
        """
        Same interface as base, but inject duplicate filtering.
        """
        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get('bias_lr_mult', 1.)
        bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', 1.)
        norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', 1.)
        dwconv_decay_mult = self.paramwise_cfg.get('dwconv_decay_mult', 1.)
        bypass_duplicate = self.paramwise_cfg.get('bypass_duplicate', True)  # force True by default here
        dcn_offset_lr_mult = self.paramwise_cfg.get('dcn_offset_lr_mult', 1.)

        is_norm = isinstance(
            module, (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        is_dwconv = (isinstance(module, torch.nn.Conv2d)
                     and module.in_channels == module.groups)

        for name, param in module.named_parameters(recurse=False):
            # build base group
            param_group = {'params': [param]}
            if not param.requires_grad:
                # still record (harmless); will be filtered by optimizer anyway
                self._register_param(params, param_group, prefix, name,
                                     param, bypass_duplicate)
                continue

            # custom key exact longest substring priority
            is_custom = False
            for key in sorted_keys:
                full_name = f'{prefix}.{name}' if prefix else name
                if key in full_name:
                    is_custom = True
                    lr_mult = custom_keys[key].get('lr_mult', 1.)
                    if self.base_lr is not None:
                        param_group['lr'] = self.base_lr * lr_mult
                    if self.base_wd is not None:
                        decay_mult = custom_keys[key].get('decay_mult', 1.)
                        param_group['weight_decay'] = self.base_wd * decay_mult
                    break

            if not is_custom:
                # bias lr multiplier
                if (name == 'bias') and not (is_norm or is_dcn_module):
                    if self.base_lr is not None:
                        param_group['lr'] = self.base_lr * bias_lr_mult

                # DCN offset special lr
                if (prefix.find('conv_offset') != -1 and is_dcn_module
                        and isinstance(module, torch.nn.Conv2d)):
                    if self.base_lr is not None:
                        param_group['lr'] = self.base_lr * dcn_offset_lr_mult

                # weight decay handling
                if self.base_wd is not None:
                    if is_norm:
                        param_group['weight_decay'] = self.base_wd * norm_decay_mult
                    elif is_dwconv:
                        param_group['weight_decay'] = self.base_wd * dwconv_decay_mult
                    elif name == 'bias' and not is_dcn_module:
                        param_group['weight_decay'] = self.base_wd * bias_decay_mult

            # register with duplicate filtering
            self._register_param(params, param_group, prefix, name, param,
                                 bypass_duplicate)

        # detect DCN
        if check_ops_exist():
            from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
            is_dcn_module = isinstance(
                module, (DeformConv2d, ModulatedDeformConv2d))
        else:
            is_dcn_module = False

        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.add_params(params,
                            child_mod,
                            prefix=child_prefix,
                            is_dcn_module=is_dcn_module)

    @staticmethod
    def _group_signature(g: Dict) -> Tuple:
        """
        Build a hashable signature for merging param groups with identical hyper-parameters.
        """
        # keys that may appear
        keys = ['lr', 'weight_decay', 'betas', 'eps', 'momentum']
        sig = []
        for k in keys:
            sig.append((k, g[k]) if k in g else (k, None))
        return tuple(sig)

    def _merge_identical_groups(self,
                                params: List[Dict],
                                enable: bool = True) -> List[Dict]:
        """
        Merge param groups that share exactly the same optimizer hyper-parameters.
        """
        if not enable:
            return params
        merged = {}
        for g in params:
            sig = self._group_signature(g)
            if sig not in merged:
                # shallow copy except param list
                new_g = {k: v for k, v in g.items() if k != 'params'}
                new_g['params'] = []
                merged[sig] = new_g
            merged[sig]['params'].extend(g['params'])
        return list(merged.values())

    def __call__(self, model: nn.Module):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        if not self.paramwise_cfg:
            optimizer_cfg['params'] = model.parameters()
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

        params: List[Dict] = []
        self.add_params(params, model)

        # Optional second-pass hard dedup (safety net)
        seen = set()
        cleaned = []
        removed = 0
        for g in params:
            new_list = []
            for p in g['params']:
                pid = id(p)
                if pid in seen:
                    removed += 1
                    continue
                seen.add(pid)
                new_list.append(p)
            if new_list:
                h = {k: v for k, v in g.items() if k != 'params'}
                h['params'] = new_list
                cleaned.append(h)

        # Merge identical groups to reduce overhead (optional)
        cleaned = self._merge_identical_groups(
            cleaned,
            enable=self.paramwise_cfg.get('merge_identical', True))

        if self._duplicate_skipped or removed:
            print(f'[UniqueOptimizer] duplicate_skipped={self._duplicate_skipped} '
                  f'post_removed={removed} final_groups={len(cleaned)}')

        optimizer_cfg['params'] = cleaned
        return build_from_cfg(optimizer_cfg, OPTIMIZERS)