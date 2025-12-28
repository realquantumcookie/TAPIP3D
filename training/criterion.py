# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

from collections import defaultdict
import einops
import torch
import numpy as np
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Tuple

from datasets.datatypes import SliceData
from training.datatypes import TrainData

from utils.common_utils import ensure_float32

class TrajectoryCriterion:
    def __init__(
        self,
        gamma: float,
        bce_weight: float,
        train_iters: int,
        eval_iters: int,
        traj_weight: float = 0.,
        conf_weight: float = 0.,
        loss_type: str = "l21",
        scale_loss_by_depth: bool = False,
        traj_visible_weight: float = 0.,
        traj_invisible_weight: float = 0.,
        pixel_loss_weight: float = 0.,
        inverse_depth_loss_weight: float = 0.,
        depth_min: float = 0.1,
        depth_max: float = 100.,
        uv_logdepth_space: bool = False,
    ):
        self.gamma = gamma
        self.traj_weight = traj_weight
        self.traj_visible_weight = traj_visible_weight
        self.traj_invisible_weight = traj_invisible_weight
        self.bce_weight = bce_weight
        self.train_iters = train_iters
        self.eval_iters = eval_iters
        self.scale_loss_by_depth = scale_loss_by_depth
        self.loss_type = loss_type
        self.pixel_loss_weight = pixel_loss_weight
        self.inverse_depth_loss_weight = inverse_depth_loss_weight
        self.uv_logdepth_space = uv_logdepth_space
        self.conf_weight = conf_weight
        self.depth_min = depth_min
        self.depth_max = depth_max

        if self.uv_logdepth_space:
            assert not self.scale_loss_by_depth and self.pixel_loss_weight == 0, "uv_logdepth_space is not supported with scale_loss_by_depth or pixel_loss_weight"

    @ensure_float32(allow_cast=False)
    def _traj_loss_func(self, gt_local, gt_pixel, pd_local, pd_pixel, gt_inverse_depth, pd_inverse_depth):

        if self.loss_type == "l21":
            loss_3d = (pd_local - gt_local).norm(dim=-1)
        elif self.loss_type == "l1":
            loss_3d = (pd_local - gt_local).abs().sum(dim=-1)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        if self.pixel_loss_weight > 0:
            # we always use l1 for pixel loss
            loss_pixel = (pd_pixel - gt_pixel).abs().sum(dim=-1)
        else:
            loss_pixel = None
        
        if self.inverse_depth_loss_weight > 0:
            loss_inverse_depth = (gt_inverse_depth - pd_inverse_depth).abs()
        else:
            loss_inverse_depth = None

        return loss_3d, loss_pixel, loss_inverse_depth
    
    @ensure_float32(allow_cast=False)
    def _compute_traj_loss(self, sample: SliceData, train_data: TrainData, num_iters: int) -> Dict[str, Any]:

        gt_sample = sample.with_annot_mode('gt')
        est_sample = sample.with_annot_mode('est')

        gamma_weight = (self.gamma ** (num_iters - 1 - train_data.iter_idx))

        if sample.same_scale:
            pd_to_gt_scale = torch.ones_like(gt_sample.trajs_3d[:, 0, 0, 0])
        else:
            gt_depths = gt_sample.depths.clone()
            est_depths = est_sample.depths.clone()
            gt_depths[gt_depths == 0] = torch.nan
            est_depths[est_depths == 0] = torch.nan
            pd_to_gt_scale = torch.nanmedian((gt_depths / est_depths).reshape(gt_depths.shape[0], -1), dim=-1).values

        gt_local = torch.einsum(
            "b t i j, b t n j -> b t n i",
            gt_sample.extrinsics,
            torch.cat([gt_sample.trajs_3d, torch.ones_like(gt_sample.trajs_3d[..., :1])], dim=-1),
        )[..., :3]
        if self.pixel_loss_weight > 0 or self.conf_weight > 0 or self.uv_logdepth_space:
            gt_pixel = torch.einsum(
                "b t i j, b t n j -> b t n i",
                gt_sample.intrinsics,
                gt_local,
            )
            gt_pixel = gt_pixel[..., :2] / torch.clamp(gt_pixel[..., 2:3], min=1e-4)
        else:
            gt_pixel = None
        
        focals = torch.sqrt(gt_sample.intrinsics[..., 0, 0] * gt_sample.intrinsics[..., 1, 1])

        if self.uv_logdepth_space:
            logdepth = torch.log(torch.clamp(gt_local[..., -1], min=self.depth_min, max=self.depth_max))
            gt_uv_logdepth = torch.cat([gt_pixel, logdepth[..., None] * focals[:, :, None, None]], dim=-1)
        else:
            gt_uv_logdepth = None

        if self.inverse_depth_loss_weight > 0:
            gt_inverse_depth = 1.0 / torch.clamp(gt_local[..., -1], min=self.depth_min, max=self.depth_max)
        else:
            gt_inverse_depth = None

        def calc_loss(pd_traj, confs):
            pd_local = torch.einsum(
                "b t i j, b t n j -> b t n i",
                est_sample.extrinsics,
                torch.cat([pd_traj, torch.ones_like(pd_traj[..., :1])], dim=-1),
            )[..., :3]

            if self.pixel_loss_weight > 0 or self.conf_weight > 0 or self.uv_logdepth_space:
                pd_pixel = torch.einsum(
                    "b t i j, b t n j -> b t n i",
                    est_sample.intrinsics,
                    pd_local,
                )
                # import ipdb; ipdb.set_trace()
                pd_pixel = pd_pixel[..., :2] / torch.clamp(pd_pixel[..., 2:3], min=1e-4)
            else:
                pd_pixel = None

            # everything blow this line is 3d loss
            pd_local = pd_local * pd_to_gt_scale[:, None, None, None]

            if self.inverse_depth_loss_weight > 0:
                pd_inverse_depth = 1.0 / torch.clamp(pd_local[..., -1], min=self.depth_min, max=self.depth_max)
            else:
                pd_inverse_depth = None

            if self.uv_logdepth_space:
                pd_logdepth = torch.log(torch.clamp(pd_local[..., -1], min=self.depth_min, max=self.depth_max))
                pd_uv_logdepth = torch.cat([pd_pixel, pd_logdepth[..., None] * focals[:, :, None, None]], dim=-1)
            else:
                pd_uv_logdepth = None

            loss_3d, loss_pixel, loss_inverse_depth = self._traj_loss_func(
                gt_local=gt_local if not self.uv_logdepth_space else gt_uv_logdepth,
                gt_pixel=gt_pixel if not self.uv_logdepth_space else None,
                pd_local=pd_local if not self.uv_logdepth_space else pd_uv_logdepth,
                pd_pixel=pd_pixel if not self.uv_logdepth_space else None,
                gt_inverse_depth=gt_inverse_depth,
                pd_inverse_depth=pd_inverse_depth,
            )

            visible_mask = sample.visibs & sample.valids
            invisible_mask = ~sample.visibs & sample.valids

            if self.scale_loss_by_depth:
                # In this case, it's user's responsibility to mask all points with negati0ve depth
                focal = torch.sqrt(gt_sample.intrinsics[..., 0, 0] * gt_sample.intrinsics[..., 1, 1])
                loss_3d = loss_3d * focal[..., None] / torch.clamp(gt_local[..., 2], min=self.depth_min) # Normalize to match with the APD metric
            
            loss_3d_visible = (loss_3d * visible_mask.float()).sum() / (visible_mask.sum() + 1e-4)
            loss_3d_invisible = (loss_3d * invisible_mask.float()).sum() / (invisible_mask.sum() + 1e-4)
            loss_3d_all = (loss_3d * sample.valids.float()).sum() / (sample.valids.sum() + 1e-4)

            if self.conf_weight > 0:
                pixel_diff = (gt_pixel.detach() - pd_pixel.detach()).norm(dim=-1)
                labels = (pixel_diff < 12.).float()
                loss_conf = F.binary_cross_entropy_with_logits(confs, labels, reduction="none")
                loss_conf = (loss_conf * sample.valids.float()).sum() / (sample.valids.sum() + 1e-4)

            if self.pixel_loss_weight > 0:
                pixel_mask = sample.valids & (gt_local[..., 2] > self.depth_min) & (pd_local[..., 2] > self.depth_min)
                loss_pixel_avg = (loss_pixel * pixel_mask.float()).sum() / (pixel_mask.sum() + 1e-4)
            
            if self.inverse_depth_loss_weight > 0:
                loss_inverse_depth_avg = (loss_inverse_depth * sample.valids.float()).sum() / (sample.valids.sum() + 1e-4)
            
            ret = {
                "3d_visible": loss_3d_visible,
                "3d_invisible": loss_3d_invisible,
                "3d_all": loss_3d_all,
            }

            if self.pixel_loss_weight > 0:
                ret["pixel"] = loss_pixel_avg
            
            if self.inverse_depth_loss_weight > 0:
                ret["inverse_depth"] = loss_inverse_depth_avg

            if self.conf_weight > 0:
                ret["conf"] = loss_conf

            return ret

        losses_dict = defaultdict(list)
        if train_data.confs is not None:
            assert len (train_data.coords) == 1
        for internal_coord in train_data.coords:
            loss_dict = calc_loss(internal_coord, train_data.confs) # type: ignore
            for k, v in loss_dict.items():
                losses_dict[k].append(gamma_weight * v)
        
        return losses_dict

    @ensure_float32(allow_cast=False)
    def _compute_bce_loss(self, sample: SliceData, train_data: TrainData, num_iters: int) -> torch.Tensor:
        traj_mask = sample.valids

        if train_data.visibs is None:
            return sample.trajs_3d.new_zeros(tuple())

        gamma_weight = (self.gamma ** (num_iters - 1 - train_data.iter_idx))

        bce_loss = F.binary_cross_entropy_with_logits(train_data.visibs, sample.visibs.float(), reduction="none")
        bce_loss = (bce_loss * traj_mask.float()).sum() / (traj_mask.sum() + 1e-4)

        return gamma_weight * bce_loss

    @ensure_float32(allow_cast=False)
    def compute_loss(self, sample: SliceData, train_data: TrainData, bidirectional: bool, is_train: bool) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = sample.copy().with_annot_mode(None)
        
        if not bidirectional:
            sample = sample.with_causal_mask()

        sample = sample.time_slice(train_data.frame_range[0], train_data.frame_range[1])

        num_iters = self.train_iters if is_train else self.eval_iters

        traj_losses_dict = self._compute_traj_loss(sample, train_data, num_iters)
        traj_loss_dict = {k: torch.mean(torch.stack(v)) for k, v in traj_losses_dict.items()}

        bce_loss = self._compute_bce_loss(sample, train_data, num_iters)
        assert (train_data.confs is not None) == (self.conf_weight > 0)
        
        total_loss = (
            traj_loss_dict['3d_visible'] * self.traj_visible_weight + 
            traj_loss_dict['3d_invisible'] * self.traj_invisible_weight + 
            traj_loss_dict['3d_all'] * self.traj_weight + 
            bce_loss * self.bce_weight
        )
        if self.pixel_loss_weight > 0:
            total_loss += traj_loss_dict['pixel'] * self.pixel_loss_weight # type: ignore

        if self.inverse_depth_loss_weight > 0:
            total_loss += traj_loss_dict['inverse_depth'] * self.inverse_depth_loss_weight # type: ignore

        if self.conf_weight > 0:
            total_loss += traj_loss_dict['conf'] * self.conf_weight

        train_info = {}
        if self.traj_weight > 0:
            train_info['traj_loss_all'] = traj_loss_dict['3d_all']
            for i, loss in enumerate(traj_losses_dict['3d_all']):
                train_info[f'internal_traj_loss_all_{i}'] = loss
        if self.bce_weight > 0:
            train_info['bce_loss'] = bce_loss
        if self.traj_visible_weight > 0:
            train_info['traj_loss_visible'] = traj_loss_dict['3d_visible']
            for i, loss in enumerate(traj_losses_dict['3d_visible']):
                train_info[f'internal_traj_loss_visible_{i}'] = loss
        if self.traj_invisible_weight > 0:
            train_info['traj_loss_invisible'] = traj_loss_dict['3d_invisible']
            for i, loss in enumerate(traj_losses_dict['3d_invisible']):
                train_info[f'internal_traj_loss_invisible_{i}'] = loss
        if self.pixel_loss_weight > 0:
            train_info['traj_loss_pixel'] = traj_loss_dict['pixel']
            for i, loss in enumerate(traj_losses_dict['pixel']): # type: ignore
                train_info[f'internal_traj_loss_pixel_{i}'] = loss

        if self.conf_weight > 0:
            train_info['conf_loss'] = traj_loss_dict['conf']

        if self.inverse_depth_loss_weight > 0:
            train_info['traj_loss_inverse_depth'] = traj_loss_dict['inverse_depth']
            for i, loss in enumerate(traj_losses_dict['inverse_depth']): # type: ignore
                train_info[f'internal_traj_loss_inverse_depth_{i}'] = loss

        return total_loss, train_info