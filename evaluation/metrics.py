import torch
import numpy as np
from einops import rearrange, repeat
from typing import Dict, Tuple
from datasets.datatypes import SliceData
from training.datatypes import Prediction
from . import tapvid3d_metrics, tapvid2d_metrics
from utils.common_utils import ensure_float32

def apply_intrinsics(pts: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    pts_proj = torch.einsum('...ij,...j->...i', intrinsics, pts)
    return pts_proj[..., :2] / pts_proj[..., 2:3]

def get_new_hw_with_given_smallest_side_length(
    *, orig_resolution: torch.Tensor, smallest_side_length: int = 256
):
  scaling_factor = torch.full_like(orig_resolution[..., 0], smallest_side_length) / orig_resolution.min(dim=-1).values
  return scaling_factor

@ensure_float32(allow_cast=False)
def _compute_metrics(sample: SliceData, preds: Prediction, bidirectional: bool) -> Dict[str, torch.Tensor]:
    sample = sample.copy().with_annot_mode(None)
    gt_sample = sample.with_annot_mode('gt')
    est_sample = sample.with_annot_mode('est')

    gt_tracks = torch.einsum(
        "b t i j, b t n j -> b t n i",
        gt_sample.extrinsics,
        torch.cat([gt_sample.trajs_3d, torch.ones_like(gt_sample.trajs_3d[..., :1])], dim=-1),
    )[..., :3]
    pred_tracks = torch.einsum(
        "b t i j, b t n j -> b t n i",
        est_sample.extrinsics,
        torch.cat([preds.coords, torch.ones_like(preds.coords[..., :1])], dim=-1),
    )[..., :3]

    if torch.isnan(preds.coords).any():
        import pudb.remote; pudb.remote.set_trace()

    B, T, C, H, W = sample.rgbs.shape

    scaling_factor = (
        get_new_hw_with_given_smallest_side_length(
            orig_resolution=sample.orig_resolution,
            smallest_side_length=256,
        )
    )

    gt_resized_intrinsics = gt_sample.intrinsics.clone()
    gt_resized_intrinsics[..., 0, 0] *= ((256 - 1) / (W - 1))
    gt_resized_intrinsics[..., 1, 1] *= ((256 - 1) / (H - 1))
    gt_resized_intrinsics[..., 0, 2] *= ((256 - 1) / (W - 1))
    gt_resized_intrinsics[..., 1, 2] *= ((256 - 1) / (H - 1))

    est_resized_intrinsics = est_sample.intrinsics.clone()
    est_resized_intrinsics[..., 0, 0] *= ((256 - 1) / (W - 1))
    est_resized_intrinsics[..., 1, 1] *= ((256 - 1) / (H - 1))
    est_resized_intrinsics[..., 0, 2] *= ((256 - 1) / (W - 1))
    est_resized_intrinsics[..., 1, 2] *= ((256 - 1) / (H - 1))

    # to match exactly with the implementation in https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid3d/evaluation/evaluate_model.py
    gt_resized_intrinsics_iso = gt_sample.intrinsics.clone()
    gt_resized_intrinsics_iso[..., 0, 0] *= (((sample.orig_resolution[..., 1] - 1) / (W - 1)) * scaling_factor)[:, None]
    gt_resized_intrinsics_iso[..., 1, 1] *= (((sample.orig_resolution[..., 0] - 1) / (H - 1)) * scaling_factor)[:, None]
    gt_resized_intrinsics_iso[..., 0, 2] *= (((sample.orig_resolution[..., 1] - 1) / (W - 1)) * scaling_factor)[:, None]
    gt_resized_intrinsics_iso[..., 1, 2] *= (((sample.orig_resolution[..., 0] - 1) / (H - 1)) * scaling_factor)[:, None]
    
    gt_resized_intrinsics_params_iso = torch.stack([
        gt_resized_intrinsics_iso[..., 0, 0],
        gt_resized_intrinsics_iso[..., 1, 1],
        gt_resized_intrinsics_iso[..., 0, 2],
        gt_resized_intrinsics_iso[..., 1, 2],
    ], dim=-1).cpu().numpy()

    gt_query_points = gt_sample.query_point.clone()
    est_query_points = est_sample.query_point.clone()
    gt_query_frames = gt_query_points[..., 0].long()
    est_query_frames = est_query_points[..., 0].long()
    gt_extrinsics_at_query_frames = gt_sample.extrinsics[repeat(torch.arange(B), "b -> b n", n=gt_query_frames.shape[1]), gt_query_frames]
    est_extrinsics_at_query_frames = est_sample.extrinsics[repeat(torch.arange(B), "b -> b n", n=est_query_frames.shape[1]), est_query_frames]
    gt_resized_intrinsics_at_query_frames = gt_resized_intrinsics[repeat(torch.arange(B), "b -> b n", n=gt_query_frames.shape[1]), gt_query_frames]
    est_resized_intrinsics_at_query_frames = est_resized_intrinsics[repeat(torch.arange(B), "b -> b n", n=est_query_frames.shape[1]), est_query_frames]
    gt_query_points[..., 1:] = torch.einsum("b n i j, b n j -> b n i",
        gt_extrinsics_at_query_frames,
        torch.cat([gt_query_points[..., 1:], torch.ones_like(gt_query_points[..., :1])], dim=-1),
    )[..., :3]
    est_query_points[..., 1:] = torch.einsum("b n i j, b n j -> b n i",
        est_extrinsics_at_query_frames,
        torch.cat([est_query_points[..., 1:], torch.ones_like(est_query_points[..., :1])], dim=-1),
    )[..., :3]

    gt_query_points_2d_resized = torch.cat(
        [
            gt_query_points[..., :1], 
            apply_intrinsics(gt_query_points[..., 1:], gt_resized_intrinsics_at_query_frames)
        ], 
        dim=-1
    ).cpu().numpy()

    gt_tracks_2d_resized = apply_intrinsics(gt_tracks, repeat(gt_resized_intrinsics, "b t i j -> b t n i j", n=gt_tracks.shape[2]))
    pred_tracks_2d_resized = apply_intrinsics(pred_tracks, repeat(est_resized_intrinsics, "b t i j -> b t n i j", n=pred_tracks.shape[2]))

    gt_query_points = gt_query_points.cpu().numpy()
    est_query_points = est_query_points.cpu().numpy()
    gt_tracks = gt_tracks.cpu().numpy()
    pred_tracks = pred_tracks.cpu().numpy()
    gt_tracks_2d_resized = gt_tracks_2d_resized.cpu().numpy()
    pred_tracks_2d_resized = pred_tracks_2d_resized.cpu().numpy()
    
    metrics_dict = {}

    gt_occluded = ~sample.visibs.cpu().numpy()
    valids = sample.valids.cpu().numpy()
    visib_probs = (torch.sigmoid(preds.visibs)).cpu().numpy()
    vis_related_keys = set()

    for threshold in np.linspace(0, 1, 11)[1:-1]:
        pred_occluded = visib_probs <= threshold
        metrics_dict_ = {}

        tapvid3d_256_iso_metrics: Dict[str, np.ndarray] = tapvid3d_metrics.compute_tapvid3d_metrics(
            gt_occluded=gt_occluded,
            gt_tracks=gt_tracks,
            pred_occluded=pred_occluded,
            pred_tracks=pred_tracks,
            query_points=gt_query_points,
            intrinsics_params=gt_resized_intrinsics_params_iso,
            valids=valids,
            order='b t n',
            bidirectional=bidirectional
        ) # type: ignore
        metrics_dict_.update({f"tapvid3d_{k}": v for k, v in tapvid3d_256_iso_metrics.items()})
        
        tapvid2d_256_metrics: Dict[str, np.ndarray] = tapvid2d_metrics.compute_tapvid_metrics(
            query_points=gt_query_points_2d_resized[..., [0, 2, 1]], # [t, x, y] -> [t, y, x]
            gt_occluded=rearrange(gt_occluded, "b t n -> b n t"),
            gt_tracks=rearrange(gt_tracks_2d_resized, "b t n i -> b n t i"),
            pred_occluded=repeat(pred_occluded, "b t n -> b n t"),
            pred_tracks=rearrange(pred_tracks_2d_resized, "b t n i -> b n t i"),
            query_mode="strided" if bidirectional else "first",
            valids=rearrange(valids, "b t n -> b n t"),
            get_trackwise_metrics=False,
        ) # type: ignore
        metrics_dict_.update({f"tapvid2d_{k}": v for k, v in tapvid2d_256_metrics.items()})

        vis_related_keys_ = [k for k in metrics_dict_.keys() if 'occlusion_accuracy' in k or 'jaccard' in k]
        vis_related_keys.update(vis_related_keys_)
        vis_related_metrics = {k: metrics_dict_.pop(k) for k in vis_related_keys_}
        if len(metrics_dict) == 0:
            metrics_dict = metrics_dict_
        for k, v in vis_related_metrics.items():
            metrics_dict[f'{k}_visthr_{threshold:.2f}'] = v

    metrics_dict = {
        k: torch.tensor(v, device=sample.rgbs.device, dtype=torch.float32)
        for k, v in metrics_dict.items()
    }
    return metrics_dict


@ensure_float32(allow_cast=False)
def compute_metrics(sample: SliceData, preds: Prediction, bidirectional: bool) -> Dict[str, torch.Tensor]:
    metrics_dict_base = _compute_metrics(sample, preds, bidirectional)

    occ_sample = sample.copy()
    occ_sample.visibs = ~sample.visibs & sample.valids
    occ_metrics_dict = _compute_metrics(occ_sample, preds, bidirectional)
    occ_metrics_dict = {f"occ_only_{k}": v for k, v in occ_metrics_dict.items()}

    metrics_dict_base.update(occ_metrics_dict)

    return metrics_dict_base
