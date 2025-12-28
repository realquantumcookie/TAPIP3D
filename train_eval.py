# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

import time
import re
import os
from einops import repeat, rearrange
from omegaconf import DictConfig, OmegaConf
import logging
from multiprocessing import Process
from tqdm import tqdm

import hydra
import numpy as np
import torch
import torch.multiprocessing
import pickle

import wandb

torch.multiprocessing.set_sharing_strategy('file_system')

from training.datatypes import Prediction, TrainData
from utils.inference_utils import _inference_with_grid

from datasets.datatypes import SliceData
import models
from datasets.base_dataset import BaseDataset
from utils.common_utils import (
    batch_project, count_parameters, setup_logger
)
from training.engine import BaseTrainTester, get_rank
from third_party.cotracker.visualizer import Visualizer
from training.criterion import TrajectoryCriterion
import utils.rerun_visualizer as visualizer
import evaluation.metrics as metrics
import torch.profiler as profiler

logger = logging.getLogger(__name__)

class TrainTester(BaseTrainTester):
    """Train/test a trajectory optimization algorithm."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_criterion(self):
        return TrajectoryCriterion(train_iters=self.cfg.train.train_iters, eval_iters=self.cfg.train.eval_iters, ** self.cfg.criterion)

    def get_datasets(self):
        """Initialize datasets."""
        # Initialize datasets with arguments
        if not self.cfg.train.get('eval_only', False):
            train_dataset_cfg = self.cfg.train_dataset.copy()
            if self.cfg.train.get('additional_transform'):
                train_dataset_cfg.transform = self.cfg.train_dataset.transform + self.cfg.train.additional_transform
            train_dataset = BaseDataset.from_config(
                train_dataset_cfg,
            )
        else:
            train_dataset = None
        test_datasets = {}
        for key, cfg in self.cfg.test_datasets.items():
            cfg = cfg.copy()
            if self.cfg.get('eval', {}).get('additional_transforms', []):
                if 'transform' in cfg:
                    cfg.transform = cfg.transform + self.cfg.eval.get('additional_transforms', [])
                    test_datasets[key] = BaseDataset.from_config(cfg)
                else:
                    test_datasets[key] = BaseDataset.from_config(cfg, transform=self.cfg.eval.get('additional_transforms', []))
            else:
                test_datasets[key] = BaseDataset.from_config(cfg)
        return train_dataset, test_datasets

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        _model = models.from_config(
            self.cfg.model,
            image_size=tuple(self.cfg.train_dataset.resolution),
        )
        logger.info(f"Model initialized with {count_parameters(_model):,} trainable parameters")
        return _model

    def train_one_step(self, model, criterion, step_id, sample):
        """Run a single training step."""

        sample.skip_post_init = True
        est_sample = sample.with_annot_mode('est')
        assert est_sample.depth_roi is None, "depth_roi should be None for training"
            
        # Forward pass
        if self.cfg.train.streaming_backward:
            pred = None
            train_data_list = []

            with self.accelerator.autocast():
                feats = model.encode_rgbs(est_sample.rgbs)

            feats_param = torch.nn.Parameter(feats.detach().clone(), requires_grad=True)

            generator = model.streaming_forward(
                rgb_obs=est_sample.rgbs,
                depth_obs=est_sample.depths,
                num_iters=self.cfg.train.train_iters,
                query_point=est_sample.query_point,
                intrinsics=est_sample.intrinsics,
                extrinsics=est_sample.extrinsics,
                image_feats=feats_param,
                mode="training",
                check_ref=False,
                flags=est_sample.flags,
            )

            while True:
                with self.accelerator.autocast():
                    output = next(generator, None)
                if output is None:
                    break

                assert not torch.is_autocast_enabled(), "AMP should not be enabled during loss computation"
                if isinstance(output, TrainData):
                    loss, _ = criterion.compute_loss(sample=sample, train_data=output, bidirectional=model.bidirectional, is_train=True)
                    self.accelerator.backward(loss, retain_graph=True)
                    train_data_list.append(output.detach(clone=True)) # type: ignore
                    del output, loss, _
                elif isinstance(output, Prediction):
                    assert pred is None, "Internal error"
                    pred = output
                else:
                    raise ValueError(f"Unknown output type: {type(output)}")
            
            feats_grad = feats_param.grad
            assert not feats_grad.requires_grad, "feats_grad should not require grad"
            del feats_param
            self.accelerator.backward((feats_grad * feats).sum())
            del feats_grad

            assert pred is not None, "Internal error"
        else:
            pred, train_data_list = model(
                rgb_obs=est_sample.rgbs,
                depth_obs=est_sample.depths,
                num_iters=self.cfg.train.train_iters,
                query_point=est_sample.query_point,
                intrinsics=est_sample.intrinsics,
                extrinsics=est_sample.extrinsics,
                flags=est_sample.flags,
                mode="training",
            )

        # Loss for backward pass
        loss_list, info_list = [], []
        for train_data in train_data_list:
            loss, info = criterion.compute_loss(sample=sample, train_data=train_data, bidirectional=self.accelerator.unwrap_model(model).bidirectional, is_train=True)
            loss_list.append(loss)
            info_list.append(info)

        loss = torch.sum(torch.stack(loss_list), dim=0)
        info = {k: torch.sum(torch.stack([info[k] for info in info_list]), dim=0) for k in info_list[0]}

        self.output_stream.write(f"step: {step_id}, seq_name: {sample.seq_name}, loss: {loss.item():.4f}\n")

        if self.cfg.train.streaming_backward:
            assert not loss.requires_grad, "Internal error"
        else:
            assert loss.requires_grad, "Internal error"
        
        if (step_id + 1) % 10 == 0:
            log_str = f"[Step {step_id+1}] loss: {loss.item():.4f}"
            for k, v in info.items():
                log_str += f" | {k}: {v.item():.4f}"
            if get_rank() == 0:
                logger.info(log_str)
            self.output_stream.flush()

        if self._should_log() and (step_id + 1) % 10 == 0:
            wandb.log(
                {
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "train-loss/total_loss": loss,
                    "step": step_id
                },
            )

        return loss

    @torch.no_grad()
    def evaluate_dataset(self, model, criterion, loader, step_id, dataset_name):
        """Run evaluation on a dataset."""
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process and self.cfg.train.get("save_preds", False):
            (self.output_dir / f"pred_{step_id}").mkdir(parents=True, exist_ok=True)
        
        self.accelerator.wait_for_everyone()

        eval_grid_size = self.cfg.train.get("eval_grid_size", 0)
        
        values = {"seq_id": []}
        device = next(model.parameters()).device
        model.eval()

        bidirectional_eval: bool = self.cfg.train.bidirectional_eval
        is_model_bidirectional: bool = self.accelerator.unwrap_model(model).bidirectional
        visited_indices = set()
        rank = get_rank()

        for i, sample in enumerate(tqdm(loader, position=1, desc=f"Evaluating {dataset_name}")):
            start_time = time.time()
            sample = sample.to(device)
            sample_id = sample.sample_id.item()

            # Handle redundant samples from DistributedSampler padding
            if torch.distributed.is_initialized():
                sample_ids = [None for _ in range(self.accelerator.num_processes)]
                torch.distributed.all_gather_object(sample_ids, (rank, sample_id))
            else:
                sample_ids = [(rank, sample_id)]

            should_skip = False
            for r, idx in sample_ids:
                if idx in visited_indices and r == rank:
                    should_skip = True
                visited_indices.add(idx)

            if should_skip:
                logger.debug(f"Skipping sample {sample_id} because it has already been evaluated (likely a padded sample)")
                continue
            
            if (i + 1) % 10 == 0:
                if rank == 0:
                    logger.info(f"Evaluating sample {sample_id} ({i+1}/{len(loader)}) on rank {rank}")

            est_sample = sample.with_annot_mode('est')
            
            preds, train_data = _inference_with_grid(
                grid_size=eval_grid_size,
                model=model,
                video=est_sample.rgbs,
                depths=est_sample.depths,
                num_iters=self.cfg.train.eval_iters,
                query_point=est_sample.query_point,
                intrinsics=est_sample.intrinsics,
                extrinsics=est_sample.extrinsics,
                flags=est_sample.flags,
                depth_roi=est_sample.depth_roi,
            )

            # Use forward prediction to compute loss
            loss_list, info_list = [], []
            for train_data_item in train_data:
                loss, info = criterion.compute_loss(sample=sample, train_data=train_data_item, bidirectional=is_model_bidirectional, is_train=False)
                loss_list.append(loss)
                info_list.append(info)
            
            if len(loss_list) > 0:
                loss = torch.sum(torch.stack(loss_list), dim=0)
                info = {k: torch.sum(torch.stack([info[k] for info in info_list]), dim=0) for k in info_list[0]}
            else:
                loss = torch.tensor(0., device=device)
                info = {}
            assert sample.seq_id.shape[0] == 1, "batch size must be 1 for evaluation"
            values["seq_id"].append(sample.seq_id.squeeze(0))
            
            for n, l in info.items():
                key = f"{dataset_name}-losses/mean/{n}"
                if key not in values:
                    values[key] = []
                values[key].append(l)

            # Before computing the metrics, we might need to perform backward tracking
            if bidirectional_eval and not is_model_bidirectional:
                B, T = sample.rgbs.shape[:2]
                N = est_sample.query_point.shape[1]
                preds_backward, _ = _inference_with_grid(
                    grid_size=eval_grid_size,
                    model=model,
                    video=est_sample.rgbs.flip(dims=(1,)),
                    depths=est_sample.depths.flip(dims=(1,)),
                    intrinsics=est_sample.intrinsics.flip(dims=(1,)),
                    extrinsics=est_sample.extrinsics.flip(dims=(1,)),
                    query_point=torch.cat([est_sample.rgbs.shape[1] - 1 - est_sample.query_point[..., :1], est_sample.query_point[..., 1:]], dim=-1),
                    num_iters=self.cfg.train.eval_iters,
                    flags=est_sample.flags,
                    depth_roi=est_sample.depth_roi,
                )
                preds.coords = torch.where(
                    repeat(torch.arange(T, device=sample.rgbs.device), 't -> b t n 3', b=B, n=N) < repeat(est_sample.query_point[..., 0], 'b n -> b t n 3', t=T, n=N),
                    preds_backward.coords.flip(dims=(1,)),
                    preds.coords
                )
                preds.visibs = torch.where(
                    repeat(torch.arange(T, device=sample.rgbs.device), 't -> b t n', b=B, n=N) < repeat(est_sample.query_point[..., 0], 'b n -> b t n', t=T, n=N),
                    preds_backward.visibs.flip(dims=(1,)),
                    preds.visibs
                )

            metrics_dict = metrics.compute_metrics(sample=sample, preds=preds, bidirectional=is_model_bidirectional or bidirectional_eval)

            if self.cfg.train.get("save_preds", False):
                metrics_dict_np = {k: v.cpu().numpy()[0] for k, v in metrics_dict.items()}
                preds_np = {"coords": preds.coords.cpu().numpy()[0], "visibs": preds.visibs.cpu().numpy()[0]}

                with open(self.output_dir / f"pred_{step_id}" / f"{sample_id}.pkl", "wb") as f:
                    pickle.dump(
                        {
                            "preds": preds_np,
                            "metrics": metrics_dict_np,
                        },
                        f
                    )

            for n, m in metrics_dict.items():
                key = f"{dataset_name}-metrics/mean/{n}"
                if key not in values:
                    values[key] = []
                values[key].append(m)
            
            if i < self.cfg.train.visualize_nbatches:
                if self._should_log():
                    # In this case we should log visualization to wandb
                    if step_id > -1:
                        samples_cpu = sample.to('cpu')
                        preds_cpu = preds.to('cpu')

                        if not is_model_bidirectional and not bidirectional_eval:
                            samples_cpu = samples_cpu.with_causal_mask()

                        generate_videos(
                            sample=samples_cpu.with_annot_mode('est'),
                            preds=preds_cpu,
                            output_dir=self.output_dir,
                            step_id=step_id,
                            batch_idx=i,
                            dataset_name=dataset_name,
                        )

                        if self.cfg.train.get("visualize_with_rerun", False):
                            # Log with rerun-sdk without interrupting training
                            try:
                                rerun_process = Process(target=generate_visualizations_with_rerun, kwargs={
                                    'sample': samples_cpu.with_annot_mode('est'),
                                    'preds': preds_cpu,
                                    'output_dir': self.output_dir,
                                    'step_id': step_id,
                                    'dataset_name': dataset_name,
                                    'batch_idx': i,
                                })
                                rerun_process.start()
                                rerun_process.join()
                            except Exception as e:
                                logger.error(f"Failed to log with rerun-sdk: {e}", exc_info=True)
            end_time = time.time()
            # logger.info(f"Time taken for batch {i}: {end_time - start_time:.2f} seconds")
        
        # Synchronize between processes
        self.accelerator.wait_for_everyone()

        values = {k: torch.stack(v).cpu() for k, v in values.items()}
        if torch.distributed.is_initialized():
            values_list = [None for _ in range(self.accelerator.num_processes)]
            torch.distributed.all_gather_object(values_list, values)
        else:
            values_list = [values]

        values_all = {k: torch.cat([values_list[i][k] for i in range(len(values_list))]).numpy() for k in values.keys()}
        values = {k: v.mean() for k, v in values_all.items()}

        vis_related_keys = set([re.fullmatch(r'(.*)_visthr_.*', k).groups()[0] for k in values.keys() if re.fullmatch(r'(.*)_visthr_.*', k)])
        for k in vis_related_keys:
            values[k + "_best"] = np.max([v for k_, v in values.items() if k_.startswith(f'{k}_visthr')])

        if self._should_log():
            # Log to wandb
            to_log = {
                "step": step_id,
            }

            core_keys_mapping = {
                "tapvid3d_average_pts_within_thresh": "3D APD",
                "tapvid2d_average_pts_within_thresh": "2D APD",
            }
            core_keys = list (core_keys_mapping.keys())
            core_keys.sort()
            columns = ["seq_id"] + [core_keys_mapping[k] for k in core_keys]
            table_content = []
            values_all_suffix = {k.split("/")[-1]: k for k in values_all.keys()}
            for i in range(len(values_all['seq_id'])):
                table_content.append([int(values_all['seq_id'][i])] + ["%.4f" % float(values_all[values_all_suffix[k]][i]) for k in core_keys])
            table_content.sort(key=lambda x: x[0])

            to_log.update({f"{dataset_name}-eval-details/table": wandb.Table(columns=columns, data=table_content)})
            for k in core_keys:
                to_log.update({f"{dataset_name}-eval-details/{core_keys_mapping[k]}-hist": wandb.Histogram(values_all[values_all_suffix[k]])})
            to_log.update(values)
            wandb.log(to_log)

            # Log to terminal
            logger.info(f"Evaluation Results for {dataset_name} at Step {step_id}:")
            max_key_len = max(len(k) for k in values.keys())
            for key, value in sorted(values.items()):
                logger.info(f"  {key:<{max_key_len}}: {value:.4f}")

        return - values[f"{dataset_name}-metrics/mean/tapvid3d_average_pts_within_thresh"]

def generate_videos(sample, preds, output_dir, step_id, batch_idx, dataset_name):
    for i in range(len(sample.rgbs)):
        save_dir = output_dir / f"visualization_{dataset_name}"
        filename = f"video_{step_id}_{batch_idx}_{i}"
        save_dir.mkdir(parents=True, exist_ok=True)
        vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=3)
        tracks_3d = preds.coords[i:i+1]
        tracks_2d = batch_project(
            tracks_3d, 
            repeat(sample.intrinsics[i:i+1], 'b t i j -> b t n i j', n=tracks_3d.shape[2]),
            repeat(sample.extrinsics[i:i+1], 'b t i j -> b t n i j', n=tracks_3d.shape[2])
        )
        vis.visualize(
            video=(sample.rgbs[i:i+1] * 255).to(torch.uint8),
            tracks=tracks_2d,
            visibility=(preds.visibs[i:i+1] > 0.),
            valids=torch.ones_like(sample.valids[i:i+1]),
            filename=filename
        )
        viz_key = f'{dataset_name}-viz/viz-{batch_idx}-{i}'
        wandb.log(
            {
                "step": step_id,
                viz_key: wandb.Video(str(f"{save_dir}/{filename}.mp4")),
            }
        )

def generate_visualizations_with_rerun(sample, preds, output_dir, step_id, batch_idx, dataset_name):
    depths = sample.depths
    for i in range(len(sample.rgbs)):
        visualizer.setup_visualizer(app_name="Visualization", serve=False)
        visualizer.log_video(
            entity_name=f"video",
            rgb=sample.rgbs[i],
            intrinsics=sample.intrinsics[i],
            extrinsics=sample.extrinsics[i],
            depth=depths[i],
        )
        visualizer.log_trajectory(
            entity_name=f"video",
            track_name="pred",
            intrinsics=sample.intrinsics[i],
            extrinsics=sample.extrinsics[i],
            trajs=preds.coords[i],
            visibs=preds.visibs[i] > 0,
            valids=torch.ones_like(sample.valids[i]), # no need to mask invalid points for prediction
            queries=sample.query_point[i],
            cmap_name="plasma"
        )
        visualizer.save_recording(output_dir / f"visualization_{dataset_name}" / f"{step_id}_{batch_idx}_{i}.rrd")

@hydra.main(config_path="configs", config_name="test", version_base="1.3")
def main(cfg: DictConfig):
    setup_logger()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if get_rank() == 0:
        logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train_tester = TrainTester(cfg)
    # torch.cuda.set_sync_debug_mode(1)
    if cfg.get("profile", False):
        if cfg.train.num_workers > 0:
            logger.warning("Profile is not supported with num_workers > 0, so we manually set it to 0. This will slow down the training.")
            cfg.train.num_workers = 0
            cfg.train.prefetch_factor = None
        with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA, profiler.ProfilerActivity.XPU], with_stack=True) as prof:
            train_tester.main(collate_fn=SliceData.collate)
        print(prof.key_averages(group_by_stack_n=1).table(sort_by="self_cpu_time_total", row_limit=20))
        print(prof.key_averages(group_by_stack_n=1).table(sort_by="self_cuda_time_total", row_limit=20))
    else:
        train_tester.main(collate_fn=SliceData.collate)

if __name__ == "__main__":
    main()
