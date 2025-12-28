# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

"""Shared utilities for all main scripts."""

from collections import OrderedDict
from copy import deepcopy
import datetime
import os
import socket
from pathlib import Path
from typing import Dict, Optional, Tuple
import accelerate
from accelerate import DistributedDataParallelKwargs

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate, Dataset
import torch.distributed as dist
from tqdm import trange, tqdm
import wandb
import logging
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from accelerate.utils import set_seed

from torch.nn.parallel import DistributedDataParallel
from transformers.optimization import get_constant_schedule_with_warmup

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
    
class BaseTrainTester:
    """Basic train/test class to be inherited."""

    def _should_log(self) -> bool:
        return self.cfg.wandb.enable and get_rank() == 0

    def __init__(self, cfg: DictConfig):
        """Initialize."""
        self.cfg = cfg
        self.gradient_accumulation_steps = self.cfg.train.gradient_accumulation_steps

    @staticmethod
    def get_datasets() -> Tuple[Dataset, Dict[str, Dataset]]:
        """Initialize datasets."""
        raise NotImplementedError()

    def get_loaders(self, collate_fn=default_collate, train_only=False) -> Tuple[DataLoader, Dict[str, DataLoader]]:
        """Initialize data loaders."""
        if self.train_dataset is not None:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.cfg.train.batch_size,
                shuffle=True,
                num_workers=self.cfg.train.num_workers,
                collate_fn=collate_fn,
                prefetch_factor=self.cfg.train.get("prefetch_factor", 2),
                pin_memory=True,
                drop_last=True,
            )
        else:
            train_loader = None
        test_loaders = OrderedDict({
            key: DataLoader(
                self.test_datasets[key],
                batch_size=1,
                shuffle=False,
                num_workers=self.cfg.train.num_workers,
                prefetch_factor=self.cfg.train.get("prefetch_factor", 2),
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=False,
            )
            for key in self.test_datasets
        }) if not train_only else {}
        return train_loader, test_loaders

    @staticmethod
    def get_model():
        """Initialize the model."""
        return None

    def get_criterion(self):
        """Get loss criterion for training."""
        # criterion is a class, must have compute_loss and compute_metrics
        return None

    def get_optimizer(self, model):
        """Initialize optimizer."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.cfg.train.lr},
            {"params": [], "weight_decay": self.cfg.train.weight_decay, "lr": self.cfg.train.lr}
        ]
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", "GroupNorm.weight", "GroupNorm.bias", "InstanceNorm.weight", "InstanceNorm.bias",
                    "InstanceNorm1d.weight", "InstanceNorm1d.bias", "InstanceNorm2d.weight", "InstanceNorm2d.bias"]
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                optimizer_grouped_parameters[0]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        self.trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

        optimizer_name = self.cfg.train.get('optimizer_name', 'adamw').lower()

        if optimizer_name == "adamw":
            optimizer = optim.AdamW(optimizer_grouped_parameters)
            assert not self.cfg.train.get("optimizer_kwargs", {}), "We do not support optimizer_kwargs for AdamW"
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        return optimizer

    def get_scheduler(self, optimizer):
        """Initialize scheduler."""
        if self.cfg.train.scheduler_name == "constant":
            for p in range(len(optimizer.param_groups)):
                optimizer.param_groups[p]['lr'] = self.cfg.train.lr
            return None
        elif self.cfg.train.scheduler_name == "onecycle":
            return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.train.lr, total_steps=self.cfg.train.train_steps + 100, **self.cfg.train.scheduler_kwargs)
        elif self.cfg.train.scheduler_name == "constant_with_warmup":
            return get_constant_schedule_with_warmup(optimizer, **self.cfg.train.scheduler_kwargs)
        else:
            raise ValueError(f"Unsupported scheduler: {self.cfg.train.scheduler_name}")

    def broadcast_model(self, model: torch.nn.Module):
        assert not isinstance(model, DistributedDataParallel), "The model should not be manually broadcasted when using DDP"
        params = list (model.named_parameters())
        buffers = list (model.named_buffers())

        params.sort(key=lambda x: x[0])
        buffers.sort(key=lambda x: x[0])

        for name, param in params:
            accelerate.utils.broadcast(param.data)

        for name, buffer in buffers:
            accelerate.utils.broadcast(buffer.data) 

    def sync_gradients(self, model: torch.nn.Module):
        """
        Manually synchronize gradients. Required in 'streaming_backward' mode 
        because multiple backward passes with 'retain_graph=True' can interfere 
        with DDP's default synchronization behavior.
        """
        assert not isinstance(model, DistributedDataParallel), "Gradients should not be manually synced when using DDP"
        world_size = self.accelerator.num_processes

        grads = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grads.append((name, param.grad))
                else:
                    param.grad = torch.zeros_like(param.data)
                    grads.append((name, param.grad))
        grads.sort(key=lambda x: x[0])

        grad_tensors = [grad.view(-1) for name, grad in grads]
        flat_grads = torch.cat(grad_tensors)

        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads /= world_size

        pointer = 0
        for name, grad in grads:
            numel = grad.numel()
            grad.copy_(flat_grads[pointer:pointer + numel].view_as(grad))
            pointer += numel

    def check_nan_gradients(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and torch.isnan(param.grad).any():
                return False
        return True

    def main(self, collate_fn=default_collate):
        """Run main training/testing pipeline."""
        set_seed(self.cfg.seed)

        self.train_dataset, self.test_datasets = self.get_datasets()
        train_loader, test_loaders = self.get_loaders(collate_fn)
        model = self.get_model()

        auto_init_dist_single_process()

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False, static_graph=False)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_steps,
            mixed_precision=self.cfg.train.mixed_precision,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs]
        )

        logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")

        criterion = self.get_criterion()
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer)

        start_iter, best_ref_loss = 0, None

        experiment_id = self.cfg.get("experiment_id", None)
        if experiment_id is None:
            experiment_id = f"{self.cfg.experiment_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            self.output_dir = OUTPUT_DIR / "auto_generated" / experiment_id
        else:
            self.output_dir = OUTPUT_DIR / "id_specified" / experiment_id

        if get_rank() == 0:
            if (self.output_dir / "last.pth").is_file():
                logger.info(f"Automatically resumed from {self.output_dir / 'last.pth'}")
                self.cfg.train.checkpoint = str (self.output_dir / "last.pth")
            else:
                logger.info(f"Starting new experiment {experiment_id}")
                self.output_dir.mkdir(parents=True, exist_ok=True)
        
        _output_dir = [self.output_dir]
        _output_dir = accelerate.utils.broadcast_object_list(_output_dir, from_process=0)
        self.output_dir = _output_dir[0]

        self.accelerator.wait_for_everyone()

        if self._should_log():
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.get("entity"),
                name=self.cfg.experiment_name,
                resume="allow",
                id=experiment_id,
                dir=str(self.output_dir),
                config=OmegaConf.to_container(self.cfg), # type: ignore
            )
            if self.cfg.wandb.log_code:
                logger.warning(f"All code in {os.path.dirname(os.path.dirname(__file__))} will be logged to wandb. If this is not intended, set wandb.log_code to False.")
                wandb.run.log_code( # type: ignore
                    str(os.path.dirname(os.path.dirname(__file__))),
                    include_fn=lambda x: x.endswith((".py", ".sh", ".yaml", ".json")),
                    exclude_fn=lambda path, root: os.path.relpath(path, root).startswith(("outputs", ".git"))
                )
            wandb.define_metric("*", step_metric="step")

        self.output_stream = open(os.path.join(self.output_dir, f"output_localrank{os.environ.get('LOCAL_RANK', 0)}.txt"), "a")

        checkpoint_list = [self.cfg.train.checkpoint]
        checkpoint_list = accelerate.utils.broadcast_object_list(checkpoint_list, from_process=0)
        self.cfg.train.checkpoint = checkpoint_list[0]
        if self.cfg.train.checkpoint:
            start_iter, best_ref_loss = self.load_checkpoint(model, optimizer, scheduler)
        elif self.cfg.train.get('pretrained', None):
            logger.info(f"=> loading pretrained weights from '{self.cfg.train.pretrained}'")
            model_dict = torch.load(self.cfg.train.pretrained, map_location="cpu")
            model.load_state_dict(model_dict["weight"], strict=True)
            del model_dict

        # Wrap objects with accelerator
        if self.cfg.train.streaming_backward:
            model = self.accelerator.prepare_model(model, evaluation_mode=True) # type: ignore
            self.broadcast_model(model)
            for module in model.modules(): # type: ignore
                if isinstance(module, torch.nn.SyncBatchNorm):
                    raise ValueError("SyncBatchNorm is not supported for streaming backward mode")
        else:
            model = self.accelerator.prepare_model(model, evaluation_mode=False) # type: ignore

        optimizer, train_loader, scheduler = self.accelerator.prepare(
            optimizer, train_loader, scheduler, device_placement=[True, True, True]
        )
        self.optimizer = optimizer
        test_loaders = {
            key: self.accelerator.prepare(test_loaders[key])
            for key in test_loaders
        }
        
        if not self.cfg.train.get('eval_only', False):
            trained_epoch = start_iter * self.gradient_accumulation_steps // len(train_loader)
            trained_batch_in_epoch = start_iter * self.gradient_accumulation_steps % len(train_loader)
            train_loader.set_epoch(trained_epoch)
            skipped_train_loader = self.accelerator.skip_first_batches(train_loader, num_batches=trained_batch_in_epoch)
            train_iter = iter(skipped_train_loader)
        else: 
            train_iter = None

        # Training loop
        model.train()
        
        if hasattr(self.optimizer.optimizer, "train"):
            self.optimizer.optimizer.train()

        if get_rank() == 0:
            pbar = tqdm(total=self.cfg.train.train_steps)
            pbar.update(start_iter)
            
        for step_id in range(start_iter, self.cfg.train.train_steps):
            for _ in range(self.gradient_accumulation_steps):
                if not self.cfg.train.get('eval_only', False):
                    try:
                        sample = next(train_iter)
                    except StopIteration:
                        assert step_id % len(train_loader) == 0
                        train_loader.set_epoch(step_id // len(train_loader))
                        train_iter = iter(train_loader)
                        sample = next(train_iter)

                    with self.accelerator.accumulate(model):
                        loss = self.train_one_step(model, criterion, step_id, sample)
                        assert not torch.is_autocast_enabled(), "AMP should only be enabled in the train_one_step"
                        if not self.cfg.train.streaming_backward:
                            self.accelerator.backward(loss)
                        else:
                            assert not loss.requires_grad # type: ignore

                        loss = loss.detach().clone()

                        if self.accelerator.sync_gradients:
                            if self.cfg.train.streaming_backward:
                                self.sync_gradients(model) # type: ignore

                            max_grad_norm = self.cfg.train.max_grad_norm if self.cfg.train.max_grad_norm is not None else float("inf")
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.trainable_parameters,
                                max_grad_norm,
                            ).item() # type: ignore
                            # logger.info(f"grad_norm: {grad_norm}")
                                
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()

            if get_rank() == 0:
                pbar.update(1)

            if (step_id + 1) % self.cfg.train.eval_freq == 0 or self.cfg.train.get("eval_only", False):
                model.eval()
                if hasattr(self.optimizer.optimizer, "eval"):
                    self.optimizer.optimizer.eval()
                
                ref_losses = []
                for key in sorted(test_loaders.keys()):
                    logger.info(f"Evaluating on {key}.......")
                    ref_loss = self.evaluate_dataset(
                        model, criterion, test_loaders[key], step_id,
                        dataset_name=key
                    )
                    ref_losses.append(ref_loss)

                mean_ref_loss = np.mean(ref_losses)

                self.accelerator.wait_for_everyone()
                if get_rank() == 0 and not self.cfg.train.get("eval_only", False):
                    best_ref_loss = self.save_checkpoint(
                        model, optimizer, scheduler, step_id,
                        mean_ref_loss, best_ref_loss
                    )

                if hasattr(self.optimizer.optimizer, "train"):
                    self.optimizer.optimizer.train()
                model.train()

                if self.cfg.train.get("eval_only", False):
                    break

            if (step_id + 1) % self.cfg.train.save_freq == 0:

                self.accelerator.wait_for_everyone()
                if get_rank() == 0:
                    self.save_checkpoint(
                        model, optimizer, scheduler, step_id,
                        None, None
                    )

            if (step_id + 1) % self.cfg.train.get("keep_freq", self.cfg.train.train_steps // 10) == 0:
                self.accelerator.wait_for_everyone()
                if get_rank() == 0:
                    self.save_checkpoint(
                        model, optimizer, scheduler, step_id,
                        None, None, name=f"checkpoint_{step_id}"
                    )

        return model

    def train_one_step(self, model, criterion, step_id, sample):
        """Run a single training step."""
        pass

    @torch.no_grad()
    def evaluate_dataset(self, model, criterion, loader, step_id, dataset_name):
        """Run evaluation on a dataset."""
        return None

    def load_checkpoint(self, model, optimizer, scheduler):
        """Load from checkpoint."""
        if get_rank() == 0:
            if Path(self.cfg.train.checkpoint).with_suffix('.pth_old').is_file():
                if Path(self.cfg.train.checkpoint).is_file():
                    os.unlink(self.cfg.train.checkpoint)
                os.rename(Path(self.cfg.train.checkpoint).with_suffix('.pth_old'), self.cfg.train.checkpoint)

        self.accelerator.wait_for_everyone()

        assert Path(self.cfg.train.checkpoint).is_file(), f"Checkpoint not found: {self.cfg.train.checkpoint}"
        
        logger.info(f"=> loading checkpoint from '{self.cfg.train.checkpoint}'")

        model_dict = torch.load(self.cfg.train.checkpoint, map_location="cpu")
        model.load_state_dict(model_dict["weight"])
        if 'optimizer' in model_dict:
            optimizer.load_state_dict(model_dict["optimizer"])

            if model_dict.get("scheduler", None) is None:
                for p in range(len(optimizer.param_groups)):
                    optimizer.param_groups[p]['lr'] = self.cfg.train.lr
            else:
                logger.warning("Scheduler state dict found in checkpoint. We won't use the learning rate specified in config.")

        if 'scheduler' in model_dict and model_dict['scheduler'] is not None and scheduler is not None:
            scheduler.load_state_dict(model_dict["scheduler"])

        start_iter = model_dict.get("iter", 0)
        best_ref_loss = model_dict.get("best_ref_loss", None)

        logger.info(f"=> loaded successfully (step {start_iter})")
        del model_dict
        torch.cuda.empty_cache()
        return start_iter, best_ref_loss

    def _save_with_backup(self, data, path):
        path = Path(path)
        backup_path = path.with_suffix('.pth_old')
        if path.is_file():
            os.rename(path, backup_path)
        torch.save(data, path)
        if backup_path.is_file():
            os.unlink(backup_path)
            
    def save_checkpoint(self, model, optimizer, scheduler, step_id, new_ref_loss, best_ref_loss, name=None):
        """Save checkpoint if requested."""
        assert get_rank() == 0, "Save checkpoint should only be called on rank 0!"

        is_training = model.training

        if is_training:
            model.eval()
            if hasattr(optimizer.optimizer, "eval"):
                optimizer.optimizer.eval()
        if hasattr(optimizer.optimizer, "eval"):
            assert not optimizer.optimizer.param_groups[0]['train_mode'], "Optimizer is not in eval mode"

        unwrapped_model = self.accelerator.unwrap_model(model)
        if name is None:
            if new_ref_loss is not None and (best_ref_loss is None or new_ref_loss <= best_ref_loss):
                best_ref_loss = new_ref_loss
                save_path = self.output_dir / "best.pth"
                logger.info(f"=> saving best checkpoint to '{save_path}'")
                self._save_with_backup({
                    "cfg": self.cfg,
                    "weight": unwrapped_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "iter": step_id + 1,
                    "best_ref_loss": best_ref_loss
                }, save_path)
            
            save_path = self.output_dir / "last.pth"
            logger.info(f"=> saving last checkpoint to '{save_path}'")
            self._save_with_backup({
                "cfg": self.cfg,
                "weight": unwrapped_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "iter": step_id + 1,
                "best_ref_loss": best_ref_loss
            }, save_path)
        else:
            assert new_ref_loss is None and best_ref_loss is None
            save_path = self.output_dir / f"{name}.pth"
            logger.info(f"=> saving checkpoint to '{save_path}'")
            self._save_with_backup({
                "cfg": self.cfg,
                "weight": unwrapped_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "iter": step_id + 1,
                "best_ref_loss": best_ref_loss
            }, save_path)

        if is_training:
            model.train()
            if hasattr(optimizer.optimizer, "train"):
                optimizer.optimizer.train()
        return best_ref_loss


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))  # 0 => let OS choose a free port
    port = s.getsockname()[1]
    s.close()
    return port

def auto_init_dist_single_process():
    # If accelerate/torchrun already initialized (multi-proc), do nothing.
    if dist.is_available() and dist.is_initialized():
        return

    if "WORLD_SIZE" in os.environ and os.environ.get("WORLD_SIZE", "1") != "1":
        return

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))  # avoid conflicts

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    dist.init_process_group(backend=backend, init_method="env://", rank=0, world_size=1)