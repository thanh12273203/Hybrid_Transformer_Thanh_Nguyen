import os
import re
import csv
from typing import List, Tuple, Dict, Callable, Optional, Union

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from ..configs import TrainConfig
from ..loss import LOSS_REGISTRY
from ..optim import OPTIM_REGISTRY, SCHEDULER_REGISTRY
from ..utils import (
    BaseCallback,
    get_loss_from_config,
    get_optim_from_config,
    get_scheduler_from_config
)


class Trainer:
    """
    Base class for training models.

    Parameters
    ----------
    model: nn.Module
        The model to train.
    train_dataset: torch.utils.data.Dataset
        The dataset to use for training.
    val_dataset: torch.utils.data.Dataset
        The dataset to use for validation.
    test_dataset: torch.utils.data.Dataset, optional
        The dataset to use for testing.
    metric: Callable, optional
        A function to compute a metric for evaluation.
    callbacks: List[BaseCallback], optional
        A list of callbacks to execute during training.
    config: TrainConfig, optional
        Configuration object containing training parameters.
    batch_size: int, optional
        Batch size for training. Overrides config if provided.
    criterion: Dict, optional
        Loss function configuration. Overrides config if provided.
    optimizer: Dict, optional
        Optimizer configuration. Overrides config if provided.
    scheduler: Dict, optional
        Learning rate scheduler configuration. Overrides config if provided.
    num_epochs: int, optional
        Number of epochs to train for. Overrides config if provided.
    start_epoch: int, optional
        Epoch to start training from. Overrides config if provided.
    history: Dict[str, List[float]], optional
        History of training metrics. If not provided, initializes an empty history.
    logging_dir: str, optional
        Directory to save logs. Overrides config if provided.
    logging_steps: int, optional
        Frequency of logging during training. Overrides config if provided.
    progress_bar: bool, optional
        Whether to display a tqdm progress bar. Useful to disable on HPC.
    save_best: bool, optional
        Whether to save the best model based on validation loss. Overrides config if provided.
    save_ckpt: bool, optional
        Whether to save checkpoints during training. Overrides config if provided.
    device: torch.device, optional
        Device to run the training on. Overrides config if provided.
    num_workers: int, optional
        Number of workers for data loading. Overrides config if provided.
    pin_memory: bool, optional
        Whether to use pinned memory for data loading. Overrides config if provided.
    """
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        metric: Optional[Callable] = None,
        callbacks: Optional[List[BaseCallback]] = None,
        config: Optional[TrainConfig] = None,
        # Parameters below can override config if supplied explicitly
        batch_size: Optional[int] = None,
        criterion: Optional[Dict] = None,
        optimizer: Optional[Dict] = None,
        scheduler: Optional[Dict] = None,
        num_epochs: Optional[int] = None,
        start_epoch: Optional[int] = None,
        history: Optional[Dict[str, List[float]]] = None,
        logging_dir: Optional[str] = None,
        logging_steps: Optional[int] = None,
        progress_bar: Optional[bool] = None,
        save_best: Optional[bool] = None,
        save_ckpt: Optional[bool] = None,
        device: Optional[torch.device] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None
    ):
        self.model = model
        self.metric = metric
        self.scheduler = scheduler
        self.callbacks = callbacks or []

        # Use config if provided, otherwise use defaults
        if config is not None:
            self.batch_size = batch_size if batch_size is not None else config.batch_size
            self.criterion = get_loss_from_config(criterion if criterion is not None else config.criterion, LOSS_REGISTRY)
            self.optimizer = get_optim_from_config(optimizer if optimizer is not None else config.optimizer, OPTIM_REGISTRY, self.model)
            self.scheduler = get_scheduler_from_config(scheduler if scheduler is not None else config.scheduler, SCHEDULER_REGISTRY, self.optimizer)
            self.num_epochs = num_epochs if num_epochs is not None else config.num_epochs
            self.start_epoch = start_epoch if start_epoch is not None else config.start_epoch
            self.logging_dir = logging_dir if logging_dir is not None else config.logging_dir
            self.progress_bar = progress_bar if progress_bar is not None else config.progress_bar
            self.logging_steps = logging_steps if logging_steps is not None else config.logging_steps
            self.save_best = save_best if save_best is not None else config.save_best
            self.save_ckpt = save_ckpt if save_ckpt is not None else config.save_ckpt
            self.device = device if device is not None else torch.device(config.device)
            self.num_workers = num_workers if num_workers is not None else config.num_workers
            self.pin_memory = pin_memory if pin_memory is not None else config.pin_memory
        else:
            self.batch_size = batch_size if batch_size is not None else 64
            if criterion is None: raise ValueError("Criterion must be provided if config is not supplied.")
            self.criterion = get_loss_from_config(criterion, LOSS_REGISTRY)
            if optimizer is None: raise ValueError("Optimizer must be provided if config is not supplied.")
            self.optimizer = get_optim_from_config(optimizer, OPTIM_REGISTRY, self.model)
            self.scheduler = get_scheduler_from_config(scheduler, OPTIM_REGISTRY, self.optimizer) if scheduler else None
            self.num_epochs = num_epochs if num_epochs is not None else 20
            self.start_epoch = start_epoch if start_epoch is not None else 0
            self.logging_dir = logging_dir if logging_dir is not None else 'logs'
            self.progress_bar = progress_bar if progress_bar is not None else True
            self.logging_steps = logging_steps if logging_steps is not None else 500
            self.save_best = save_best if save_best is not None else True
            self.save_ckpt = save_ckpt if save_ckpt is not None else True
            self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.num_workers = num_workers if num_workers is not None else 0
            self.pin_memory = pin_memory if pin_memory is not None else False

        # Initialize data loaders
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        ) if test_dataset is not None else None
        self.history = history or {
            'epoch': [],
            'train_loss': [],
            'train_metric': [],
            'val_loss': [],
            'val_metric': []
        }

        self.best_val_loss = min(self.history['val_loss']) if self.history['val_loss'] else float('inf')

        self.model_name = self.model.__class__.__name__
        os.makedirs(self.logging_dir, exist_ok=True)
        self.log_dir = os.path.join(self.logging_dir, self.model_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # Subfolders
        self.best_models_dir = os.path.join(self.log_dir, 'best')
        self.checkpoints_dir = os.path.join(self.log_dir, 'checkpoints')
        self.loggings_dir = os.path.join(self.log_dir, 'loggings')
        os.makedirs(self.best_models_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.loggings_dir, exist_ok=True)

        # Determine run index
        run_index = self._get_next_run_index(self.loggings_dir, 'run', '.csv')
        self.run_name = f"run_{run_index:02d}"

        # Logging and best model paths
        self._log_header_written = False
        self.best_model_path = os.path.join(self.best_models_dir, f"{self.run_name}.pt") if self.save_best else None
        self.checkpoint_path = os.path.join(self.checkpoints_dir, f"{self.run_name}.pt") if self.save_ckpt else None
        self.logging_path = os.path.join(self.loggings_dir, f"{self.run_name}.csv")

    def _get_next_run_index(self, directory: str, prefix: str, suffix: str) -> int:
        os.makedirs(directory, exist_ok=True)
        existing = [
            f for f in os.listdir(directory)
            if f.startswith(prefix) and f.endswith(suffix)
        ]
        indices = [
            int(m.group(1)) for f in existing
            if (m := re.search(rf"{prefix}_(\d+)", f))
        ]
        return max(indices, default=0) + 1

    def save_checkpoint(self, epoch: int):
        if self.checkpoint_path:
            checkpoint = {
                'run_name': self.run_name,
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'history': self.history
            }
            torch.save(checkpoint, self.checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.run_name = checkpoint['run_name']
        self._log_header_written = True
        self.logging_path = os.path.join(self.loggings_dir, f"{self.run_name}.csv")
        self.best_model_path = os.path.join(self.best_models_dir, f"{self.run_name}.pt") if self.save_best else None
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch']
        self.history = checkpoint['history']

        return self.start_epoch
    
    def load_best_model(self, best_model_path: str):
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

    def log_csv(self, log_dict: Dict[str, float]):
        write_header = not self._log_header_written
        with open(self.logging_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=log_dict.keys())
            if write_header:
                writer.writeheader()
                self._log_header_written = True
            writer.writerow(log_dict)

    def train(self) -> Tuple[Dict[str, List[float]], nn.Module]:
        try:
            # Callback before training
            for cb in self.callbacks:
                cb.on_train_begin(trainer=self)

            total_steps = self.num_epochs * len(self.train_loader)
            start_step = self.start_epoch * len(self.train_loader)
            
            if self.progress_bar:
                global_bar = tqdm(
                    total=total_steps,
                    initial=start_step,
                    desc="Training",
                    dynamic_ncols=True
                )
            else:
                class _NoOpBar:
                    def set_postfix(self, *args, **kwargs):
                        pass
                    def update(self, *args, **kwargs):
                        pass
                global_bar = _NoOpBar()

            for epoch in range(self.start_epoch, self.num_epochs):
                # Callback at the beginning of each epoch
                for cb in self.callbacks:
                    cb.on_epoch_begin(epoch, trainer=self)

                # Training phase
                self.model.train()
                running_loss = 0.0
                running_metric = 0.0

                for batch_idx, (X, y) in enumerate(self.train_loader):
                    step = epoch * len(self.train_loader) + batch_idx + 1

                    X = X.to(self.device, non_blocking=self.pin_memory)
                    y = y.to(self.device, non_blocking=self.pin_memory)

                    self.optimizer.zero_grad()

                    outputs = self.model(X)

                    loss = self.criterion(outputs, y)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                    if self.metric:
                        running_metric += self.metric(outputs, y)

                    avg_loss = running_loss / (batch_idx + 1)
                    avg_metric = running_metric / (batch_idx + 1)

                    # Short summary
                    if step % self.logging_steps == 0 or step == total_steps:
                        tqdm.write(
                            f"step: {step}/{total_steps} | "
                            f"train_loss: {avg_loss:.4f} | "
                            f"train_metric: {avg_metric:.4f}"
                        )

                    global_bar.set_postfix({
                        "epoch": f"{epoch + 1}/{self.num_epochs}",
                        "avg_loss": f"{avg_loss:.4f}",
                        "avg_metric": f"{avg_metric:.4f}"
                    })
                    global_bar.update(1)

                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_metric = 0.0

                with torch.no_grad():
                    for X_val, y_val in self.val_loader:
                        X_val = X_val.to(self.device, non_blocking=self.pin_memory)
                        y_val = y_val.to(self.device, non_blocking=self.pin_memory)

                        outputs_val = self.model(X_val)

                        val_loss += self.criterion(outputs_val, y_val).item()

                        if self.metric:
                            val_metric += self.metric(outputs_val, y_val)

                val_loss /= len(self.val_loader)
                val_metric /= len(self.val_loader)

                # Short summary for validation
                tqdm.write(
                    f"epoch: {epoch + 1}/{self.num_epochs} | "
                    f"val_loss: {val_loss:.4f} | "
                    f"val_metric: {val_metric:.4f}"
                )

                if self.scheduler:
                    self.scheduler.step()

                # Get learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                # Save best model
                if self.best_model_path and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.best_model_path)

                # Update history
                self.history['epoch'].append(epoch + 1)
                self.history['train_loss'].append(avg_loss)
                self.history['train_metric'].append(avg_metric)
                self.history['val_loss'].append(val_loss)
                self.history['val_metric'].append(val_metric)

                # Save a checkpoint every epoch
                self.save_checkpoint(epoch)

                # Log results
                logs = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'train_metric': avg_metric,
                    'val_loss': val_loss,
                    'val_metric': val_metric,
                    'learning_rate': current_lr,
                }
                self.log_csv(logs)

                # Callback after each epoch
                for cb in self.callbacks:
                    cb.on_epoch_end(epoch, trainer=self, logs=logs)

                # Break if any callback says to stop
                if any(getattr(cb, 'should_stop', False) for cb in self.callbacks):
                    break

            # Callback after training
            for cb in self.callbacks:
                cb.on_train_end(trainer=self)
        except KeyboardInterrupt:
            print(f"\nTraining interrupted at epoch {epoch + 1}. Saving current checkpoint.")
            self.save_checkpoint(epoch)

        return self.history, self.model
    
    @torch.no_grad()
    def evaluate(
        self,
        loss_type: str,
        plot: Optional[Union[Callable, List[Callable]]] = None
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        if self.test_loader is None:
            raise ValueError("Test dataset is not provided.")
        
        self.model.eval()
        test_loss = 0.0
        test_metric = 0.0
        y_true_list = []
        y_pred_list = []

        for X_test, y_test in self.test_loader:
            X_test = X_test.to(self.device, non_blocking=self.pin_memory)
            y_test = y_test.to(self.device, non_blocking=self.pin_memory)

            outputs_test = self.model(X_test)

            test_loss += self.criterion(outputs_test, y_test).item()

            if self.metric:
                test_metric += self.metric(outputs_test, y_test)

            probs = outputs_test
            if loss_type == 'cross_entropy':
                probs = F.softmax(outputs_test, dim=1)
            elif loss_type == 'bce':
                probs = torch.sigmoid(outputs_test)

            y_pred_list.append(probs.cpu().numpy())
            y_true_list.append(y_test.cpu().numpy())

        test_loss /= len(self.test_loader)
        test_metric /= len(self.test_loader)

        print(f"test_loss: {test_loss:.4f} | test_metric: {test_metric:.4f}")

        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)

        # Visualization
        if plot is not None:
            if isinstance(plot, list):
                for viz in plot:
                    viz(y_true, y_pred)
            else:
                plot(y_true, y_pred)

        return test_loss, test_metric, y_true, y_pred
    

class MaskedModelTrainer(Trainer):
    """
    Trainer for model with one masked particle.

    Parameters
    ----------
    model: nn.Module
        The model to train.
    train_dataset: torch.utils.data.Dataset
        The dataset to use for training.
    val_dataset: torch.utils.data.Dataset
        The dataset to use for validation.
    test_dataset: torch.utils.data.Dataset, optional
        The dataset to use for testing.
    metric: Callable, optional
        A function to compute a metric for evaluation.
    callbacks: List[BaseCallback], optional
        A list of callbacks to execute during training.
    config: TrainConfig, optional
        Configuration object containing training parameters.
    batch_size: int, optional
        Batch size for training. Overrides config if provided.
    criterion: Dict, optional
        Loss function configuration. Overrides config if provided.
    optimizer: Dict, optional
        Optimizer configuration. Overrides config if provided.
    scheduler: Dict, optional
        Learning rate scheduler configuration. Overrides config if provided.
    num_epochs: int, optional
        Number of epochs to train for. Overrides config if provided.
    start_epoch: int, optional
        Epoch to start training from. Overrides config if provided.
    history: Dict[str, List[float]], optional
        History of training metrics. If not provided, initializes an empty history.
    logging_dir: str, optional
        Directory to save logs. Overrides config if provided.
    logging_steps: int, optional
        Frequency of logging during training. Overrides config if provided.
    progress_bar: bool, optional
        Whether to display a tqdm progress bar. Useful to disable on HPC.
    save_best: bool, optional
        Whether to save the best model based on validation loss. Overrides config if provided.
    save_ckpt: bool, optional
        Whether to save checkpoints during training. Overrides config if provided.
    device: torch.device, optional
        Device to run the training on. Overrides config if provided.
    num_workers: int, optional
        Number of workers for data loading. Overrides config if provided.
    pin_memory: bool, optional
        Whether to use pinned memory for data loading. Overrides config if provided.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = {
            'epoch': [], 
            'pT_loss': [],
            'eta_loss': [],
            'phi_loss': [],
            'energy_loss': [],
            'val_loss': []
        }

    def train(self) -> Tuple[Dict[str, List[float]], nn.Module]:
        try:
            for cb in self.callbacks:
                cb.on_train_begin(trainer=self)

            total_steps = self.num_epochs * len(self.train_loader)
            start_step = self.start_epoch * len(self.train_loader)

            if self.progress_bar:
                global_bar = tqdm(
                    total=total_steps,
                    initial=start_step,
                    desc="Training",
                    dynamic_ncols=True
                )
            else:
                class _NoOpBar:
                    def set_postfix(self, *args, **kwargs):
                        pass
                    def update(self, *args, **kwargs):
                        pass
                global_bar = _NoOpBar()

            for epoch in range(self.start_epoch, self.num_epochs):
                for cb in self.callbacks:
                    cb.on_epoch_begin(epoch, trainer=self)

                self.model.train()
                running_loss = 0.0

                # For tracking each loss component
                loss_components_sum = None

                for batch_idx, (X, y, mask_idx) in enumerate(self.train_loader):
                    step = epoch * len(self.train_loader) + batch_idx + 1

                    X = X.to(self.device, non_blocking=self.pin_memory)
                    y = y.to(self.device, non_blocking=self.pin_memory)

                    self.optimizer.zero_grad()

                    outputs = self.model(X, mask_idx)

                    loss, components = self.criterion(outputs, y)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                    # Track sum of each component for average
                    if loss_components_sum is None:
                        loss_components_sum = [0.0 for _ in components]

                    for i, comp in enumerate(components):
                        loss_components_sum[i] += comp.item()

                    avg_components = [s / (batch_idx + 1) for s in loss_components_sum]

                    avg_loss = running_loss / (batch_idx + 1)
                    
                    # Short summary
                    if step % self.logging_steps == 0 or step == total_steps:
                        tqdm.write(
                            f"step: {step}/{total_steps} | "
                            f"pT_loss: {avg_components[0]:.4f} | "
                            f"eta_loss: {avg_components[1]:.4f} | "
                            f"phi_loss: {avg_components[2]:.4f} | "
                            f"energy_loss: {avg_components[3]:.4f} | "
                            f"total_train_loss: {avg_loss:.4f}"
                            #f"momentum_loss: {avg_components[1]:.4f}"
                        )

                    global_bar.set_postfix({
                        "epoch": f"{epoch + 1}/{self.num_epochs}",
                        "avg_loss": f"{avg_loss:.4f}"
                    })
                    global_bar.update(1)

                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_components_sum = None

                with torch.no_grad():
                    for X_val, y_val, mask_idx in self.val_loader:
                        X_val = X_val.to(self.device, non_blocking=self.pin_memory)
                        y_val = y_val.to(self.device, non_blocking=self.pin_memory)

                        outputs_val = self.model(X_val, mask_idx)

                        vloss, v_components = self.criterion(outputs_val, y_val)
                        val_loss += vloss.item()

                        if val_components_sum is None:
                            val_components_sum = [0.0 for _ in v_components]
                        for i, comp in enumerate(v_components):
                            val_components_sum[i] += comp.item()

                val_loss /= len(self.val_loader)
                avg_val_components = [s / len(self.val_loader) for s in val_components_sum]

                # Short summary for validation
                tqdm.write(
                    f"epoch: {epoch + 1}/{self.num_epochs} | "
                    f"pT_loss: {avg_val_components[0]:.4f} | "
                    f"eta_loss: {avg_val_components[1]:.4f} | "
                    f"phi_loss: {avg_val_components[2]:.4f} | "
                    f"energy_loss: {avg_val_components[3]:.4f} | "
                    f"total_val_loss: {val_loss:.4f}"
                    #f"momentum_loss: {avg_val_components[1]:.4f}"
                )

                if self.scheduler:
                    self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']

                # Save best model
                if self.best_model_path and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), self.best_model_path)

                self.history['epoch'].append(epoch + 1)
                self.history['pT_loss'].append(avg_components[0])
                self.history['eta_loss'].append(avg_components[1])
                self.history['phi_loss'].append(avg_components[2])
                self.history['energy_loss'].append(avg_components[3])
                self.history['val_loss'].append(val_loss)

                self.save_checkpoint(epoch)

                # Log results
                logs = {
                    'epoch': epoch + 1,
                    'train_loss': avg_loss,
                    'val_loss': val_loss,
                    'learning_rate': current_lr,
                }
                self.log_csv(logs)

                # Callback after each epoch
                for cb in self.callbacks:
                    cb.on_epoch_end(epoch, trainer=self, logs=logs)

                # Break if any callback says to stop
                if any(getattr(cb, 'should_stop', False) for cb in self.callbacks):
                    break

            # Callback after training
            for cb in self.callbacks:
                cb.on_train_end(trainer=self)
        except KeyboardInterrupt:
            print(f"\nTraining interrupted at epoch {epoch + 1}. Saving current checkpoint.")
            self.save_checkpoint(epoch)

        return self.history, self.model

    @torch.no_grad()
    def evaluate(
        self,
        plot: Optional[Union[Callable, List[Callable]]] = None
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        if self.test_loader is None:
            raise ValueError("Test dataset is not provided.")
        
        self.model.eval()
        test_loss = 0.0
        test_metric = 0.0
        test_components_sum = None
        y_true_list = []
        y_pred_list = []

        for X_test, y_test, mask_idx in self.test_loader:
            X_test = X_test.to(self.device, non_blocking=self.pin_memory)
            y_test = y_test.to(self.device, non_blocking=self.pin_memory)

            outputs_test = self.model(X_test, mask_idx)

            # Guarantee phi to be in [-pi, pi]
            outputs_test[..., 2] = torch.remainder(outputs_test[..., 2] + torch.pi, 2 * torch.pi) - torch.pi
            
            tloss, t_components = self.criterion(outputs_test, y_test)
            test_loss += tloss.item()

            if self.metric:
                test_metric += self.metric(outputs_test, y_test)

            if test_components_sum is None:
                test_components_sum = [0.0 for _ in t_components]

            for i, comp in enumerate(t_components):
                test_components_sum[i] += comp.item()

            y_pred_list.append(outputs_test.cpu().numpy())
            y_true_list.append(y_test.cpu().numpy())

        test_loss /= len(self.test_loader)
        test_metric /= len(self.test_loader)
        avg_test_components = [s / len(self.test_loader) for s in test_components_sum]

        print(
            f"test_loss: {test_loss:.4f} | "
            f"pT_loss: {avg_test_components[0]:.4f} | "
            f"eta_loss: {avg_test_components[1]:.4f} | "
            f"phi_loss: {avg_test_components[2]:.4f} | "
            f"energy_loss: {avg_test_components[3]:.4f}"
        )

        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)

        # Visualization
        if plot is not None:
            if isinstance(plot, list):
                for p in plot:
                    p(y_true, y_pred)
            else:
                plot(y_true, y_pred)

        return test_loss, test_metric, y_true, y_pred        