"""
Checkpoint Management for KLTN Training
=======================================

Handles saving, loading, and managing model checkpoints during training.

Features:
- Automatic checkpoint saving at intervals
- Best model tracking based on validation metrics
- Training state recovery (optimizer, scheduler, epoch)
- Checkpoint rotation to save disk space

Usage:
    >>> manager = CheckpointManager(save_dir='./checkpoints')
    >>> manager.save(model, optimizer, epoch, metrics={'val_loss': 0.5})
    >>> epoch = manager.load(model, optimizer)
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints during training.
    
    Args:
        save_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep (0 = unlimited)
        save_best_only: Only save when metric improves
        metric_name: Name of metric to track for best model
        mode: 'min' or 'max' for metric comparison
        
    Example:
        >>> manager = CheckpointManager('./checkpoints', max_checkpoints=5)
        >>> for epoch in range(100):
        ...     # Training loop
        ...     manager.save(model, optimizer, epoch, {'loss': train_loss})
    """
    
    def __init__(
        self,
        save_dir: str = "./checkpoints",
        max_checkpoints: int = 5,
        save_best_only: bool = False,
        metric_name: str = "val_loss",
        mode: str = "min",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.mode = mode
        
        # Track best metric
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = -1
        
        # Track all checkpoints for rotation
        self.checkpoint_history: List[Path] = []
        
        # Load existing checkpoint history
        self._load_history()
        
        logger.info(f"CheckpointManager initialized at {save_dir}")
    
    def save(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        scheduler: Optional[Any] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Save a checkpoint.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state to save
            epoch: Current epoch number
            metrics: Dictionary of metric values
            scheduler: Optional learning rate scheduler
            extra_state: Any additional state to save
            filename: Custom filename (default: checkpoint_epoch_{epoch}.pth)
            
        Returns:
            Path to saved checkpoint, or None if not saved
        """
        metrics = metrics or {}
        
        # Check if we should save based on metric
        if self.save_best_only and self.metric_name in metrics:
            current_metric = metrics[self.metric_name]
            is_best = (
                (self.mode == 'min' and current_metric < self.best_metric) or
                (self.mode == 'max' and current_metric > self.best_metric)
            )
            if not is_best:
                logger.debug(f"Skipping save: {self.metric_name}={current_metric:.4f} "
                           f"not better than {self.best_metric:.4f}")
                return None
            
            self.best_metric = current_metric
            self.best_epoch = epoch
        
        # Build checkpoint state
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if extra_state is not None:
            checkpoint['extra_state'] = extra_state
        
        # Determine filename
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pth"
        
        filepath = self.save_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
        
        # Update history and rotate old checkpoints
        self.checkpoint_history.append(filepath)
        self._rotate_checkpoints()
        
        # Save best model separately
        if self.metric_name in metrics:
            current_metric = metrics[self.metric_name]
            is_best = (
                (self.mode == 'min' and current_metric <= self.best_metric) or
                (self.mode == 'max' and current_metric >= self.best_metric)
            )
            if is_best:
                best_path = self.save_dir / "best_model.pth"
                shutil.copy(filepath, best_path)
                logger.info(f"Updated best model (epoch {epoch}, "
                          f"{self.metric_name}={current_metric:.4f})")
        
        # Always save latest
        latest_path = self.save_dir / "checkpoint_latest.pth"
        shutil.copy(filepath, latest_path)
        
        return filepath
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        filename: str = "checkpoint_latest.pth",
        device: Optional[torch.device] = None,
    ) -> int:
        """
        Load a checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            filename: Checkpoint filename to load
            device: Device to map tensors to
            
        Returns:
            Epoch number from checkpoint (0 if no checkpoint found)
        """
        filepath = self.save_dir / filename
        
        if not filepath.exists():
            logger.warning(f"No checkpoint found at {filepath}")
            return 0
        
        # Load checkpoint
        map_location = device if device else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model state from epoch {checkpoint['epoch']}")
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Loaded optimizer state")
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Loaded scheduler state")
        
        return checkpoint['epoch']
    
    def load_best(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> int:
        """Load the best model checkpoint."""
        return self.load(model, filename="best_model.pth", device=device)
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        latest = self.save_dir / "checkpoint_latest.pth"
        return latest if latest.exists() else None
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        best = self.save_dir / "best_model.pth"
        return best if best.exists() else None
    
    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints to stay within max_checkpoints limit."""
        if self.max_checkpoints <= 0:
            return
        
        # Keep special checkpoints
        special = {'checkpoint_latest.pth', 'best_model.pth'}
        
        # Filter to only regular checkpoints
        regular = [p for p in self.checkpoint_history if p.name not in special]
        
        # Remove oldest if over limit
        while len(regular) > self.max_checkpoints:
            oldest = regular.pop(0)
            if oldest.exists():
                oldest.unlink()
                logger.debug(f"Removed old checkpoint: {oldest}")
            self.checkpoint_history.remove(oldest)
    
    def _load_history(self) -> None:
        """Load checkpoint history from directory."""
        if self.save_dir.exists():
            for path in sorted(self.save_dir.glob("checkpoint_epoch_*.pth")):
                self.checkpoint_history.append(path)


class EarlyStopping:
    """
    Early stopping callback to stop training when metric stops improving.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max' for metric comparison
        
    Example:
        >>> early_stop = EarlyStopping(patience=10)
        >>> for epoch in range(100):
        ...     val_loss = validate(model)
        ...     if early_stop(val_loss):
        ...         print("Early stopping triggered")
        ...         break
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
    
    def __call__(self, metric: float, epoch: int = 0) -> bool:
        """
        Check if training should stop.
        
        Args:
            metric: Current metric value
            epoch: Current epoch (for logging)
            
        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = metric < self.best_metric - self.min_delta
        else:
            improved = metric > self.best_metric + self.min_delta
        
        if improved:
            self.best_metric = metric
            self.best_epoch = epoch
            self.counter = 0
            return False
        
        self.counter += 1
        
        if self.counter >= self.patience:
            logger.info(f"Early stopping: no improvement for {self.patience} epochs "
                       f"(best: {self.best_metric:.4f} at epoch {self.best_epoch})")
            return True
        
        return False
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0


class MetricsLogger:
    """
    Training metrics logger with file and console output.
    
    Args:
        log_dir: Directory to save metric logs
        experiment_name: Name for this experiment
        
    Example:
        >>> logger = MetricsLogger('./logs', 'exp_001')
        >>> for epoch in range(100):
        ...     logger.log({'epoch': epoch, 'loss': train_loss, 'acc': val_acc})
        >>> logger.save()
    """
    
    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: str = "experiment",
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.json"
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics for a training step.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (added to metrics if provided)
        """
        if step is not None:
            metrics['step'] = step
        
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(metrics)
        
        # Auto-save periodically
        if len(self.metrics_history) % 10 == 0:
            self._save_incremental()
    
    def get_history(self, metric_name: str) -> List[float]:
        """Get history of a specific metric."""
        return [m[metric_name] for m in self.metrics_history if metric_name in m]
    
    def get_best(self, metric_name: str, mode: str = 'min') -> Dict[str, Any]:
        """Get the entry with best value for a metric."""
        entries = [m for m in self.metrics_history if metric_name in m]
        if not entries:
            return {}
        
        if mode == 'min':
            return min(entries, key=lambda x: x[metric_name])
        return max(entries, key=lambda x: x[metric_name])
    
    def save(self) -> Path:
        """Save all metrics to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Saved metrics to {self.log_file}")
        return self.log_file
    
    def _save_incremental(self) -> None:
        """Save metrics incrementally."""
        try:
            self.save()
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")
    
    def summary(self) -> str:
        """Generate summary of logged metrics."""
        if not self.metrics_history:
            return "No metrics logged"
        
        lines = [f"Experiment: {self.experiment_name}"]
        lines.append(f"Total entries: {len(self.metrics_history)}")
        
        # Get all metric names
        all_keys = set()
        for m in self.metrics_history:
            all_keys.update(m.keys())
        
        # Skip meta keys
        metric_keys = all_keys - {'timestamp', 'step', 'epoch'}
        
        for key in sorted(metric_keys):
            values = [m[key] for m in self.metrics_history if key in m and isinstance(m[key], (int, float))]
            if values:
                lines.append(f"  {key}: min={min(values):.4f}, max={max(values):.4f}, "
                           f"last={values[-1]:.4f}")
        
        return '\n'.join(lines)


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test CheckpointManager
    print("Testing CheckpointManager...")
    
    # Create dummy model and optimizer
    model = nn.Linear(10, 5)
    optimizer = optim.Adam(model.parameters())
    
    manager = CheckpointManager('./test_checkpoints', max_checkpoints=3)
    
    # Save several checkpoints
    for epoch in range(5):
        metrics = {'val_loss': 1.0 - epoch * 0.1, 'accuracy': epoch * 0.2}
        manager.save(model, optimizer, epoch, metrics)
    
    # Load latest
    new_model = nn.Linear(10, 5)
    loaded_epoch = manager.load(new_model)
    print(f"Loaded epoch: {loaded_epoch}")
    
    # Test EarlyStopping
    print("\nTesting EarlyStopping...")
    early_stop = EarlyStopping(patience=3)
    
    losses = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.88]  # Stops improving after index 2
    for i, loss in enumerate(losses):
        if early_stop(loss, i):
            print(f"Stopped at epoch {i}")
            break
    
    # Test MetricsLogger
    print("\nTesting MetricsLogger...")
    metrics_logger = MetricsLogger('./test_logs', 'test_exp')
    
    for i in range(10):
        metrics_logger.log({'epoch': i, 'loss': 1.0 - i * 0.1, 'lr': 0.001})
    
    print(metrics_logger.summary())
    metrics_logger.save()
    
    # Cleanup
    import shutil
    shutil.rmtree('./test_checkpoints', ignore_errors=True)
    shutil.rmtree('./test_logs', ignore_errors=True)
    
    print("\nAll tests passed!")
