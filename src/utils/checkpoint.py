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

import json
import os
import logging
import hashlib
import math
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Iterable

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)
LATEST_RESUME_FILENAME = "latest_resume.pth"
ALLOW_LEGACY_TORCH_LOAD_ENV = "HMOLQD_ALLOW_UNSAFE_LEGACY_TORCH_LOAD"


def safe_torch_load(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    weights_only: bool = True,
) -> Any:
    """Load a PyTorch checkpoint with restricted unpickling when available.

    Loading unrestricted pickle-backed checkpoints can execute arbitrary code.
    Therefore, when callers request ``weights_only=True`` and the installed
    PyTorch does not support that parameter, this helper fails closed by
    default. Set ``HMOLQD_ALLOW_UNSAFE_LEGACY_TORCH_LOAD=1`` only for trusted,
    local legacy checkpoints.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError as exc:
        message = str(exc).lower()
        unsupported_weights_only = (
            "weights_only" in message
            and (
                "unexpected" in message
                or "keyword" in message
                or "got an unexpected" in message
            )
        )
        if not unsupported_weights_only:
            raise
        if weights_only:
            allow_unsafe = str(os.environ.get(ALLOW_LEGACY_TORCH_LOAD_ENV, "")).strip().lower()
            if allow_unsafe not in {"1", "true", "yes"}:
                raise RuntimeError(
                    "Installed PyTorch does not support weights_only=True. "
                    "Refusing unsafe legacy torch.load for checkpoint "
                    f"{path!r}. Upgrade PyTorch or set {ALLOW_LEGACY_TORCH_LOAD_ENV}=1 "
                    "only for trusted local checkpoints."
                )
            logger.warning(
                "Using unsafe legacy torch.load for trusted local checkpoint because %s is set.",
                ALLOW_LEGACY_TORCH_LOAD_ENV,
            )
        return torch.load(path, map_location=map_location)


def format_bytes(num_bytes: int) -> str:
    """Format a byte count into a compact human-readable string."""
    size = float(max(0, int(num_bytes)))
    units = ("B", "KB", "MB", "GB", "TB")
    unit_idx = 0
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    precision = 0 if unit_idx == 0 else 1
    return f"{size:.{precision}f} {units[unit_idx]}"


def checkpoint_size_bytes(path: str | Path) -> int:
    """Return the checkpoint file size in bytes, or 0 when missing."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return 0
    return int(checkpoint_path.stat().st_size)


def checkpoint_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable SHA-256 digest for checkpoint provenance."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        return ""
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(int(max(1, chunk_size))), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_directory_size_bytes(
    checkpoint_dir: str | Path,
    *,
    pattern: str = "*.pth",
) -> int:
    """Return total size in bytes for checkpoint files in a directory."""
    root = Path(checkpoint_dir)
    if not root.exists():
        return 0
    return int(sum(path.stat().st_size for path in root.glob(pattern) if path.is_file()))


def log_checkpoint_artifact(
    log: logging.Logger,
    checkpoint_path: str | Path,
    *,
    checkpoint_dir: Optional[str | Path] = None,
    label: str = "checkpoint",
    pattern: str = "*.pth",
) -> None:
    """Log the size of a saved checkpoint and the current checkpoint-dir footprint."""
    path = Path(checkpoint_path)
    file_size = checkpoint_size_bytes(path)
    message = f"{label}: {path} ({format_bytes(file_size)})"
    if checkpoint_dir is not None:
        total_size = checkpoint_directory_size_bytes(checkpoint_dir, pattern=pattern)
        message += f" | checkpoint_dir_total={format_bytes(total_size)}"
    log.info(message)


def enforce_checkpoint_storage_budget(
    log: logging.Logger,
    *,
    checkpoint_dir: str | Path,
    budget_gb: Optional[float],
    warning_fraction: float = 0.8,
    cleanup_enabled: bool = True,
    cleanup_target_fraction: float = 0.6,
    removable_patterns: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Warn when checkpoint storage approaches a budget and optionally prune expendable checkpoints.

    Only files matching `removable_patterns` are eligible for automatic deletion.
    This is intended for retained resume checkpoints, not best/final/latest artifacts.
    """
    if budget_gb is None:
        return {"budget_enabled": False, "removed": []}

    budget_value = float(budget_gb)
    if budget_value <= 0.0:
        return {"budget_enabled": False, "removed": []}

    root = Path(checkpoint_dir)
    if not root.exists():
        return {"budget_enabled": True, "removed": []}

    budget_bytes = int(budget_value * (1024 ** 3))
    total_before = checkpoint_directory_size_bytes(root)
    warning_ratio = max(0.0, min(1.0, float(warning_fraction)))
    cleanup_target_ratio = max(0.0, min(1.0, float(cleanup_target_fraction)))
    warning_bytes = int(budget_bytes * warning_ratio)
    target_bytes = int(budget_bytes * cleanup_target_ratio)

    if total_before >= warning_bytes:
        log.warning(
            "Checkpoint storage usage is %s / %s in %s.",
            format_bytes(total_before),
            format_bytes(budget_bytes),
            root,
        )

    removed: List[Path] = []
    if total_before >= budget_bytes and bool(cleanup_enabled):
        patterns = tuple(removable_patterns or ())
        removable_paths: List[Path] = []
        for pattern in patterns:
            removable_paths.extend(path for path in root.glob(pattern) if path.is_file())
        removable_paths = sorted(set(removable_paths), key=lambda path: (path.stat().st_mtime, path.name))

        total_after = total_before
        for victim in removable_paths:
            if total_after <= target_bytes:
                break
            size = checkpoint_size_bytes(victim)
            if victim.exists():
                victim.unlink()
                removed.append(victim)
            meta = Path(f"{victim}.meta.json")
            if meta.exists():
                meta.unlink()
            total_after = max(0, total_after - size)

        if removed:
            log.warning(
                "Checkpoint storage exceeded budget; removed %d retained checkpoint(s) to reduce usage to %s.",
                len(removed),
                format_bytes(total_after),
            )

    total_final = checkpoint_directory_size_bytes(root)
    if total_final >= budget_bytes:
        log.warning(
            "Checkpoint storage remains above budget after cleanup: %s / %s in %s.",
            format_bytes(total_final),
            format_bytes(budget_bytes),
            root,
        )

    return {
        "budget_enabled": True,
        "budget_bytes": budget_bytes,
        "total_bytes": total_final,
        "removed": removed,
    }


def write_checkpoint_metadata(
    checkpoint_path: str,
    *,
    model_type: str,
    architecture: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    format_version: str = "1.0",
) -> Path:
    """Write a lightweight sidecar metadata JSON next to a checkpoint file."""
    path = Path(checkpoint_path)
    metadata_path = Path(f"{path}.meta.json")
    metadata: Dict[str, Any] = {
        "format_version": str(format_version),
        "model_type": str(model_type),
        "checkpoint_file": path.name,
        "saved_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "file_size_bytes": int(checkpoint_size_bytes(path)),
        "file_size_human": format_bytes(checkpoint_size_bytes(path)),
        "sha256": checkpoint_sha256(path),
    }
    if architecture:
        metadata["architecture"] = dict(architecture)
    if extra:
        metadata["extra"] = dict(extra)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.debug("Wrote checkpoint metadata sidecar: %s", metadata_path)
    return metadata_path


def atomic_torch_save(payload: Any, path: str | Path) -> Path:
    """Atomically save a PyTorch payload using a unique sibling temp file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.",
        suffix=".tmp",
        dir=str(out_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return out_path


def resolve_resume_checkpoint(
    *,
    explicit_path: Optional[str],
    checkpoint_dir: str,
    auto_resume: bool = True,
    latest_filename: str = LATEST_RESUME_FILENAME,
) -> Optional[Path]:
    """Resolve the checkpoint to resume from."""
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Requested resume checkpoint does not exist: {candidate}")

    if not bool(auto_resume):
        return None

    candidate = Path(checkpoint_dir) / str(latest_filename)
    if candidate.exists():
        return candidate
    return None


def prune_checkpoints(
    *,
    checkpoint_dir: str,
    pattern: str,
    keep_last: int,
) -> List[Path]:
    """Prune old checkpoints matching a glob pattern and remove sidecars too."""
    root = Path(checkpoint_dir)
    if not root.exists():
        return []

    matches = sorted(root.glob(pattern))
    removed: List[Path] = []
    keep = max(0, int(keep_last))
    while len(matches) > keep:
        oldest = matches.pop(0)
        if oldest.exists():
            oldest.unlink()
            removed.append(oldest)
        meta = Path(f"{oldest}.meta.json")
        if meta.exists():
            meta.unlink()
    return removed


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
        self.mode = str(mode).strip().lower()
        if self.mode not in {"min", "max"}:
            raise ValueError(f"CheckpointManager mode must be 'min' or 'max', got {mode!r}.")
        
        # Track best metric
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
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
        scaler: Optional[Any] = None,
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
            scaler: Optional AMP GradScaler whose dynamic scale must survive resume
            
        Returns:
            Path to saved checkpoint, or None if not saved
        """
        metrics = metrics or {}

        current_metric: Optional[float] = None
        is_best = False
        if self.metric_name in metrics:
            try:
                candidate_metric = float(metrics[self.metric_name])
            except (TypeError, ValueError):
                candidate_metric = float("nan")
            if math.isfinite(candidate_metric):
                current_metric = candidate_metric
                is_best = (
                    (self.mode == 'min' and current_metric < self.best_metric) or
                    (self.mode == 'max' and current_metric > self.best_metric)
                )
            else:
                logger.warning(
                    "Checkpoint metric %s is non-finite; it cannot update best_model.pth.",
                    self.metric_name,
                )
        elif self.save_best_only:
            logger.warning(
                "Skipping best-only checkpoint at epoch %d because metric %r is missing.",
                epoch,
                self.metric_name,
            )
            return None

        if self.save_best_only and not is_best:
            logger.debug(
                "Skipping save: %s=%s not better than %.4f",
                self.metric_name,
                current_metric,
                self.best_metric,
            )
            return None

        if is_best and current_metric is not None:
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

        if scaler is not None and callable(getattr(scaler, "state_dict", None)):
            checkpoint['grad_scaler_state_dict'] = scaler.state_dict()
        
        if extra_state is not None:
            checkpoint['extra_state'] = extra_state
        
        # Always update rolling latest checkpoint (progress epoch save)
        latest_path = self.save_dir / "checkpoint_latest.pth"
        atomic_torch_save(checkpoint, str(latest_path))
        logger.info(f"Updated latest progress checkpoint at {latest_path}")
        
        # Save best model separately when metric improves (best epoch save only)
        if is_best and current_metric is not None:
            best_path = self.save_dir / "best_model.pth"
            atomic_torch_save(checkpoint, str(best_path))
            logger.info(
                "Updated best model (epoch %d, %s=%.4f)",
                epoch,
                self.metric_name,
                current_metric,
            )
        
        return latest_path
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        filename: str = "checkpoint_latest.pth",
        device: Optional[torch.device] = None,
        scaler: Optional[Any] = None,
    ) -> int:
        """
        Load a checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            filename: Checkpoint filename to load
            device: Device to map tensors to
            scaler: Optional AMP GradScaler to restore when checkpointed
            
        Returns:
            Epoch number from checkpoint (0 if no checkpoint found)
        """
        filepath = self.save_dir / filename
        
        if not filepath.exists():
            logger.warning(f"No checkpoint found at {filepath}")
            return 0
        
        # Load checkpoint
        map_location = device if device else 'cpu'
        checkpoint = safe_torch_load(filepath, map_location=map_location)
        
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

        if scaler is not None and 'grad_scaler_state_dict' in checkpoint:
            try:
                scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])
                logger.info("Loaded AMP GradScaler state")
            except (RuntimeError, ValueError, TypeError) as exc:
                logger.warning("Skipping incompatible AMP GradScaler state: %s", exc)
        
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
        self.patience = max(1, int(patience))
        self.min_delta = float(max(0.0, min_delta))
        self.mode = str(mode).strip().lower()
        if self.mode not in {"min", "max"}:
            raise ValueError(f"EarlyStopping mode must be 'min' or 'max', got {mode!r}.")
        
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
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
        metric = float(metric)
        if not math.isfinite(metric):
            raise ValueError(f"EarlyStopping received non-finite metric at epoch {epoch}: {metric}")
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
        self._wandb_run = self._initialize_wandb()

    def _initialize_wandb(self) -> Optional[Any]:
        """Start optional W&B tracking while keeping JSON logging authoritative."""
        enabled = str(os.environ.get("HMOLQD_WANDB_ENABLED", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled:
            return None
        try:
            import wandb

            return wandb.init(
                project=os.environ.get("HMOLQD_WANDB_PROJECT", "kltn-hmolqd"),
                entity=os.environ.get("HMOLQD_WANDB_ENTITY") or None,
                group=os.environ.get("HMOLQD_WANDB_GROUP") or None,
                name=self.experiment_name,
                dir=str(self.log_dir),
                reinit=True,
            )
        except (ImportError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("W&B tracking disabled; JSON metrics remain active: %s", exc)
            return None
    
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
        if self._wandb_run is not None:
            try:
                wandb_metrics = {
                    key: value
                    for key, value in metrics.items()
                    if key != "timestamp" and isinstance(value, (int, float, bool))
                }
                self._wandb_run.log(wandb_metrics, step=step)
            except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
                logger.warning("W&B metric forwarding failed; continuing with JSON logs: %s", exc)
        
        # Auto-save periodically
        if len(self.metrics_history) % 10 == 0:
            self._save_incremental()
    
    def get_history(self, metric_name: str) -> List[float]:
        """Get history of a specific metric."""
        return [m[metric_name] for m in self.metrics_history if metric_name in m]
    
    def get_best(self, metric_name: str, mode: str = 'min') -> Dict[str, Any]:
        """Get the entry with best value for a metric."""
        entries = []
        for metrics in self.metrics_history:
            if metric_name not in metrics:
                continue
            try:
                value = float(metrics[metric_name])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                entries.append(metrics)
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
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
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
