import csv
import logging
import os
import random
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from deepecgkit.utils.device import resolve_device

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

#: Used for any key a train_config leaves out.
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_SCHEDULER = {"factor": 0.5, "patience": 5}


def _package_version() -> str | None:
    try:
        return version("deepecgkit")
    except PackageNotFoundError:
        return None


class ECGTrainer:
    """
    Trainer for ECG signal classification and regression models.

    Wraps a plain nn.Module and provides fit/test methods with built-in
    early stopping, checkpointing, LR scheduling, and CSV metric logging.

    Args:
        model: The ECG model to train (any nn.Module)
        train_config: Optional dictionary of training settings; every key has
            a default, so ``ECGTrainer(model)`` trains a multi-class classifier:
            - learning_rate: Learning rate for optimizer (default 1e-3)
            - scheduler: Dict with 'factor' and 'patience' for
              ReduceLROnPlateau (default factor 0.5, patience 5)
            - binary_classification: Bool, if True uses BCE loss for binary tasks
            - multi_label: Bool, if True uses BCE loss for multi-label tasks
            - task_type: 'classification' (default) or 'regression'
            - pos_weight: Optional list of positive class weights for BCE loss
        device: Device to train on ('auto', 'cpu', 'cuda', 'mps')
        use_plateau_scheduler: If True, uses ReduceLROnPlateau, else StepLR

    Example:
        >>> model = KanResWideX(input_channels=1, output_size=4)
        >>> trainer = ECGTrainer(model)
        >>> trainer.fit(data_module, epochs=50)
    """

    def __init__(
        self,
        model: nn.Module,
        train_config: dict[str, Any] | None = None,
        device: str | torch.device = "auto",
        use_plateau_scheduler: bool = True,
    ) -> None:
        self.model = model
        train_config = dict(train_config or {})
        train_config.setdefault("learning_rate", DEFAULT_LEARNING_RATE)
        train_config["scheduler"] = {**DEFAULT_SCHEDULER, **(train_config.get("scheduler") or {})}
        self.train_config = train_config

        self.learning_rate = train_config["learning_rate"]
        self.scheduler_factor = train_config["scheduler"]["factor"]
        self.scheduler_patience = train_config["scheduler"]["patience"]
        self.use_plateau_scheduler = use_plateau_scheduler
        self.multi_label = train_config.get("multi_label", False)

        if self.multi_label or train_config.get("binary_classification", False):
            pos_weight = train_config.get("pos_weight")
            if pos_weight is not None:
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif train_config.get("task_type", "classification") == "classification":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = torch.nn.MSELoss()

        self.device = resolve_device(device)

        self.model.to(self.device)
        self.criterion.to(self.device)

        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None
        self.test_predictions: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []
        self.test_probabilities: list[torch.Tensor] = []
        self.test_logits: list[torch.Tensor] = []
        self.log_dir: str | None = None
        self.best_checkpoint_path: str | None = None
        self.best_val_loss = float("inf")
        #: How to rebuild the model: registry name, lead count, output size.
        self.model_config: dict[str, Any] = {}
        #: What the model was trained on: dataset, preprocessing, class names.
        #: Both are written into every checkpoint and filled in by fit() when
        #: left empty, so a checkpoint can be loaded and used on its own.
        self.data_config: dict[str, Any] = {}
        self._gradient_clip_val: float | None = None

    @property
    def _is_binary(self) -> bool:
        return isinstance(self.criterion, torch.nn.BCEWithLogitsLoss) and not self.multi_label

    @property
    def _is_classification(self) -> bool:
        return isinstance(self.criterion, torch.nn.CrossEntropyLoss | torch.nn.BCEWithLogitsLoss)

    def _calculate_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.multi_label:
            return self.criterion(y_hat, y.float())
        if self._is_binary:
            return self.criterion(y_hat.squeeze(-1), y.float())
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            return self.criterion(y_hat, y.long())
        return self.criterion(y_hat.float(), y.float())

    def _compute_acc(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        with torch.no_grad():
            if self.multi_label:
                preds = (y_hat > 0).float()
                return (preds == y.float()).float().mean().item()
            if self._is_binary:
                preds = (y_hat.squeeze(-1) > 0).long()
                return (preds == y.long()).float().mean().item()
            return (torch.argmax(y_hat, dim=1) == y.long()).float().mean().item()

    def _setup_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.use_plateau_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)

    def _run_epoch(self, dataloader: DataLoader, train: bool = True) -> tuple[float, float | None]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        ctx = torch.no_grad() if not train else _NullContext()
        with ctx:
            for batch in dataloader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                y_hat = self.model(x)
                loss = self._calculate_loss(y_hat, y)

                if train:
                    if hasattr(self.model, "l2_regularization_loss"):
                        loss = loss + self.model.l2_regularization_loss()
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self._gradient_clip_val is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self._gradient_clip_val
                        )
                    self.optimizer.step()

                total_loss += loss.item()
                if self._is_classification:
                    total_acc += self._compute_acc(y_hat, y)
                n_batches += 1
                if hasattr(dataloader, "set_postfix"):
                    dataloader.set_postfix(loss=f"{total_loss / n_batches:.4f}", refresh=False)

        avg_loss = total_loss / max(n_batches, 1)
        avg_acc = total_acc / max(n_batches, 1) if self._is_classification else None
        return avg_loss, avg_acc

    def fit(
        self,
        data_module: Any,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
        progress_bar: bool = True,
        gradient_clip_val: float | None = None,
        save_top_k: int = 3,
        restore_best: bool = True,
    ) -> None:
        """Train the model.

        On completion the weights achieving the lowest validation loss are restored,
        so any subsequent test() or validate() call evaluates the selected model
        rather than the final epoch. Requires checkpoint_dir to be set; without it
        no checkpoint exists to restore and the final-epoch weights are kept.

        Args:
            data_module: ECGDataModule (or any object with train_dataloader/val_dataloader)
            epochs: Maximum number of training epochs
            early_stopping_patience: Stop after this many epochs without val_loss improvement
            checkpoint_dir: Directory to save checkpoints (None to disable)
            log_dir: Directory to save CSV metrics log (None to disable)
            progress_bar: Whether to show tqdm progress: one bar over the epochs and,
                on an interactive terminal, a second one over the batches of the
                current epoch
            gradient_clip_val: Max gradient norm for clipping (None to disable)
            save_top_k: Number of best checkpoints to keep
            restore_best: Reload the best-validation-loss checkpoint when training ends
        """
        self._gradient_clip_val = gradient_clip_val
        self._setup_optimizer()

        if hasattr(data_module, "setup"):
            data_module.setup(stage="fit")

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        self._describe_run(data_module)

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        csv_writer = None
        csv_file = None
        if log_dir is not None:
            self.log_dir = str(log_dir)
            os.makedirs(log_dir, exist_ok=True)
            metrics_path = Path(log_dir) / "metrics.csv"
            csv_file = open(metrics_path, "w", newline="")
            fieldnames = ["epoch", "train_loss", "val_loss"]
            if self._is_classification:
                fieldnames.extend(["train_acc", "val_acc"])
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

        self.best_val_loss = float("inf")
        patience_counter = 0
        saved_checkpoints = []

        use_tqdm = progress_bar and tqdm is not None
        epoch_iter = tqdm(range(epochs), desc="Training") if use_tqdm else range(epochs)
        # A bar per batch is only readable where it can redraw in place; in a
        # log file it would add a line every refresh.
        show_batches = use_tqdm and sys.stderr.isatty()

        try:
            for epoch in epoch_iter:
                batches: Any = train_loader
                if show_batches:
                    batches = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
                train_loss, train_acc = self._run_epoch(batches, train=True)
                val_loss, val_acc = self._run_epoch(val_loader, train=False)

                if self.use_plateau_scheduler:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                if use_tqdm:
                    desc = f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
                    if self._is_classification:
                        desc += f" val_acc={val_acc:.4f}"
                    epoch_iter.set_description(desc)

                if csv_writer is not None:
                    row = {
                        "epoch": epoch + 1,
                        "train_loss": f"{train_loss:.6f}",
                        "val_loss": f"{val_loss:.6f}",
                    }
                    if self._is_classification:
                        row["train_acc"] = f"{train_acc:.6f}"
                        row["val_acc"] = f"{val_acc:.6f}"
                    csv_writer.writerow(row)
                    csv_file.flush()

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0

                    if checkpoint_dir is not None:
                        ckpt_path = (
                            checkpoint_dir / f"epoch={epoch + 1:02d}-val_loss={val_loss:.4f}.pt"
                        )
                        self.save_checkpoint(str(ckpt_path), epoch=epoch + 1)
                        self.best_checkpoint_path = str(ckpt_path)
                        saved_checkpoints.append((val_loss, str(ckpt_path)))
                        saved_checkpoints.sort(key=lambda x: x[0])
                        while len(saved_checkpoints) > save_top_k:
                            _, old_path = saved_checkpoints.pop()
                            if os.path.exists(old_path):
                                os.remove(old_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if not use_tqdm:
                            logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
        finally:
            if csv_file is not None:
                csv_file.close()

        if restore_best:
            self.restore_best_checkpoint()

    def _describe_run(self, data_module: Any) -> None:
        """Fill in whatever model_config and data_config leave unset.

        Callers such as the CLI set both explicitly; this covers direct use of
        the trainer, so its checkpoints still name their architecture and
        record the preprocessing their inputs went through.

        Args:
            data_module: The data module being trained on
        """
        described: dict[str, Any] = {}
        describe = getattr(data_module, "describe", None)
        if callable(describe):
            try:
                described = describe() or {}
            except Exception as error:  # a description must never stop a training run
                logger.debug(f"Could not describe the data module: {error}")

        if not self.data_config and described:
            self.data_config = described

        if not self.model_config.get("name"):
            name = getattr(type(self.model), "_registry_name", None)
            if name is not None:
                self.model_config["name"] = name
        if self.model_config.get("input_channels") is None and described.get("input_channels"):
            self.model_config["input_channels"] = described["input_channels"]
        if self.model_config.get("output_size") is None:
            from deepecgkit.models.factory import infer_output_size

            output_size = infer_output_size(
                self.model.state_dict(), getattr(self.model, "HEAD_ATTR", None)
            )
            if output_size is not None:
                self.model_config["output_size"] = output_size

    def restore_best_checkpoint(self) -> bool:
        """Reload the weights of the best-validation-loss checkpoint written during fit.

        Returns:
            True if a checkpoint was restored, False if none was available.
        """
        if self.best_checkpoint_path is None or not os.path.exists(self.best_checkpoint_path):
            if self.best_checkpoint_path is not None:
                warnings.warn(
                    f"Best checkpoint {self.best_checkpoint_path} is missing; "
                    "evaluation will use final-epoch weights.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return False

        checkpoint = torch.load(self.best_checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        return True

    def test(self, data_module: Any) -> dict[str, float]:
        """Evaluate the model on the test set.

        Args:
            data_module: ECGDataModule (or any object with test_dataloader)

        Returns:
            Dict with test_loss and test_acc (if classification)
        """
        if hasattr(data_module, "setup"):
            data_module.setup(stage="test")

        test_loader = data_module.test_dataloader()
        return self._evaluate_loader(test_loader)

    def validate(self, data_module: Any) -> dict[str, float]:
        """Evaluate the model on the validation set.

        Args:
            data_module: ECGDataModule (or any object with val_dataloader)

        Returns:
            Dict with val_loss and val_acc (if classification)
        """
        if hasattr(data_module, "setup"):
            data_module.setup(stage="validate")

        val_loader = data_module.val_dataloader()
        return self._evaluate_loader(val_loader, prefix="val")

    def _evaluate_loader(self, dataloader: DataLoader, prefix: str = "test") -> dict[str, float]:
        self.model.eval()
        self.test_predictions = []
        self.test_targets = []
        self.test_probabilities = []
        self.test_logits = []

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                loss = self._calculate_loss(y_hat, y)
                total_loss += loss.item()
                self.test_logits.append(y_hat.detach().cpu())

                if self._is_classification:
                    if self.multi_label:
                        probs = torch.sigmoid(y_hat)
                        preds = (probs > 0.5).long()
                        acc = (preds == y.long()).float().mean().item()
                        self.test_predictions.append(preds.cpu())
                        self.test_targets.append(y.long().cpu())
                        self.test_probabilities.append(probs.cpu())
                    elif self._is_binary:
                        probs_pos = torch.sigmoid(y_hat.squeeze(-1))
                        probs = torch.stack([1 - probs_pos, probs_pos], dim=1)
                        preds = (probs_pos > 0.5).long()
                        acc = (preds == y.long()).float().mean().item()
                        self.test_predictions.append(preds.cpu())
                        self.test_targets.append(y.long().cpu())
                        self.test_probabilities.append(probs.cpu())
                    else:
                        probs = torch.softmax(y_hat, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        acc = (preds == y.long()).float().mean().item()
                        self.test_predictions.append(preds.cpu())
                        self.test_targets.append(y.long().cpu())
                        self.test_probabilities.append(probs.cpu())
                    total_acc += acc

                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        results = {f"{prefix}_loss": avg_loss}
        if self._is_classification:
            results[f"{prefix}_acc"] = total_acc / max(n_batches, 1)
        return results

    def get_test_results(self) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Get test predictions, targets, and probabilities as numpy arrays.

        Returns:
            Tuple of (predictions, targets, probabilities) as numpy arrays,
            or (None, None, None) if no test results available.
        """
        if not self.test_predictions:
            return None, None, None
        return (
            torch.cat(self.test_predictions).numpy(),
            torch.cat(self.test_targets).numpy(),
            torch.cat(self.test_probabilities).numpy(),
        )

    def get_test_logits(self) -> np.ndarray | None:
        """Get the raw model outputs from the most recent evaluation pass.

        Logits are kept alongside probabilities because temperature scaling
        needs them and the softmax and sigmoid that produce the probabilities
        are not losslessly invertible in float32: a probability saturates to
        exactly 1.0 once its logit passes about 16.6, which is well inside the
        range an overconfident model reaches.

        Returns:
            Raw outputs as a numpy array, shaped (n_samples, output_size), or
            None when no evaluation has been run.
        """
        if not self.test_logits:
            return None
        return torch.cat(self.test_logits).numpy()

    def save_checkpoint(self, path: str | Path, epoch: int | None = None) -> None:
        """Save a checkpoint.

        Args:
            path: File path to save to
            epoch: Current epoch number (optional)
        """
        checkpoint: dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "train_config": self.train_config,
            "model_config": self.model_config,
            "data_config": self.data_config,
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "deepecgkit_version": _package_version(),
        }
        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        model: nn.Module | None = None,
        device: str | torch.device = "auto",
    ) -> "ECGTrainer":
        """Load a trainer from a checkpoint.

        Args:
            path: Path to checkpoint file
            model: Model instance to load weights into. When omitted, the
                architecture recorded in the checkpoint is rebuilt.
            device: Device to load onto

        Returns:
            ECGTrainer instance with loaded weights

        Raises:
            ValueError: If no model is given and the checkpoint does not
                record which architecture it holds.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        if model is None:
            from deepecgkit.models.factory import build_model_from_checkpoint

            model, _ = build_model_from_checkpoint(path)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        trainer = cls(
            model=model,
            train_config=checkpoint.get("train_config"),
            device=device,
        )
        trainer.model_config = dict(checkpoint.get("model_config") or {})
        trainer.data_config = dict(checkpoint.get("data_config") or {})
        trainer.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        return trainer

    @staticmethod
    def seed_everything(seed: int) -> None:
        """Set random seeds for reproducibility.

        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class _NullContext:
    """Minimal no-op context manager for Python 3.8 compat."""

    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *args: object) -> None:
        pass
