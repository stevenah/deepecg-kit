import pytorch_lightning as pl
import torch


class ECGLitModel(pl.LightningModule):
    """
    PyTorch Lightning module for ECG signal classification and regression.

    This module wraps ECG models and provides training, validation, and testing
    functionality with configurable optimizers and learning rate schedulers.

    Args:
        model: The ECG model to train (e.g., KanResWideX, AFModel)
        train_config: Dictionary containing training configuration:
            - learning_rate: Learning rate for optimizer
            - scheduler: Dict with 'factor' and 'patience' for ReduceLROnPlateau
            - binary_classification: Bool, if True uses BCE loss for binary tasks
            - multi_label: Bool, if True uses BCE loss for multi-label tasks
        use_plateau_scheduler: If True, uses ReduceLROnPlateau, else StepLR

    Example:
        >>> model = KanResWideX(input_channels=1, output_size=4)
        >>> config = {
        ...     "learning_rate": 0.001,
        ...     "scheduler": {"factor": 0.5, "patience": 10},
        ...     "binary_classification": False
        ... }
        >>> lit_model = ECGLitModel(model=model, train_config=config)
    """

    def __init__(self, model, train_config, use_plateau_scheduler=True):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model

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

        self.test_predictions = []
        self.test_targets = []
        self.test_probabilities = []

    @property
    def _is_binary(self):
        return isinstance(self.criterion, torch.nn.BCEWithLogitsLoss) and not self.multi_label

    @property
    def _is_classification(self):
        return isinstance(self.criterion, (torch.nn.CrossEntropyLoss, torch.nn.BCEWithLogitsLoss))

    def _calculate_loss(self, y_hat, y):
        if self.multi_label:
            return self.criterion(y_hat, y.float())
        if self._is_binary:
            return self.criterion(y_hat.squeeze(-1), y.float())
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            return self.criterion(y_hat, y.long())
        return self.criterion(y_hat.float(), y.float())

    def _compute_acc(self, y_hat, y):
        if self.multi_label:
            preds = (y_hat > 0).float()
            return (preds == y.float()).float().mean()
        if self._is_binary:
            preds = (y_hat.squeeze(-1) > 0).long()
            return (preds == y.long()).float().mean()
        return (torch.argmax(y_hat, dim=1) == y.long()).float().mean()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._calculate_loss(y_hat, y)
        if hasattr(self.model, "l2_regularization_loss"):
            l2_loss = self.model.l2_regularization_loss()
            loss = loss + l2_loss
        self.log("train_loss", loss, prog_bar=True)
        if self._is_classification:
            self.log("train_acc", self._compute_acc(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._calculate_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        if self._is_classification:
            self.log("val_acc", self._compute_acc(y_hat, y), prog_bar=True)
        return loss

    def on_test_epoch_start(self):
        self.test_predictions = []
        self.test_targets = []
        self.test_probabilities = []

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._calculate_loss(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        if self._is_classification:
            if self.multi_label:
                probs = torch.sigmoid(y_hat)
                preds = (probs > 0.5).long()
                acc = (preds == y.long()).float().mean()
                self.test_predictions.append(preds.cpu())
                self.test_targets.append(y.long().cpu())
                self.test_probabilities.append(probs.cpu())
            elif self._is_binary:
                probs_pos = torch.sigmoid(y_hat.squeeze(-1))
                probs = torch.stack([1 - probs_pos, probs_pos], dim=1)
                preds = (probs_pos > 0.5).long()
                acc = (preds == y.long()).float().mean()
                self.test_predictions.append(preds.cpu())
                self.test_targets.append(y.long().cpu())
                self.test_probabilities.append(probs.cpu())
            else:
                probs = torch.softmax(y_hat, dim=1)
                preds = torch.argmax(probs, dim=1)
                acc = (preds == y.long()).float().mean()
                self.test_predictions.append(preds.cpu())
                self.test_targets.append(y.long().cpu())
                self.test_probabilities.append(probs.cpu())
            self.log("test_acc", acc, prog_bar=True)
        return loss

    def get_test_results(self):
        if not self.test_predictions:
            return None, None, None
        return (
            torch.cat(self.test_predictions).numpy(),
            torch.cat(self.test_targets).numpy(),
            torch.cat(self.test_probabilities).numpy(),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.use_plateau_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            }
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
            }
