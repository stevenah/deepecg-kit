"""Named training recipes.

A recipe fixes a complete training and evaluation protocol under one name, so
that a published protocol is reproduced by naming it rather than by restating a
dozen settings. ``deepecg train --recipe NAME`` applies one. A setting passed
explicitly next to it still wins, and the run records it as a deviation.

Without a recipe, ``deepecg train`` uses the package's single default recipe
(see deepecgkit.cli.defaults), which every architecture and task shares.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from torch import nn

#: Settings a recipe fixes that the train command also takes as flags.
RECIPE_FLAGS = (
    "epochs",
    "batch_size",
    "learning_rate",
    "sampling_rate",
    "normalization",
    "early_stopping_patience",
)

#: How the outputs of overlapping windows are combined into one per recording.
WINDOW_AGGREGATES = ("max", "mean")


@dataclass(frozen=True)
class Recipe:
    """A complete training and evaluation protocol.

    Attributes:
        name: Registered name, as passed to --recipe
        description: One line saying what the recipe reproduces
        reference: Where the protocol is published
        datasets: The datasets the recipe is defined for
        multi_label: Whether the recipe trains one sigmoid output per label
        sampling_rate: Rate the signals are resampled to (Hz)
        normalization: "global" fits one mean and one standard deviation over
            every value of the training partition and applies them to all
            partitions; any other value is the dataset's per-sample option
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Peak learning rate
        optimizer: The trainer's optimizer settings (see ECGTrainer)
        scheduler: The trainer's learning-rate schedule (see ECGTrainer)
        crop_seconds: Length of the random crop every training sample is cut
            to, drawn anew each time the sample is used
        window: Inference on overlapping windows: length and stride in
            seconds, and how the window outputs are combined
        class_weights: Whether the multi-label loss weights each label's
            positives by its ratio of negatives to positives
        keep: "last" keeps the weights of the final epoch; "best" keeps the
            lowest validation loss and stops early
        init: "kaiming-normal" re-initialises every convolution and linear
            layer from a Kaiming normal distribution with zero bias, and every
            batch norm layer to unit scale and zero shift; "default" keeps
            PyTorch's initialisation
        model_overrides: Settings that differ by architecture, keyed by
            registered model name: learning_rate and gradient_clip_val
    """

    name: str
    description: str
    reference: str
    datasets: tuple[str, ...]
    multi_label: bool
    sampling_rate: int
    normalization: str
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]
    crop_seconds: float | None
    window: Mapping[str, Any] | None
    class_weights: bool
    keep: str
    init: str
    model_overrides: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def settings_for(self, model_name: str) -> dict[str, Any]:
        """The recipe's flag values for one architecture, overrides applied."""
        settings = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "sampling_rate": self.sampling_rate,
            "normalization": self.normalization,
        }
        override = self.model_overrides.get(model_name, {})
        if "learning_rate" in override:
            settings["learning_rate"] = override["learning_rate"]
        return settings

    def gradient_clip_val(self, model_name: str) -> float | None:
        """Gradient-norm clipping for one architecture, None for none."""
        return self.model_overrides.get(model_name, {}).get("gradient_clip_val")

    def inference_window(self, sampling_rate: float) -> dict[str, Any] | None:
        """The inference window in samples at the given rate, as ECGTrainer takes it."""
        if self.window is None:
            return None
        return {
            "length": round(self.window["length_seconds"] * sampling_rate),
            "stride": round(self.window["stride_seconds"] * sampling_rate),
            "aggregate": self.window["aggregate"],
        }

    def optimizer_config(self) -> dict[str, Any]:
        """The optimizer settings as plain data, for train_config and checkpoints."""
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in self.optimizer.items()
        }

    def describe(self) -> dict[str, Any]:
        """The recipe as plain data, for run metadata."""
        return {
            "name": self.name,
            "reference": self.reference,
            "sampling_rate": self.sampling_rate,
            "normalization": self.normalization,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer_config(),
            "scheduler": dict(self.scheduler),
            "crop_seconds": self.crop_seconds,
            "window": dict(self.window) if self.window else None,
            "class_weights": self.class_weights,
            "keep": self.keep,
            "init": self.init,
            "model_overrides": {name: dict(value) for name, value in self.model_overrides.items()},
        }


#: The fastai protocol of the PTB-XL benchmark (Strodthoff et al. 2021), read
#: from helme/ecg_ptbxl_benchmarking: models/fastai_model.py and
#: utils/utils.py, with fastai 1.0.61's fit_one_cycle defaults. The benchmark
#: also drops the last incomplete training batch, which this recipe keeps.
PTBXL_BENCHMARK = Recipe(
    name="ptbxl-benchmark",
    description="The training and evaluation protocol of the PTB-XL benchmark",
    reference=(
        "Strodthoff N, Wagner P, Schaeffter T, Samek W. Deep learning for ECG analysis: "
        "benchmarks and insights from PTB-XL. IEEE JBHI 25(5):1519-1528, 2021"
    ),
    datasets=("ptbxl",),
    multi_label=True,
    sampling_rate=100,
    normalization="global",
    epochs=50,
    batch_size=128,
    learning_rate=1e-2,
    optimizer=MappingProxyType({"name": "adamw", "betas": (0.9, 0.99), "weight_decay": 1e-2}),
    scheduler=MappingProxyType(
        {
            "name": "one-cycle",
            "pct_start": 0.3,
            "div_factor": 25.0,
            "final_div_factor": 1e4,
            "base_momentum": 0.85,
            "max_momentum": 0.95,
        }
    ),
    crop_seconds=2.5,
    window=MappingProxyType({"length_seconds": 2.5, "stride_seconds": 1.25, "aggregate": "max"}),
    class_weights=False,
    keep="last",
    init="kaiming-normal",
    model_overrides=MappingProxyType(
        {
            "lstm": MappingProxyType({"learning_rate": 1e-3, "gradient_clip_val": 0.25}),
            "gru": MappingProxyType({"learning_rate": 1e-3, "gradient_clip_val": 0.25}),
        }
    ),
)

RECIPES: Mapping[str, Recipe] = MappingProxyType({PTBXL_BENCHMARK.name: PTBXL_BENCHMARK})


def get_recipe_names() -> list[str]:
    """Names of the registered recipes."""
    return sorted(RECIPES)


def get_recipe(name: str) -> Recipe:
    """Look a recipe up by name.

    Raises:
        KeyError: If no recipe has that name.
    """
    try:
        return RECIPES[name]
    except KeyError:
        raise KeyError(
            f"Unknown recipe '{name}'. Available recipes: {', '.join(get_recipe_names())}"
        ) from None


def kaiming_normal_init(model: nn.Module) -> None:
    """Re-initialise a model the way the PTB-XL benchmark does before training.

    Every convolution and linear layer gets Kaiming normal weights and zero
    bias, and every batch norm layer unit scale and zero shift. Recurrent and
    attention layers keep their own initialisation.
    """
    for module in model.modules():
        if isinstance(module, nn.Conv1d | nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d) and module.affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
