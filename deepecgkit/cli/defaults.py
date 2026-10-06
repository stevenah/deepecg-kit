"""Single source of the CLI's default training/evaluation values.

Everything that documents or replays a training run — the argparse defaults,
scripts/run_experiments.py, examples, and the docs tables — reads these
constants instead of restating the numbers.
"""

from types import MappingProxyType

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
#: The built-in datasets hold their signals in memory, so loader workers only
#: add start-up cost (each one receives a copy of the dataset under the spawn
#: start method macOS and Windows use).
NUM_WORKERS = 0
ACCELERATOR = "auto"
DEVICES = 1
EARLY_STOPPING_PATIENCE = 10
SEED = 42

#: Accelerator vocabulary accepted by every command (resolved by
#: deepecgkit.utils.device.resolve_device).
ACCELERATOR_CHOICES = ("auto", "cpu", "gpu", "mps")

#: Datasets that accept the binary_classification flag.
AF_DATASETS = frozenset({"mitbih-afdb", "ltafdb", "unified-af"})

#: The `deepecg train` defaults as one mapping, for batch runners.
TRAIN_DEFAULTS = MappingProxyType(
    {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "val_split": VAL_SPLIT,
        "test_split": TEST_SPLIT,
        "num_workers": NUM_WORKERS,
        "accelerator": ACCELERATOR,
        "devices": DEVICES,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "seed": SEED,
    }
)
