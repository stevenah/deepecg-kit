"""Training recipes and the pieces they are built from.

The ptbxl-benchmark recipe reproduces the PTB-XL benchmark's protocol: global
standardisation fitted on the training folds, random training crops, AdamW
under a one-cycle schedule, the final epoch kept, and the maximum over
overlapping windows at test time.
"""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.optim.optimizer import register_optimizer_step_pre_hook

from deepecgkit.cli import main
from deepecgkit.cli._common import (
    NORMALIZATION_STATS_KEY,
    checkpoint_dataset_kwargs,
    dataset_kwargs_for,
)
from deepecgkit.cli.commands.train import _resolve_recipe
from deepecgkit.cli.logger import CLILogger
from deepecgkit.datasets import ECGDataModule
from deepecgkit.datasets.io import prepare_recording
from deepecgkit.datasets.modules import FIT_STATS, RandomCrop
from deepecgkit.datasets.preprocessing import ECGStandardizer
from deepecgkit.training import ECGTrainer
from deepecgkit.training.recipes import (
    PTBXL_BENCHMARK,
    get_recipe,
    get_recipe_names,
    kaiming_normal_init,
)
from deepecgkit.utils.checkpoints import load_checkpoint_file
from tests.test_ptbxl_protocol import STATEMENTS, _write_corpus

#: The synthetic corpus's statements plus one for each superclass it lacks.
ALL_SUPERCLASSES = pd.concat(
    [
        STATEMENTS,
        pd.DataFrame(
            {"diagnostic": [1.0, 1.0], "diagnostic_class": ["STTC", "HYP"]},
            index=["NDT", "LVH"],
        ),
    ]
)


class Signals(torch.utils.data.Dataset):
    """Multi-label signals with a transform attribute, as the built-in datasets have."""

    multi_label = True

    def __init__(self, x, y):
        self.x, self.y = torch.as_tensor(x), torch.as_tensor(y)
        self.transform = None
        self.group_ids = [f"r{i}" for i in range(len(self.x))]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        signal = self.x[index]
        if self.transform is not None:
            signal = self.transform(signal)
        return signal, self.y[index]


class WindowMean(nn.Module):
    """Scores a window by the mean of its first lead, so window outputs can be computed by hand."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return (x[:, 0].mean(dim=-1, keepdim=True) * self.scale).repeat(1, 2)


def _signals(n=40, length=100, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2, length)).astype(np.float32)
    y = (rng.random((n, 2)) < 0.5).astype(np.float32)
    return x, y


class TestRegistry:
    def test_the_benchmark_recipe_is_registered(self):
        assert get_recipe_names() == ["ptbxl-benchmark"]
        assert get_recipe("ptbxl-benchmark") is PTBXL_BENCHMARK

    def test_an_unknown_recipe_names_the_known_ones(self):
        with pytest.raises(KeyError, match="ptbxl-benchmark"):
            get_recipe("nope")

    def test_recurrent_models_get_their_own_rate_and_clipping(self):
        assert PTBXL_BENCHMARK.settings_for("fcn-wang")["learning_rate"] == 1e-2
        assert PTBXL_BENCHMARK.settings_for("lstm")["learning_rate"] == 1e-3
        assert PTBXL_BENCHMARK.gradient_clip_val("lstm") == 0.25
        assert PTBXL_BENCHMARK.gradient_clip_val("fcn-wang") is None

    def test_the_window_is_given_in_samples(self):
        assert PTBXL_BENCHMARK.inference_window(100) == {
            "length": 250,
            "stride": 125,
            "aggregate": "max",
        }

    def test_the_description_is_plain_data(self):
        json.dumps(PTBXL_BENCHMARK.describe())
        assert PTBXL_BENCHMARK.optimizer_config()["betas"] == [0.9, 0.99]


class TestInitialisation:
    def test_convolutions_and_linear_layers_get_zero_bias_and_norms_unit_scale(self):
        model = nn.Sequential(nn.Conv1d(2, 4, 3), nn.BatchNorm1d(4), nn.Linear(4, 2))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.5)
        kaiming_normal_init(model)
        assert torch.all(model[0].bias == 0) and torch.all(model[2].bias == 0)
        assert torch.all(model[1].weight == 1) and torch.all(model[1].bias == 0)
        assert not torch.all(model[0].weight == 0.5)

    def test_recurrent_layers_keep_their_own_initialisation(self):
        model = nn.LSTM(2, 3)
        before = [parameter.clone() for parameter in model.parameters()]
        kaiming_normal_init(model)
        assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


class TestWindowedInference:
    def test_window_outputs_are_combined_by_their_maximum(self):
        trainer = ECGTrainer(
            WindowMean(),
            {"multi_label": True, "inference_window": {"length": 4, "stride": 2}},
            device="cpu",
        )
        x = torch.tensor([[[0.0, 0, 4, 4, 0, 0, 8, 8, 9]], [[1.0, 1, 1, 1, 1, 1, 1, 1, 1]]])
        x = x.repeat(1, 2, 1)
        # Windows start at 0, 2 and 4; the ninth sample is in no whole window.
        expected = torch.tensor([[4.0, 4.0], [1.0, 1.0]])
        assert torch.allclose(trainer._infer(x), expected)

    def test_the_mean_is_available_too(self):
        trainer = ECGTrainer(
            WindowMean(),
            {"inference_window": {"length": 4, "stride": 2, "aggregate": "mean"}},
            device="cpu",
        )
        x = torch.tensor([[[0.0, 0, 4, 4, 0, 0, 8, 8]]]).repeat(1, 2, 1)
        assert torch.allclose(trainer._infer(x), torch.tensor([[8.0 / 3, 8.0 / 3]]))

    def test_inputs_no_longer_than_the_window_are_scored_whole(self):
        trainer = ECGTrainer(
            WindowMean(), {"inference_window": {"length": 8, "stride": 4}}, device="cpu"
        )
        x = torch.arange(8.0).reshape(1, 1, 8).repeat(1, 2, 1)
        assert torch.allclose(trainer._infer(x), torch.full((1, 2), 3.5))


class TestOptimisation:
    def _fit(self, tmp_path, train_config, keep="best", epochs=3, patience=10):
        x, y = _signals()
        ECGTrainer.seed_everything(0)
        data = ECGDataModule(dataset=Signals(x, y), batch_size=8, verbose=False)
        model = nn.Sequential(
            nn.Conv1d(2, 4, 5), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(4, 2)
        )
        trainer = ECGTrainer(model, {"multi_label": True, **train_config}, device="cpu")
        trainer.fit(
            data,
            epochs=epochs,
            early_stopping_patience=patience,
            checkpoint_dir=tmp_path / "checkpoints",
            progress_bar=False,
            keep=keep,
        )
        return trainer, data

    def test_adamw_takes_the_given_betas_and_decoupled_weight_decay(self, tmp_path):
        trainer, _ = self._fit(
            tmp_path, {"optimizer": {"name": "adamw", "betas": [0.9, 0.99], "weight_decay": 0.01}}
        )
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        group = trainer.optimizer.param_groups[0]
        assert group["weight_decay"] == 0.01 and group["betas"][1] == 0.99

    def test_a_one_cycle_schedule_steps_every_batch_and_ends_low(self, tmp_path):
        config = {
            "learning_rate": 1e-2,
            "scheduler": {"name": "one-cycle", "div_factor": 25.0, "final_div_factor": 1e4},
        }
        used = []
        hook = register_optimizer_step_pre_hook(
            lambda optimizer, args, kwargs: used.append(optimizer.param_groups[0]["lr"])
        )
        try:
            _, data = self._fit(tmp_path, config)
        finally:
            hook.remove()
        assert len(used) == 3 * len(data.train_dataloader())
        assert used[0] == pytest.approx(1e-2 / 25)
        assert max(used) == pytest.approx(1e-2, rel=0.05)
        assert used[-1] == pytest.approx(1e-2 / 25e4)

    def test_keeping_the_last_epoch_trains_every_epoch_and_saves_one_checkpoint(self, tmp_path):
        trainer, _ = self._fit(tmp_path, {"learning_rate": 1.0}, keep="last", epochs=4, patience=1)
        assert [p.name for p in (tmp_path / "checkpoints").iterdir()] == ["epoch=04-last.pt"]
        assert trainer.best_checkpoint_path.endswith("epoch=04-last.pt")
        saved = load_checkpoint_file(trainer.best_checkpoint_path)
        for name, value in trainer.model.state_dict().items():
            assert torch.equal(saved["model_state_dict"][name].to(value.device), value)

    @pytest.mark.parametrize(
        ("config", "keep"),
        [
            ({"optimizer": {"name": "sgd"}}, "best"),
            ({"scheduler": {"name": "cosine"}}, "best"),
            ({}, "worst"),
        ],
    )
    def test_unknown_settings_are_rejected(self, tmp_path, config, keep):
        with pytest.raises(ValueError):
            self._fit(tmp_path, config, keep=keep, epochs=1)


class TestGlobalStandardisation:
    def test_statistics_come_from_the_training_partition_only(self):
        x, y = _signals(n=60)
        data = ECGDataModule(
            dataset=Signals(x, y), normalization_stats=FIT_STATS, seed=1, verbose=False
        )
        data.setup()
        train = x[data.train_dataset.indices]
        mean, std = data.normalization_stats
        assert mean == pytest.approx(train.astype(np.float64).mean())
        assert std == pytest.approx(train.astype(np.float64).std())
        test_index = data.test_dataset.indices[0]
        assert torch.allclose(
            data.dataset[test_index][0], (torch.as_tensor(x[test_index]) - mean) / std
        )

    def test_given_statistics_are_applied_as_they_are(self):
        x, y = _signals()
        data = ECGDataModule(dataset=Signals(x, y), normalization_stats=(1.0, 2.0), verbose=False)
        data.setup()
        assert torch.allclose(data.dataset[0][0], (torch.as_tensor(x[0]) - 1.0) / 2.0)

    def test_the_description_records_the_statistics(self):
        x, y = _signals()
        data = ECGDataModule(dataset=Signals(x, y), normalization_stats=(1.0, 2.0), verbose=False)
        data.setup()
        preprocessing = data.describe()["preprocessing"]
        assert preprocessing["normalization"] == "global"
        assert preprocessing["normalization_stats"] == [1.0, 2.0]

    def test_a_second_setup_does_not_standardise_twice(self):
        x, y = _signals()
        data = ECGDataModule(dataset=Signals(x, y), normalization_stats=(1.0, 2.0), verbose=False)
        data.setup()
        data.val_split = 0.3
        data.setup()
        assert torch.allclose(data.dataset[0][0], (torch.as_tensor(x[0]) - 1.0) / 2.0)

    def test_the_standardiser_needs_statistics(self):
        with pytest.raises(ValueError, match="normalization_stats"):
            ECGStandardizer(normalization="global")
        standardizer = ECGStandardizer(normalization="global", normalization_stats=(1.0, 2.0))
        assert np.allclose(standardizer.normalize(np.array([3.0, 5.0])), [1.0, 2.0])

    def test_a_raw_recording_gets_the_recorded_statistics(self):
        preprocessing = {
            "sampling_rate": 100,
            "normalization": "global",
            "normalization_stats": [1.0, 2.0],
        }
        prepared = prepare_recording(np.full((2, 100), 5.0), preprocessing, 100)
        assert np.allclose(prepared.segments, 2.0)


class TestCrops:
    def test_training_samples_are_cropped_and_the_others_left_whole(self):
        x, y = _signals(length=100)
        data = ECGDataModule(dataset=Signals(x, y), train_crop_length=30, verbose=False)
        data.setup()
        assert next(iter(data.train_dataloader()))[0].shape[-1] == 30
        assert next(iter(data.val_dataloader()))[0].shape[-1] == 100

    def test_crops_are_redrawn_each_time(self):
        signal = torch.arange(100.0).reshape(1, 100)
        crop = RandomCrop([(signal, 0)], 10)
        torch.manual_seed(0)
        starts = {float(crop[0][0][0, 0]) for _ in range(20)}
        assert len(starts) > 1 and all(0 <= start <= 90 for start in starts)


class TestCommandLine:
    def test_the_package_defaults_apply_without_a_recipe(self):
        given = dict.fromkeys(
            ("epochs", "batch_size", "learning_rate", "sampling_rate", "normalization")
        )
        settings, deviations = _resolve_recipe(None, "fcn-wang", given, CLILogger(quiet=True))
        assert (settings["epochs"], settings["batch_size"], settings["learning_rate"]) == (
            50,
            32,
            1e-3,
        )
        assert settings["sampling_rate"] is None and deviations == {}

    def test_explicit_flags_win_and_are_recorded_as_deviations(self):
        given = {
            "epochs": 2,
            "batch_size": None,
            "learning_rate": None,
            "sampling_rate": 100,
            "normalization": None,
        }
        settings, deviations = _resolve_recipe(
            PTBXL_BENCHMARK, "lstm", given, CLILogger(quiet=True)
        )
        assert settings["epochs"] == 2 and settings["learning_rate"] == 1e-3
        assert settings["normalization"] == "global" and deviations == {"epochs": 2}

    def test_global_normalisation_is_fitted_by_the_data_module(self):
        kwargs = dataset_kwargs_for("ptbxl", normalization="global")
        assert kwargs["normalization"] == "none"
        assert kwargs[NORMALIZATION_STATS_KEY] == FIT_STATS

    def test_a_checkpoint_hands_back_its_statistics(self):
        stored = {
            "train_config": {"multi_label": True},
            "data_config": {
                "dataset": "ptbxl",
                "preprocessing": {
                    "sampling_rate": 100,
                    "normalization": "global",
                    "normalization_stats": [0.1, 0.2],
                },
            },
        }
        kwargs, _ = checkpoint_dataset_kwargs("ptbxl", stored, CLILogger(quiet=True))
        assert kwargs["normalization"] == "none"
        assert kwargs[NORMALIZATION_STATS_KEY] == (0.1, 0.2)

    def test_the_recipe_is_refused_for_other_datasets(self, capsys):
        result = main(
            ["train", "-m", "fcn-wang", "-d", "af-classification", "--recipe", "ptbxl-benchmark"]
        )
        assert result == 1
        assert "defined for ptbxl" in capsys.readouterr().err


@pytest.fixture(scope="module")
def recipe_run(tmp_path_factory):
    """A two-epoch ptbxl-benchmark run on a synthetic PTB-XL corpus, then its evaluation."""
    root = tmp_path_factory.mktemp("recipe")
    codes = [{"NORM": 100.0}, {"IMI": 100.0, "LAFB": 100.0}, {"NDT": 100.0}, {"LVH": 100.0}]
    codes += [{"IMI": 100.0, "NDT": 50.0}]
    _write_corpus(root / "data", [(code, fold) for fold in range(1, 11) for code in codes])
    ALL_SUPERCLASSES.to_csv(root / "data" / "scp_statements.csv")
    run = root / "run"
    trained = main(
        [
            "train",
            "-m",
            "fcn-wang",
            "-d",
            "ptbxl",
            "--data-dir",
            str(root / "data"),
            "--recipe",
            "ptbxl-benchmark",
            "--epochs",
            "2",
            "--accelerator",
            "cpu",
            "-o",
            str(run),
            "--quiet",
        ]
    )
    evaluated = main(
        [
            "evaluate",
            "--checkpoint",
            str(run),
            "-d",
            "ptbxl",
            "--data-dir",
            str(root / "data"),
            "--accelerator",
            "cpu",
            "-o",
            str(root / "evaluation"),
            "--quiet",
        ]
    )
    return {
        "trained": trained,
        "evaluated": evaluated,
        "run": run,
        "evaluation": root / "evaluation",
    }


class TestRecipeRun:
    def test_the_run_records_the_recipe_and_its_deviations(self, recipe_run):
        assert recipe_run["trained"] == 0
        metadata = json.loads((recipe_run["run"] / "run_metadata.json").read_text())
        assert metadata["recipe"]["name"] == "ptbxl-benchmark"
        assert metadata["recipe_deviations"] == {"epochs": 2}
        assert (metadata["sampling_rate"], metadata["normalization"]) == (100, "global")
        assert (metadata["batch_size"], metadata["learning_rate"]) == (128, 1e-2)
        assert metadata["multi_label"] is True

    def test_the_checkpoint_carries_everything_evaluation_needs(self, recipe_run):
        checkpoints = list((recipe_run["run"] / "checkpoints").iterdir())
        assert [path.name for path in checkpoints] == ["epoch=02-last.pt"]
        stored = load_checkpoint_file(checkpoints[0])
        config = stored["train_config"]
        assert config["inference_window"] == {"length": 250, "stride": 125, "aggregate": "max"}
        assert config["optimizer"]["name"] == "adamw" and config["pos_weight"] is None
        assert config["scheduler"]["name"] == "one-cycle"
        assert len(stored["data_config"]["preprocessing"]["normalization_stats"]) == 2

    def test_evaluation_reproduces_the_scores_of_training(self, recipe_run):
        assert recipe_run["evaluated"] == 0
        trained = pd.read_csv(recipe_run["run"] / "classification_metrics.csv")
        evaluated = pd.read_csv(recipe_run["evaluation"] / "classification_metrics.csv")
        for metric in ("auroc", "f1", "ece", "ece_platt"):
            assert np.isfinite(trained[metric].iloc[0])
            assert evaluated[metric].iloc[0] == pytest.approx(trained[metric].iloc[0])
