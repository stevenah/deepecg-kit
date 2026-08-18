"""
Tests for the scripts.run_experiments experiment driver.
"""

import json

import pytest

from scripts.run_experiments import (
    _load_state,
    _save_state,
    _state_path,
    build_command,
    expand_runs,
    load_experiments,
    main,
    resolve_seeds,
    run_is_complete,
    run_key,
    validate_seeds,
)

SHIPPED_CONFIG = "experiments.yaml"


def write_config(directory, text, name="experiments.yaml"):
    """Write a YAML config into a directory.

    Args:
        directory: Destination directory.
        text: File contents.
        name: File name.

    Returns:
        The path as a string.
    """
    path = directory / name
    path.write_text(text)
    return str(path)


def make_completed_run(directory, model="resnet", dataset="ptbxl", seed=42):
    """Create a run directory that looks successfully finished.

    Args:
        directory: Parent directory.
        model: Model name.
        dataset: Dataset name.
        seed: Seed for this run.

    Returns:
        The created run directory.
    """
    run_dir = directory / f"{model}-{dataset}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "classification_metrics.csv").write_text(
        f"model,dataset,accuracy,precision,recall,f1,auroc,mcc,average\n"
        f"{model},{dataset},0.7,0.7,0.7,0.7,0.9,0.5,macro\n"
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"key": run_key(model, dataset, seed), "seed": seed, "returncode": 0})
    )
    return run_dir


class TestSeedValidation:
    """Test rejection of malformed seed lists."""

    @pytest.mark.parametrize(
        "value,message",
        [
            (42, "must be a list"),
            ([], "must not be empty"),
            ([1, 1], "duplicate seeds"),
            ([True], "must be integers"),
            ([1.5], "must be integers"),
            (["7"], "must be integers"),
            ([-1], "outside"),
        ],
    )
    def test_malformed_seeds_are_rejected(self, value, message):
        """Seeds are validated rather than coerced."""
        with pytest.raises(ValueError, match=message):
            validate_seeds(value, "test")

    def test_valid_seeds_keep_their_order(self):
        """The configured order is preserved, not sorted."""
        assert validate_seeds([44, 42, 43], "test") == [44, 42, 43]


class TestSeedResolution:
    """Test the seed precedence ladder."""

    def test_command_line_beats_everything(self):
        """An explicit override replaces both config levels."""
        experiment = {"model": "resnet", "dataset": "ptbxl", "seeds": [7]}
        assert resolve_seeds(experiment, {"seeds": [1, 2]}, 42, seeds_override=[9]) == [9]

    def test_experiment_beats_defaults(self):
        """A per-experiment list overrides the defaults list."""
        experiment = {"model": "resnet", "dataset": "ptbxl", "seeds": [7]}
        assert resolve_seeds(experiment, {"seeds": [1, 2]}, 42) == [7]

    def test_defaults_apply_when_the_experiment_is_silent(self):
        """The defaults list covers every experiment that does not override it."""
        assert resolve_seeds({"model": "m", "dataset": "d"}, {"seeds": [1, 2]}, 42) == [1, 2]

    def test_scalar_seed_is_used_when_no_list_exists(self):
        """Without a seeds key the single configured seed is used verbatim."""
        assert resolve_seeds({"model": "m", "dataset": "d"}, {}, 7) == [7]

    def test_scalar_seed_is_not_hardcoded_to_42(self, temp_dir):
        """A config pinning another seed expands to that seed alone."""
        config = write_config(
            temp_dir,
            "defaults:\n  seed: 7\nexperiments:\n  - model: resnet\n    dataset: ptbxl\n",
        )
        assert load_experiments(config)[0]["seeds"] == [7]


class TestExpandRuns:
    """Test expansion of experiments into per-seed runs."""

    def test_run_count_is_the_product(self, temp_dir):
        """Two experiments over three seeds give six runs."""
        config = write_config(
            temp_dir,
            "defaults:\n  seeds: [1, 2, 3]\nexperiments:\n"
            "  - model: resnet\n    dataset: ptbxl\n"
            "  - model: crnn\n    dataset: ptbxl\n",
        )
        assert len(expand_runs(load_experiments(config))) == 6

    def test_order_is_experiment_major(self, temp_dir):
        """All seeds of one experiment precede the next experiment."""
        config = write_config(
            temp_dir,
            "defaults:\n  seeds: [1, 2]\nexperiments:\n"
            "  - model: resnet\n    dataset: ptbxl\n"
            "  - model: crnn\n    dataset: ptbxl\n",
        )
        keys = [run["key"] for run in expand_runs(load_experiments(config))]
        assert keys == [
            "resnet|ptbxl|1",
            "resnet|ptbxl|2",
            "crnn|ptbxl|1",
            "crnn|ptbxl|2",
        ]

    def test_seeds_key_is_replaced_by_a_scalar(self, temp_dir):
        """A run mapping is distinguishable from an experiment by shape."""
        config = write_config(
            temp_dir,
            "defaults:\n  seeds: [1, 2]\nexperiments:\n  - model: resnet\n    dataset: ptbxl\n",
        )
        run = expand_runs(load_experiments(config))[0]
        assert "seeds" not in run
        assert run["seed"] == 1

    def test_output_dir_encodes_the_seed(self, temp_dir):
        """Repeat runs of one experiment cannot land in the same directory."""
        config = write_config(
            temp_dir,
            "defaults:\n  seeds: [1, 2]\nexperiments:\n  - model: resnet\n    dataset: ptbxl\n",
        )
        directories = [run["output_dir"] for run in expand_runs(load_experiments(config))]
        assert directories == ["runs/resnet-ptbxl-seed1", "runs/resnet-ptbxl-seed2"]
        assert len(set(directories)) == len(directories)

    def test_explicit_output_dir_survives_a_single_seed(self, temp_dir):
        """An existing single-seed config keeps the path it always used."""
        config = write_config(
            temp_dir,
            "experiments:\n  - model: resnet\n    dataset: ptbxl\n    output_dir: /tmp/mine\n",
        )
        assert expand_runs(load_experiments(config))[0]["output_dir"] == "/tmp/mine"

    def test_explicit_output_dir_gains_a_sibling_suffix_per_seed(self, temp_dir):
        """Several seeds must not overwrite one chosen directory."""
        config = write_config(
            temp_dir,
            "defaults:\n  seeds: [1, 2]\nexperiments:\n"
            "  - model: resnet\n    dataset: ptbxl\n    output_dir: /tmp/mine\n",
        )
        assert [run["output_dir"] for run in expand_runs(load_experiments(config))] == [
            "/tmp/mine-seed1",
            "/tmp/mine-seed2",
        ]

    def test_experiments_differing_only_in_hyperparameters_are_separated(self, temp_dir):
        """Two runs sharing model, dataset and seed still get distinct directories."""
        config = write_config(
            temp_dir,
            "defaults:\n  seeds: [1]\nexperiments:\n"
            "  - model: resnet\n    dataset: ptbxl\n    learning_rate: 0.001\n"
            "  - model: resnet\n    dataset: ptbxl\n    learning_rate: 0.01\n",
        )
        directories = [run["output_dir"] for run in expand_runs(load_experiments(config))]
        assert len(set(directories)) == 2

    def test_only_filter_selects_a_pairing(self, temp_dir):
        """Restricting to one pairing drops the others entirely."""
        config = write_config(
            temp_dir,
            "defaults:\n  seeds: [1, 2]\nexperiments:\n"
            "  - model: resnet\n    dataset: ptbxl\n"
            "  - model: crnn\n    dataset: ptbxl\n",
        )
        runs = expand_runs(load_experiments(config), only=["crnn:ptbxl"])
        assert {run["model"] for run in runs} == {"crnn"}


class TestBackwardsCompatibility:
    """Test that a config without a seeds key behaves as it always did."""

    def test_shipped_config_without_seeds_expands_one_run_each(self, temp_dir):
        """Removing the seeds key restores exactly the previous run count."""
        text = "defaults:\n  epochs: 50\n  seed: 42\nexperiments:\n" + "".join(
            f"  - model: {model}\n    dataset: ptbxl\n"
            for model in ("resnet", "se-resnet", "inception-time")
        )
        config = write_config(temp_dir, text)
        runs = expand_runs(load_experiments(config))
        assert len(runs) == 3
        assert {run["seed"] for run in runs} == {42}

    def test_shipped_config_is_loadable(self):
        """The repository config still parses and expands."""
        experiments = load_experiments(SHIPPED_CONFIG)
        assert len(experiments) == 18

    def test_command_matches_the_previous_form_plus_output_dir(self, temp_dir):
        """Only the output directory is new on the training command."""
        config = write_config(temp_dir, "experiments:\n  - model: resnet\n    dataset: ptbxl\n")
        command = build_command(expand_runs(load_experiments(config))[0])
        assert command[:4] == ["deepecg", "train", "--model", "resnet"]
        assert "--seed" in command
        assert "--output-dir" in command


class TestStateRoundTrip:
    """Test the progress cache."""

    def test_records_survive_a_round_trip(self, temp_dir):
        """What is saved is what is loaded."""
        config = write_config(temp_dir, "experiments:\n  - model: resnet\n    dataset: ptbxl\n")
        records = {
            "resnet|ptbxl|42": {
                "key": "resnet|ptbxl|42",
                "model": "resnet",
                "dataset": "ptbxl",
                "seed": 42,
                "output_dir": "runs/resnet-ptbxl-seed42",
            }
        }
        _save_state(config, records)
        assert _load_state(config) == records

    def test_state_is_keyed_by_model_dataset_and_seed(self, temp_dir):
        """Reordering the config cannot mis-map a completed run."""
        assert run_key("resnet", "ptbxl", 42) == "resnet|ptbxl|42"

    def test_old_format_is_refused_rather_than_guessed(self, temp_dir):
        """Positional indices carry no seed, so no mapping is invented."""
        config = write_config(temp_dir, "experiments:\n  - model: resnet\n    dataset: ptbxl\n")
        _state_path(config).write_text(json.dumps({"completed": [0, 1, 2]}))
        with pytest.raises(ValueError, match="old format"):
            _load_state(config)

    def test_absent_state_is_empty_not_an_error(self, temp_dir):
        """A first run has no progress to load."""
        config = write_config(temp_dir, "experiments:\n  - model: resnet\n    dataset: ptbxl\n")
        assert _load_state(config) == {}


class TestRunIsComplete:
    """Test the on-disk completion contract."""

    def test_metrics_and_a_zero_exit_mean_complete(self, temp_dir):
        """A finished run has both artefacts."""
        assert run_is_complete(str(make_completed_run(temp_dir))) is True

    def test_missing_metrics_means_incomplete(self, temp_dir):
        """A run that crashed before evaluation is not complete."""
        run_dir = make_completed_run(temp_dir)
        (run_dir / "classification_metrics.csv").unlink()
        assert run_is_complete(str(run_dir)) is False

    def test_non_zero_exit_means_incomplete(self, temp_dir):
        """Stale metrics from an earlier attempt do not count."""
        run_dir = make_completed_run(temp_dir)
        (run_dir / "run_metadata.json").write_text(json.dumps({"returncode": 1}))
        assert run_is_complete(str(run_dir)) is False

    def test_absent_directory_is_incomplete(self, temp_dir):
        """Nothing on disk means nothing was done."""
        assert run_is_complete(str(temp_dir / "never-ran")) is False


class TestMainDryRun:
    """Test the command-line entry point without training."""

    def test_dry_run_lists_every_expanded_command(self, temp_dir, capsys):
        """Expansion is visible before anything is launched."""
        config = write_config(
            temp_dir,
            "defaults:\n  seeds: [42, 43]\nexperiments:\n"
            "  - model: resnet\n    dataset: af-classification\n"
            "  - model: crnn\n    dataset: af-classification\n",
        )
        assert main(["--config", config, "--runs-dir", str(temp_dir / "runs"), "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "4 runs loaded" in out
        assert out.count("deepecg train") == 4

    def test_dry_run_echoes_the_resolved_seeds(self, temp_dir, capsys):
        """The echo is what makes a YAML octal literal visible."""
        config = write_config(
            temp_dir,
            "defaults:\n  seeds: [042, 043]\nexperiments:\n  - model: resnet\n    dataset: ptbxl\n",
        )
        main(["--config", config, "--dry-run"])
        assert "Resolved seeds: [34, 35]" in capsys.readouterr().out

    def test_dry_run_writes_nothing(self, temp_dir):
        """No run directory appears before training starts."""
        config = write_config(
            temp_dir,
            "defaults:\n  seeds: [1, 2]\nexperiments:\n  - model: resnet\n    dataset: ptbxl\n",
        )
        runs_dir = temp_dir / "runs"
        main(["--config", config, "--runs-dir", str(runs_dir), "--dry-run"])
        assert not runs_dir.exists()

    def test_invalid_seeds_exit_non_zero(self, temp_dir, capsys):
        """A malformed config fails before any subprocess is launched."""
        config = write_config(
            temp_dir,
            "defaults:\n  seeds: 42\nexperiments:\n  - model: resnet\n    dataset: ptbxl\n",
        )
        assert main(["--config", config, "--dry-run"]) == 1
        assert "must be a list" in capsys.readouterr().err

    def test_resume_marks_finished_runs_as_skipped(self, temp_dir, capsys):
        """A completed directory is recognised without consulting the cache."""
        config = write_config(
            temp_dir,
            "defaults:\n  seeds: [42, 43]\nexperiments:\n  - model: resnet\n    dataset: ptbxl\n",
        )
        runs_dir = temp_dir / "runs"
        make_completed_run(runs_dir, "resnet", "ptbxl", 42)
        main(["--config", config, "--runs-dir", str(runs_dir), "--resume", "--dry-run"])
        out = capsys.readouterr().out
        assert out.count("[SKIP]") == 1
        assert "1 already done, 1 remaining" in out

    def test_reset_removes_saved_progress(self, temp_dir, capsys):
        """Clearing progress is explicit and reports what it did."""
        config = write_config(temp_dir, "experiments:\n  - model: resnet\n    dataset: ptbxl\n")
        _save_state(config, {"resnet|ptbxl|42": {"key": "resnet|ptbxl|42"}})
        assert main(["--config", config, "--reset"]) == 0
        assert not _state_path(config).exists()
