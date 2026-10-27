"""PTB-XL label rule and recommended-fold split.

A small synthetic PTB-XL directory (wfdb records plus the two metadata
tables) stands in for the 3 GB corpus.
"""

import numpy as np
import pandas as pd
import pytest
import wfdb

from deepecgkit.cli._common import checkpoint_dataset_kwargs
from deepecgkit.cli.logger import CLILogger
from deepecgkit.datasets import ECGDataModule, PTBXLDataset

STATEMENTS = pd.DataFrame(
    {
        "diagnostic": [1.0, 1.0, 1.0, np.nan],
        "diagnostic_class": ["NORM", "MI", "CD", np.nan],
    },
    index=["NORM", "IMI", "LAFB", "SR"],
)


def _write_corpus(root, records):
    """Write one 12-lead 10 s record per (scp_codes, strat_fold) pair."""
    rows = []
    rng = np.random.default_rng(0)
    for ecg_id, (scp_codes, fold) in enumerate(records, start=1):
        name = f"{ecg_id:05d}_hr"
        directory = root / "records500" / "00000"
        directory.mkdir(parents=True, exist_ok=True)
        signal = rng.standard_normal((5000, 12)) * 0.1
        wfdb.wrsamp(
            name,
            fs=500,
            units=["mV"] * 12,
            sig_name=PTBXLDataset.LEADS,
            p_signal=signal,
            fmt=["16"] * 12,
            write_dir=str(directory),
        )
        rows.append(
            {
                "ecg_id": ecg_id,
                "patient_id": float(100 + ecg_id),
                "scp_codes": repr(scp_codes),
                "strat_fold": fold,
                "filename_lr": f"records100/00000/{ecg_id:05d}_lr",
                "filename_hr": f"records500/00000/{name}",
            }
        )
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    STATEMENTS.to_csv(root / "scp_statements.csv")


@pytest.fixture
def corpus(tmp_path):
    """Two records per fold; fold 1 also holds a low-likelihood and an unlabelled record."""
    records = [({"NORM": 100.0, "SR": 0.0}, fold) for fold in range(1, 11)]
    records += [({"IMI": 100.0, "LAFB": 100.0}, fold) for fold in range(1, 11)]
    records += [({"IMI": 15.0}, 1), ({"SR": 0.0}, 1)]
    _write_corpus(tmp_path, records)
    return tmp_path


def _load(root, **kwargs):
    return PTBXLDataset(data_dir=root, verbose=False, **kwargs)


class TestLabelRule:
    def test_benchmark_rule_counts_every_statement_and_drops_unlabelled(self, corpus):
        dataset = _load(corpus)
        assert len(dataset) == 21
        assert dataset.unlabelled_records == 1
        low_likelihood = dataset.labels[dataset.record_names.index("21")]
        assert low_likelihood.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]

    def test_likelihood_rule_keeps_unlabelled_records_all_negative(self, corpus):
        dataset = _load(corpus, label_rule="likelihood")
        assert len(dataset) == 22
        for name in ("21", "22"):
            assert not dataset.labels[dataset.record_names.index(name)].any()

    def test_both_rules_agree_on_confident_statements(self, corpus):
        benchmark = _load(corpus)
        likelihood = _load(corpus, label_rule="likelihood")
        index = benchmark.record_names.index("11")
        assert benchmark.labels[index].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]
        assert likelihood.labels[likelihood.record_names.index("11")].tolist() == (
            benchmark.labels[index].tolist()
        )

    def test_unknown_rule_is_rejected(self, corpus):
        with pytest.raises(ValueError, match="label_rule"):
            _load(corpus, label_rule="majority")


class TestRecommendedFolds:
    def test_data_module_uses_the_recommended_folds(self, corpus):
        module = ECGDataModule(dataset=_load(corpus), seed=3, verbose=False)
        module.setup()
        assert module.split_protocol == "predefined"
        dataset = module.dataset
        folds = {
            name: sorted({dataset.strat_folds[i] for i in subset.indices})
            for name, subset in (
                ("train", module.train_dataset),
                ("val", module.val_dataset),
                ("test", module.test_dataset),
            )
        }
        assert folds == {"train": list(range(1, 9)), "val": [9], "test": [10]}
        assert len(module.test_dataset) == 2

    def test_the_split_does_not_depend_on_the_seed(self, corpus):
        tests = []
        for seed in (1, 2):
            module = ECGDataModule(dataset=_load(corpus), seed=seed, verbose=False)
            module.setup()
            tests.append(sorted(module.test_dataset.indices))
        assert tests[0] == tests[1]

    def test_switching_the_folds_off_restores_the_seeded_split(self, corpus):
        module = ECGDataModule(
            dataset=_load(corpus, use_recommended_folds=False), seed=3, verbose=False
        )
        module.setup()
        assert module.split_protocol == "seeded"

    def test_describe_records_the_protocol_and_options(self, corpus):
        module = ECGDataModule(dataset=_load(corpus), seed=3, verbose=False)
        module.setup()
        description = module.describe()
        assert description["split_protocol"] == "predefined"
        assert description["dataset_options"] == {
            "label_rule": "benchmark",
            "use_recommended_folds": True,
        }


class TestCheckpointRebuild:
    def _kwargs(self, data_config, dataset="ptbxl"):
        stored = {
            "model_state_dict": {},
            "train_config": {"multi_label": True},
            "data_config": data_config,
        }
        kwargs, _ = checkpoint_dataset_kwargs(dataset, stored, CLILogger(quiet=True))
        return kwargs

    def test_recorded_options_are_restored(self):
        kwargs = self._kwargs(
            {
                "dataset": "ptbxl",
                "preprocessing": {"sampling_rate": 500},
                "dataset_options": {"label_rule": "benchmark", "use_recommended_folds": True},
            }
        )
        assert kwargs["label_rule"] == "benchmark"
        assert kwargs["use_recommended_folds"] is True

    def test_checkpoints_without_options_get_the_earlier_protocol(self):
        kwargs = self._kwargs({"dataset": "ptbxl", "preprocessing": {"sampling_rate": 500}})
        assert kwargs["label_rule"] == "likelihood"
        assert kwargs["use_recommended_folds"] is False

    def test_options_of_another_dataset_are_not_carried_over(self):
        kwargs = self._kwargs(
            {"dataset": "af-classification", "preprocessing": {"sampling_rate": 500}}
        )
        assert "label_rule" not in kwargs
        assert "use_recommended_folds" not in kwargs
