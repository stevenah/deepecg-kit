import ast
import os
import shutil
import zipfile
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import wfdb
from tqdm import tqdm

from deepecgkit.registry import register_dataset

from ..utils.download import download_file
from .base import BaseECGDataset
from .preprocessing import ECGStandardizer, convert_to_tensor


@register_dataset(
    name="ptbxl",
    input_channels=12,
    num_classes=5,
    description="PTB-XL 12-lead ECG dataset (multi-task)",
)
class PTBXLDataset(BaseECGDataset):
    """PTB-XL ECG Dataset.

    PTB-XL is a large publicly available electrocardiography dataset containing
    21,837 clinical 12-lead ECGs from 18,885 patients of 10 second length.
    Each ECG is annotated with up to 71 different diagnostic statements conforming
    to the SCP-ECG standard.

    The dataset supports multiple diagnostic classification tasks:
    - Superclass: 5 diagnostic superclasses (NORM, MI, STTC, CD, HYP)
    - Subclass: 23 diagnostic subclasses
    - Form: 19 form statements
    - Rhythm: 12 rhythm statements
    - All: All 71 statements

    Reference:
        Wagner P, Strodthoff N, Bousseljot RD, Kreiseler D, Lunze FI, Samek W, Schaeffter T.
        PTB-XL, a large publicly available electrocardiography dataset.
        Scientific Data. 2020 May 25;7(1):154.

    URL:
        https://physionet.org/content/ptb-xl/1.0.3/
    """

    CLASS_LABELS_SUPERCLASS: ClassVar[List[str]] = ["NORM", "MI", "STTC", "CD", "HYP"]
    CLASS_LABELS_SUBCLASS: ClassVar[List[str]] = [
        "NORM",
        "IMI",
        "ASMI",
        "ILMI",
        "AMI",
        "ALMI",
        "INJAS",
        "LMI",
        "INJAL",
        "ISCAL",
        "ISCAN",
        "INJIN",
        "INJLA",
        "PMI",
        "INJIL",
        "ISCIN",
        "ISCIL",
        "ISCAS",
        "LAFB",
        "IRBBB",
        "LPFB",
        "CRBBB",
        "CLBBB",
    ]
    CLASS_LABELS_FORM: ClassVar[List[str]] = [
        "NDT",
        "NST_",
        "DIG",
        "LNGQT",
        "ABQRS",
        "PVC",
        "STD_",
        "VCLVH",
        "QWAVE",
        "LOWT",
        "NT_",
        "PAC",
        "LPR",
        "INVT",
        "LVOLT",
        "HVOLT",
        "TAB_",
        "STE_",
        "PRC(S)",
    ]
    CLASS_LABELS_RHYTHM: ClassVar[List[str]] = [
        "SR",
        "AFIB",
        "STACH",
        "SARRH",
        "SBRAD",
        "PACE",
        "SVARR",
        "BIGU",
        "AFLT",
        "SVTAC",
        "PSVT",
        "TRIGU",
    ]

    LABEL_MAPPING_SUPERCLASS: ClassVar[Dict[str, int]] = {
        label: i for i, label in enumerate(CLASS_LABELS_SUPERCLASS)
    }

    LEADS: ClassVar[List[str]] = [
        "I",
        "II",
        "III",
        "AVR",
        "AVL",
        "AVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]
    SAMPLING_RATE_HR: ClassVar[int] = 500
    SAMPLING_RATE_LR: ClassVar[int] = 100

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        sampling_rate: int = 500,
        task: str = "superclass",
        use_high_resolution: bool = True,
        folds: Optional[List[int]] = None,
        leads: Optional[List[str]] = None,
        normalization: str = "zscore",
        multi_label: bool = True,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        download: bool = False,
        force_download: bool = False,
        verbose: bool = True,
    ):
        """Initialize the PTB-XL dataset.

        Args:
            data_dir: Directory where the dataset is stored or will be downloaded.
                     If None, uses ~/.deepecgkit/datasets/ptbxldataset
            sampling_rate: Target sampling rate for the ECG signals (Hz)
            task: Classification task - one of "superclass", "subclass", "form",
                  "rhythm", or "all"
            use_high_resolution: Whether to use 500Hz (True) or 100Hz (False) recordings
            folds: List of folds to include (1-10). None means all folds.
                   Recommended: folds 1-8 for training, 9 for validation, 10 for testing
            leads: List of lead names to use. None means all 12 leads.
            normalization: Normalization method - "zscore", "minmax", or "none"
            multi_label: If True, returns multi-hot encoded labels. If False, returns
                        single label (first/primary diagnosis)
            transform: Optional transform to be applied to the ECG signals
            target_transform: Optional transform to be applied to the labels
            download: Whether to download the dataset if it doesn't exist
            verbose: Whether to print progress information
        """
        valid_tasks = ["superclass", "subclass", "form", "rhythm", "all"]
        if task not in valid_tasks:
            raise ValueError(f"task must be one of {valid_tasks}, got '{task}'")

        self.task = task
        self.use_high_resolution = use_high_resolution
        self.folds = folds
        self.multi_label = multi_label
        self.verbose = verbose
        self._leads = leads

        source_sampling_rate = (
            self.SAMPLING_RATE_HR if use_high_resolution else self.SAMPLING_RATE_LR
        )

        self.standardizer = ECGStandardizer(
            target_sampling_rate=sampling_rate,
            target_duration_seconds=10.0,
            normalization=normalization,
            clip_method="center",
        )

        self.signals: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []
        self.record_names: List[str] = []
        self.metadata_df: Optional[pd.DataFrame] = None
        self.scp_statements: Optional[pd.DataFrame] = None
        self.source_sampling_rate = source_sampling_rate

        super().__init__(
            data_dir=data_dir,
            sampling_rate=sampling_rate,
            leads=leads if leads else self.LEADS,
            transform=transform,
            target_transform=target_transform,
            download=download,
            force_download=force_download,
        )

    @staticmethod
    def _fix_record_list(records: List[str]) -> List[str]:
        """Fix corrupted RECORDS file from PhysioNet.

        The PTB-XL RECORDS file on PhysioNet has a missing newline that
        concatenates two record paths (e.g. 'records100/...records500/...').
        This splits them back into separate entries.
        """
        fixed = []
        for rec in records:
            if "records100" in rec and "records500" in rec:
                idx = rec.index("records500")
                fixed.append(rec[:idx])
                fixed.append(rec[idx:])
            else:
                fixed.append(rec)
        return fixed

    DOWNLOAD_URL: ClassVar[str] = (
        "https://physionet.org/static/published-projects/ptb-xl/"
        "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    )

    def download(self):
        """Download the PTB-XL dataset from PhysioNet as a single ZIP."""
        if self.verbose:
            print(f"Downloading PTB-XL dataset to {self.data_dir}")
            print("Note: This is a large dataset (~2.5GB). Download may take a while.")

        os.makedirs(self.data_dir, exist_ok=True)

        zip_path = self.data_dir / "ptb-xl-1.0.3.zip"
        download_file(
            self.DOWNLOAD_URL,
            zip_path,
            desc="Downloading PTB-XL",
            max_retries=5,
        )

        if self.verbose:
            print("Extracting dataset...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.data_dir)

        nested_dir = (
            self.data_dir / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
        )
        if nested_dir.is_dir():
            for item in nested_dir.iterdir():
                dest = self.data_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            nested_dir.rmdir()

        zip_path.unlink(missing_ok=True)

        if self.verbose:
            print("Download complete!")

    def _load_data(self):
        """Load the PTB-XL dataset into memory."""
        if self.verbose:
            print("Loading PTB-XL dataset...")

        metadata_path = self.data_dir / "ptbxl_database.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                "Use download=True or manually download from: "
                "https://physionet.org/content/ptb-xl/1.0.3/"
            )

        self.metadata_df = pd.read_csv(metadata_path, index_col="ecg_id")
        self.metadata_df.scp_codes = self.metadata_df.scp_codes.apply(ast.literal_eval)

        scp_path = self.data_dir / "scp_statements.csv"
        if scp_path.exists():
            self.scp_statements = pd.read_csv(scp_path, index_col=0)
        else:
            self.scp_statements = None

        if self.folds is not None:
            self.metadata_df = self.metadata_df[self.metadata_df.strat_fold.isin(self.folds)]
            if self.verbose:
                print(f"Using folds: {self.folds} ({len(self.metadata_df)} records)")

        label_columns = self._get_label_columns()

        if self._leads:
            lead_indices = [self.LEADS.index(lead) for lead in self._leads if lead in self.LEADS]
        else:
            lead_indices = list(range(len(self.LEADS)))

        iterator = (
            tqdm(self.metadata_df.iterrows(), total=len(self.metadata_df), desc="Loading records")
            if self.verbose
            else self.metadata_df.iterrows()
        )

        for ecg_id, row in iterator:
            try:
                record_path = (
                    self.data_dir / row.filename_hr
                    if self.use_high_resolution
                    else self.data_dir / row.filename_lr
                )
                record_path = str(record_path).replace(".hea", "").replace(".dat", "")

                record = wfdb.rdrecord(record_path)
                signal = record.p_signal.T

                signal = signal[lead_indices, :]

                if self.source_sampling_rate != self.sampling_rate:
                    signal = self.standardizer.resample(signal, self.source_sampling_rate)

                signal = self.standardizer.normalize(signal)

                labels = self._extract_labels(row.scp_codes, label_columns)

                self.signals.append(signal.astype(np.float32))
                self.labels.append(labels)
                self.record_names.append(str(ecg_id))

            except Exception as e:
                if self.verbose:
                    print(f"Error loading record {ecg_id}: {e}")
                continue

        if self.verbose:
            print(f"Successfully loaded {len(self.signals)} records")
            self._print_class_distribution()

    def _get_label_columns(self) -> List[str]:
        """Get the label columns based on the task."""
        if self.task == "superclass":
            return self.CLASS_LABELS_SUPERCLASS
        elif self.task == "subclass":
            return self.CLASS_LABELS_SUBCLASS
        elif self.task == "form":
            return self.CLASS_LABELS_FORM
        elif self.task == "rhythm":
            return self.CLASS_LABELS_RHYTHM
        else:
            all_labels = (
                self.CLASS_LABELS_SUPERCLASS
                + self.CLASS_LABELS_SUBCLASS
                + self.CLASS_LABELS_FORM
                + self.CLASS_LABELS_RHYTHM
            )
            return list(set(all_labels))

    def _extract_labels(self, scp_codes: Dict[str, float], label_columns: List[str]) -> np.ndarray:
        """Extract labels from SCP codes."""
        if self.multi_label:
            labels = np.zeros(len(label_columns), dtype=np.float32)
            if self.task == "superclass" and self.scp_statements is not None:
                for code, likelihood in scp_codes.items():
                    if likelihood >= 50.0 and code in self.scp_statements.index:
                        stmt = self.scp_statements.loc[code]
                        if pd.notna(stmt.diagnostic_class):
                            superclass = stmt.diagnostic_class
                            if superclass in label_columns:
                                idx = label_columns.index(superclass)
                                labels[idx] = 1.0
            else:
                for code, likelihood in scp_codes.items():
                    if code in label_columns:
                        idx = label_columns.index(code)
                        labels[idx] = 1.0 if likelihood >= 50.0 else 0.0
            return labels
        else:
            if self.task == "superclass" and self.scp_statements is not None:
                superclass_counts = {cls: 0.0 for cls in self.CLASS_LABELS_SUPERCLASS}
                for code, likelihood in scp_codes.items():
                    if likelihood >= 50.0 and code in self.scp_statements.index:
                        stmt = self.scp_statements.loc[code]
                        if pd.notna(stmt.diagnostic_class):
                            superclass = stmt.diagnostic_class
                            if superclass in superclass_counts:
                                superclass_counts[superclass] += likelihood
                if any(v > 0 for v in superclass_counts.values()):
                    primary_class = max(superclass_counts, key=superclass_counts.get)
                    return np.array(
                        self.CLASS_LABELS_SUPERCLASS.index(primary_class), dtype=np.int64
                    )
            for code, likelihood in scp_codes.items():
                if code in label_columns and likelihood >= 50.0:
                    return np.array(label_columns.index(code), dtype=np.int64)
            return np.array(0, dtype=np.int64)

    def _print_class_distribution(self):
        """Print class distribution statistics."""
        if len(self.labels) == 0:
            return

        label_columns = self._get_label_columns()
        print(f"\nClass distribution ({self.task}):")

        if self.multi_label:
            labels_array = np.stack(self.labels)
            for i, class_name in enumerate(label_columns):
                count = int(labels_array[:, i].sum())
                if count > 0:
                    print(f"  {class_name}: {count}")
        else:
            unique, counts = np.unique(self.labels, return_counts=True)
            for label_idx, count in zip(unique, counts):
                if label_idx < len(label_columns):
                    print(f"  {label_columns[label_idx]}: {count}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to get

        Returns:
            Tuple of (ecg_signal, label) where:
                - ecg_signal: Tensor of shape (num_leads, signal_length)
                - label: Tensor of shape (num_classes,) for multi-label or scalar for single-label
        """
        signal = self.signals[idx]
        label = self.labels[idx]

        signal = convert_to_tensor(signal, dtype=torch.float32)

        if self.multi_label:
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            signal = self.transform(signal)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return signal, label

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def num_classes(self) -> int:
        """Get the number of classes in the dataset."""
        return len(self._get_label_columns())

    @property
    def class_names(self) -> List[str]:
        """Get the names of the classes in the dataset."""
        return self._get_label_columns()

    def get_record_info(self, idx: int) -> Dict:
        """Get record information for a specific sample.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing record information
        """
        ecg_id = int(self.record_names[idx])
        row = self.metadata_df.loc[ecg_id]

        return {
            "ecg_id": ecg_id,
            "patient_id": row.patient_id,
            "age": row.age,
            "sex": row.sex,
            "signal_shape": self.signals[idx].shape,
            "scp_codes": row.scp_codes,
            "strat_fold": row.strat_fold,
            "labels": self.labels[idx].tolist() if self.multi_label else int(self.labels[idx]),
        }

    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset.

        Returns:
            Dictionary mapping class names to their counts
        """
        label_columns = self._get_label_columns()
        distribution = {name: 0 for name in label_columns}

        if self.multi_label:
            labels_array = np.stack(self.labels)
            for i, class_name in enumerate(label_columns):
                distribution[class_name] = int(labels_array[:, i].sum())
        else:
            for lbl in self.labels:
                label_idx = int(lbl) if isinstance(lbl, np.ndarray) else lbl
                if label_idx < len(label_columns):
                    distribution[label_columns[label_idx]] += 1

        return distribution

    def get_folds_split(
        self,
        train_folds: Optional[List[int]] = None,
        val_folds: Optional[List[int]] = None,
        test_folds: Optional[List[int]] = None,
    ) -> Dict[str, "PTBXLDataset"]:
        """Create train/val/test splits based on stratified folds.

        The PTB-XL dataset comes with 10 pre-defined stratified folds.
        Recommended split: folds 1-8 for training, 9 for validation, 10 for testing.

        Args:
            train_folds: Folds for training (default: 1-8)
            val_folds: Folds for validation (default: 9)
            test_folds: Folds for testing (default: 10)

        Returns:
            Dictionary with 'train', 'val', 'test' PTBXLDataset instances
        """
        if train_folds is None:
            train_folds = list(range(1, 9))
        if val_folds is None:
            val_folds = [9]
        if test_folds is None:
            test_folds = [10]

        common_kwargs = {
            "data_dir": self.data_dir,
            "sampling_rate": self.sampling_rate,
            "task": self.task,
            "use_high_resolution": self.use_high_resolution,
            "leads": self._leads,
            "normalization": self.standardizer.normalization,
            "multi_label": self.multi_label,
            "transform": self.transform,
            "target_transform": self.target_transform,
            "download": False,
            "verbose": self.verbose,
        }

        return {
            "train": PTBXLDataset(folds=train_folds, **common_kwargs),
            "val": PTBXLDataset(folds=val_folds, **common_kwargs),
            "test": PTBXLDataset(folds=test_folds, **common_kwargs),
        }
