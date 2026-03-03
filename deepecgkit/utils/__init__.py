import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from deepecgkit.utils.weights import (
    list_pretrained_weights,
    load_pretrained_weights,
    register_weights,
)

__all__ = [
    "list_pretrained_weights",
    "load_pretrained_weights",
    "read_csv",
    "register_weights",
]


def read_csv(
    csv_file: str,
    delimiter: str = ",",
    transpose: bool = False,
    skip_header: bool = True,
    dtype: Optional[type] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Read CSV file and return data array and header mapping.

    Args:
        csv_file: Path to the CSV file
        delimiter: Column delimiter (default: ',')
        transpose: Whether to transpose the data array
        skip_header: Whether to skip the first row as header
        dtype: Data type for the numpy array

    Returns:
        Tuple of (data_array, header_mapping)
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    data: List[List[str]] = []
    header: Dict[str, int] = {}

    try:
        with open(csv_file) as f:
            csv_data = csv.reader(f, delimiter=delimiter)

            if skip_header:
                try:
                    temp = next(csv_data)
                    header = {k: v for v, k in enumerate(temp)}
                except StopIteration as err:
                    raise ValueError("File is empty") from err

            for row in csv_data:
                data.append(row)

        if not data:
            raise ValueError("No data found in CSV file")

        data_array = np.array(data, dtype=dtype)

        if transpose:
            data_array = np.transpose(data_array)

        return data_array, header

    except Exception as err:
        raise ValueError(f"Error reading CSV file: {err!s}") from err
