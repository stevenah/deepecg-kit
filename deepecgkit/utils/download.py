import time
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Union

from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(
    url: str,
    file_path: Union[str, Path],
    desc: Optional[str] = None,
    force: bool = False,
    max_retries: int = 3,
) -> None:
    """Download a file with a progress bar.

    Args:
        url: URL to download from
        file_path: Path to save the file to
        desc: Description for the progress bar
        force: If True, re-download even if file exists
        max_retries: Maximum number of download attempts
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.exists() and not force:
        if file_path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    if zip_ref.testzip() is not None:
                        print(f"Corrupted zip file detected at {file_path}, re-downloading...")
                        file_path.unlink()
                    else:
                        print(f"File already exists at {file_path}, skipping download.")
                        return
            except zipfile.BadZipFile:
                print(f"Corrupted zip file detected at {file_path}, re-downloading...")
                file_path.unlink()
        else:
            print(f"File already exists at {file_path}, skipping download.")
            return

    last_exception = None
    for attempt in range(max_retries):
        try:
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=desc) as t:
                urllib.request.urlretrieve(url, filename=file_path, reporthook=t.update_to)

            if file_path.suffix.lower() == ".zip":
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    if zip_ref.testzip() is not None:
                        raise zipfile.BadZipFile("Downloaded zip file is corrupted")
            return

        except Exception as e:
            last_exception = e
            if file_path.exists():
                file_path.unlink()

            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(f"Download attempt {attempt + 1} failed: {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(
                    f"Failed to download {url} after {max_retries} attempts: {last_exception}"
                ) from last_exception


def download_and_extract(
    urls: List[str],
    file_paths: List[Union[str, Path]],
    extract_dir: Optional[Union[str, Path]] = None,
    descs: Optional[List[str]] = None,
    force: bool = False,
) -> None:
    """Download files and optionally extract zip files.

    Args:
        urls: List of URLs to download from
        file_paths: List of paths to save the files to
        extract_dir: Directory to extract zip files to. If None, files are not extracted
        descs: List of descriptions for the progress bars
        force: If True, re-download even if files exist
    """
    if descs is None:
        descs = [f"Downloading file {i + 1}/{len(urls)}" for i in range(len(urls))]

    for url, file_path_str, desc in zip(urls, file_paths, descs):
        file_path = Path(file_path_str)
        download_file(url, file_path, desc, force=force, max_retries=3)

        if extract_dir is not None and file_path.suffix.lower() == ".zip":
            extract_dir = Path(extract_dir)
            extracted_marker = extract_dir / f".{file_path.stem}_extracted"

            if extracted_marker.exists() and not force:
                print(f"{file_path.name} already extracted, skipping...")
                continue

            print(f"Extracting {file_path.name} to {extract_dir}...")
            try:
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
                extracted_marker.touch()
                file_path.unlink()
            except Exception as e:
                if extracted_marker.exists():
                    extracted_marker.unlink()
                raise RuntimeError(f"Failed to extract {file_path}: {e}") from e
