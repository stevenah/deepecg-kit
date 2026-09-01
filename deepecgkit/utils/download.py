import logging
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        """tqdm reporthook: advance the bar to block b of size bsize."""
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(
    url: str,
    file_path: str | Path,
    desc: str | None = None,
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
                        logger.warning(
                            f"Corrupted zip file detected at {file_path}, re-downloading..."
                        )
                        file_path.unlink()
                    else:
                        logger.info(f"File already exists at {file_path}, skipping download.")
                        return
            except zipfile.BadZipFile:
                logger.warning(f"Corrupted zip file detected at {file_path}, re-downloading...")
                file_path.unlink()
        else:
            logger.info(f"File already exists at {file_path}, skipping download.")
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
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(
                    f"Failed to download {url} after {max_retries} attempts: {last_exception}"
                ) from last_exception


def flatten_directory(nested_dir: Path, dest_dir: Path) -> None:
    """Move everything inside nested_dir up into dest_dir and remove it.

    Existing entries in dest_dir with the same names are replaced.

    Args:
        nested_dir: Directory whose contents should move up
        dest_dir: Directory receiving the contents
    """
    for item in nested_dir.iterdir():
        dest = dest_dir / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))
    nested_dir.rmdir()


def download_and_extract_zip(
    url: str,
    dest_dir: str | Path,
    *,
    zip_name: str,
    desc: str,
    nested_dirs: tuple[str, ...] | None = None,
    flatten_files_dir: bool = True,
    verbose: bool = True,
    max_retries: int = 5,
) -> None:
    """Download a dataset archive, extract it in place, and tidy the layout.

    PhysioNet zips wrap their content in a versioned top-level directory and
    sometimes a further files/ directory; both are flattened away so record
    files end up directly under dest_dir. The archive is deleted afterwards.

    Args:
        url: URL of the zip archive
        dest_dir: Directory to extract into (created if missing)
        zip_name: File name for the downloaded archive
        desc: Description for the download progress bar
        nested_dirs: Top-level directory names to flatten when present;
            defaults to the archive name without its extension
        flatten_files_dir: Also flatten a files/ directory containing .hea
            record files, as some PhysioNet archives nest records there
        verbose: Print progress messages
        max_retries: Download attempts before giving up
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / zip_name
    download_file(url, zip_path, desc=desc, max_retries=max_retries)

    if verbose:
        logger.info("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    for name in nested_dirs if nested_dirs is not None else (zip_path.stem,):
        nested = dest_dir / name
        if nested.is_dir():
            flatten_directory(nested, dest_dir)

    if flatten_files_dir:
        files_dir = dest_dir / "files"
        if files_dir.is_dir() and any(files_dir.glob("*.hea")):
            flatten_directory(files_dir, dest_dir)

    zip_path.unlink(missing_ok=True)
