import http.client
import logging
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

from tqdm import tqdm

logger = logging.getLogger(__name__)

#: Suffix of an in-progress download. The final name only ever appears once
#: the file is complete, so a truncated archive can never be mistaken for the
#: real one.
PARTIAL_SUFFIX = ".part"

#: File that sits in a dataset directory while it is being downloaded into.
#: It is removed only when the download finishes, so a directory that was
#: interrupted between two files is still recognised as incomplete.
IN_PROGRESS_MARKER = ".download-in-progress"

_CHUNK_SIZE = 1 << 16
_HTTP_PARTIAL_CONTENT = 206
_HTTP_RANGE_NOT_SATISFIABLE = 416


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        """tqdm reporthook: advance the bar to block b of size bsize."""
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def partial_path(file_path: str | Path) -> Path:
    """Return the path an in-progress download of file_path is written to."""
    file_path = Path(file_path)
    return file_path.with_name(file_path.name + PARTIAL_SUFFIX)


def is_valid_zip(path: str | Path) -> bool:
    """Report whether path is a complete, uncorrupted zip archive."""
    try:
        with zipfile.ZipFile(path, "r") as archive:
            return archive.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def _fetch(
    url: str, part_path: Path, desc: str | None, timeout: float, progress: bool = True
) -> None:
    """Download url into part_path, resuming from whatever it already holds.

    Args:
        url: URL to download from
        part_path: Partial file to append to; created when absent
        desc: Description for the progress bar
        timeout: Seconds to wait on a stalled connection before giving up
        progress: Show a progress bar for this file

    Raises:
        OSError: If the connection fails, or the server closes it before
            sending the length it announced.
    """
    existing = part_path.stat().st_size if part_path.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "deepecgkit"})
    if existing:
        request.add_header("Range", f"bytes={existing}-")

    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        if existing and error.code == _HTTP_RANGE_NOT_SATISFIABLE:
            # The partial file is not a prefix of what the server holds.
            part_path.unlink()
            return _fetch(url, part_path, desc, timeout, progress)
        raise

    with response:
        if existing and getattr(response, "status", None) != _HTTP_PARTIAL_CONTENT:
            # The server ignored the range and is sending the whole file.
            existing = 0
        announced = response.headers.get("Content-Length")
        total = int(announced) + existing if announced is not None else None

        with (
            open(part_path, "ab" if existing else "wb") as out,
            DownloadProgressBar(
                total=total,
                initial=existing,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                miniters=1,
                desc=desc,
                disable=not progress,
            ) as bar,
        ):
            while chunk := response.read(_CHUNK_SIZE):
                out.write(chunk)
                bar.update(len(chunk))

    received = part_path.stat().st_size
    if total is not None and received != total:
        raise OSError(f"connection closed after {received} of {total} bytes")


def download_file(
    url: str | Sequence[str],
    file_path: str | Path,
    desc: str | None = None,
    force: bool = False,
    max_retries: int = 3,
    timeout: float = 60.0,
    progress: bool = True,
) -> None:
    """Download a file with a progress bar, resuming interrupted transfers.

    The transfer is written to ``<file_path>.part`` and renamed only once it
    is complete, so an interruption (a dropped connection, Ctrl-C) leaves a
    partial file that the next call continues from instead of a truncated
    file under the final name.

    Args:
        url: URL to download from, or several mirrors of the same file tried
            in order
        file_path: Path to save the file to
        desc: Description for the progress bar
        force: If True, re-download even if file exists
        max_retries: Maximum number of passes over the mirrors
        timeout: Seconds to wait on a stalled connection before retrying
        progress: Show a progress bar; turned off when many files are
            fetched at once and reported together

    Raises:
        RuntimeError: If every mirror failed on every pass.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    urls = [url] if isinstance(url, str) else list(url)
    if not urls:
        raise ValueError("download_file needs at least one URL")
    is_zip = file_path.suffix.lower() == ".zip"
    part_path = partial_path(file_path)

    if force:
        file_path.unlink(missing_ok=True)
        part_path.unlink(missing_ok=True)

    if file_path.exists():
        if not is_zip or is_valid_zip(file_path):
            log = logger.info if progress else logger.debug
            log(f"File already exists at {file_path}, skipping download.")
            return
        logger.warning(f"Corrupted zip file detected at {file_path}, re-downloading...")
        file_path.unlink()

    def downloaded() -> int:
        return part_path.stat().st_size if part_path.exists() else 0

    last_exception: Exception | None = None
    for attempt in range(max_retries):
        for source in urls:
            while True:
                before = downloaded()
                try:
                    _fetch(source, part_path, desc, timeout, progress)
                    if is_zip and not is_valid_zip(part_path):
                        part_path.unlink(missing_ok=True)
                        raise zipfile.BadZipFile("Downloaded zip file is corrupted")
                    part_path.replace(file_path)
                    return
                except Exception as error:
                    last_exception = error
                    logger.warning(f"Download from {source} failed: {error}")
                    # A mirror that was delivering data just dropped the
                    # connection: resume from it rather than moving to a
                    # slower one. One that delivered nothing is skipped.
                    if downloaded() <= before:
                        break

        if attempt < max_retries - 1:
            wait_time = 2**attempt
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    kept = (
        f" The partial download is kept at {part_path} and resumes on the next attempt."
        if part_path.exists()
        else ""
    )
    raise RuntimeError(
        f"Failed to download {urls[0]} after {max_retries} attempts: {last_exception}.{kept}"
    ) from last_exception


def download_files(
    mirrors: Sequence[str],
    names: Sequence[str],
    dest_dir: str | Path,
    *,
    desc: str,
    workers: int = 1,
    max_retries: int = 3,
    keep_alive: bool = True,
) -> None:
    """Download a set of files from a mirrored directory.

    Files already present are skipped, so calling this again after an
    interruption fetches only what is missing.

    Args:
        mirrors: Base URLs of the same directory, each ending in a slash,
            tried in order for every file
        names: File paths relative to the base URL, kept relative to dest_dir
        dest_dir: Directory to download into
        desc: Description for the progress bar
        workers: Files fetched at a time. With one, each file gets its own
            progress bar; with more, a single bar counts files.
        max_retries: Passes over the mirrors per file before giving up
        keep_alive: With several workers, fetch each file whole over a
            connection the worker keeps open. That suits thousands of small
            records; turn it off for large files, so that each one can
            resume where it stopped.

    Raises:
        RuntimeError: If any file could not be downloaded.
    """
    dest_dir = Path(dest_dir)
    connections = threading.local()

    def fetch(name: str, show_progress: bool) -> None:
        download_file(
            [mirror + name for mirror in mirrors],
            dest_dir / name,
            desc=f"{desc} ({name})",
            max_retries=max_retries,
            progress=show_progress,
        )

    def fetch_reusing_connection(name: str) -> None:
        # Thousands of small files are dominated by connection set-up, so
        # each worker keeps one connection open. Anything unexpected falls
        # back to the ordinary path, which retries, resumes and uses mirrors.
        try:
            _fetch_keepalive(connections, mirrors[0] + name, dest_dir / name)
        except Exception:
            _close_connection(connections)
            fetch(name, False)

    if workers <= 1:
        for name in names:
            fetch(name, True)
        return

    pending = [name for name in names if not (dest_dir / name).exists()]
    with (
        tqdm(total=len(names), initial=len(names) - len(pending), desc=desc, unit="file") as bar,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        if keep_alive:
            futures = [pool.submit(fetch_reusing_connection, name) for name in pending]
        else:
            futures = [pool.submit(fetch, name, False) for name in pending]
        try:
            for future in as_completed(futures):
                future.result()
                bar.update(1)
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise


def _close_connection(connections: threading.local) -> None:
    """Drop the calling thread's kept-alive connection, if it has one."""
    connection = getattr(connections, "connection", None)
    if connection is not None:
        connection.close()
    connections.connection = None
    connections.origin = None


def _fetch_keepalive(connections: threading.local, url: str, file_path: Path) -> None:
    """Download one whole file over the calling thread's kept-alive connection.

    Args:
        connections: Per-thread storage for the open connection
        url: URL to download
        file_path: Destination; written through a ``.part`` file

    Raises:
        OSError: If the server does not answer with the complete file.
    """
    parts = urllib.parse.urlsplit(url)
    origin = (parts.scheme, parts.netloc)
    if getattr(connections, "origin", None) != origin:
        _close_connection(connections)
        factory = (
            http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        )
        connections.connection = factory(parts.netloc, timeout=60.0)
        connections.origin = origin

    target = parts.path + (f"?{parts.query}" if parts.query else "")
    connections.connection.request("GET", target, headers={"User-Agent": "deepecgkit"})
    response = connections.connection.getresponse()
    body = response.read()
    if response.status != http.client.OK:
        raise OSError(f"HTTP {response.status} for {url}")
    announced = response.headers.get("Content-Length")
    if announced is not None and int(announced) != len(body):
        raise OSError(f"connection closed after {len(body)} of {announced} bytes")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = partial_path(file_path)
    part_path.write_bytes(body)
    part_path.replace(file_path)


@contextmanager
def marking_download(data_dir: str | Path) -> Iterator[None]:
    """Mark data_dir as incomplete until the enclosed download finishes.

    The marker is left behind on any failure or interruption, which is what
    lets the next run tell a half-downloaded directory from a complete one.

    Args:
        data_dir: Dataset directory being downloaded into
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    marker = data_dir / IN_PROGRESS_MARKER
    marker.touch()
    yield
    marker.unlink(missing_ok=True)


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
    url: str | Sequence[str],
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
    files end up directly under dest_dir. The archive is deleted afterwards,
    which is also what marks the extraction as finished: an archive still
    sitting in dest_dir means the previous run did not get that far.

    Args:
        url: URL of the zip archive, or several mirrors of it
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


def has_unfinished_download(data_dir: str | Path) -> bool:
    """Report whether data_dir holds a download that never completed.

    The in-progress marker, partial files and leftover archives all count:
    archives are deleted once their content is extracted, so one that is
    still present marks a run that stopped part-way.

    Args:
        data_dir: Dataset directory to inspect

    Returns:
        True when the marker, a ``.part`` file or a ``.zip`` archive sits at
        its top level.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return False
    if (data_dir / IN_PROGRESS_MARKER).exists():
        return True
    return any(
        entry.is_file() and entry.suffix.lower() in (PARTIAL_SUFFIX, ".zip")
        for entry in data_dir.iterdir()
    )


def _try_lock(handle: IO[Any]) -> bool | None:
    """Try to take an exclusive lock on an open file without blocking.

    Returns:
        True when the lock was taken, False when another process holds it,
        and None when this platform offers no file locking.
    """
    try:
        import fcntl
    except ImportError:
        try:
            import msvcrt
        except ImportError:
            return None
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        except OSError:
            return False
        return True
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


@contextmanager
def download_lock(
    target: str | Path, poll_seconds: float = 2.0, timeout: float = 6 * 3600.0
) -> Iterator[None]:
    """Hold the right to download into target, waiting for any other process doing so.

    Two processes started together on a machine without the dataset would
    otherwise both download it into the same files. The lock belongs to the
    process, so it is released when the process ends however it ends. It is
    best effort: where a lock cannot be taken at all, the download proceeds
    as it would have without one.

    Args:
        target: Directory about to be downloaded into
        poll_seconds: Pause between attempts while another process holds the lock
        timeout: Seconds to wait before proceeding without the lock
    """
    target = Path(target)
    lock_path = target.parent / f".{target.name}.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+")
    except OSError:
        yield
        return

    try:
        waited = 0.0
        announced = False
        while _try_lock(handle) is False and waited < timeout:
            if not announced:
                logger.info(f"Another process is downloading into {target}; waiting for it...")
                announced = True
            time.sleep(poll_seconds)
            waited += poll_seconds
        yield
    finally:
        handle.close()
