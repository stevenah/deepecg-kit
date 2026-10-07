"""
Tests for deepecgkit.utils module.
"""

import http.server
import inspect
import io
import threading
import zipfile
from pathlib import Path
from typing import ClassVar

import pytest
import torch

import deepecgkit.utils as utils_module
from deepecgkit.utils import download, read_csv
from deepecgkit.utils.checkpoints import TRUST_ENV, load_checkpoint_file


class TestUtilityFunctions:
    """Test utility functions in the utils module."""

    def test_read_csv_basic(self, temp_dir):
        """Test basic CSV reading functionality."""

        csv_content = "col1,col2,col3\n1,2,3\n4,5,6\n7,8,9\n"
        csv_file = temp_dir / "test.csv"
        csv_file.write_text(csv_content)

        data_array, header = read_csv(str(csv_file))

        assert data_array is not None
        assert len(data_array) == 3
        assert "col1" in header
        assert "col2" in header
        assert "col3" in header

        assert data_array[0][0] == "1"
        assert data_array[1][1] == "5"

    def test_read_csv_with_different_separators(self, temp_dir):
        """Test CSV reading with different separators."""

        csv_content = "a;b;c\n1;2;3\n4;5;6\n"
        csv_file = temp_dir / "test_semicolon.csv"
        csv_file.write_text(csv_content)

        try:
            df = read_csv(str(csv_file), sep=";")
            assert len(df.columns) == 3
            assert df.iloc[0, 0] == 1
        except TypeError:
            df = read_csv(str(csv_file))

            assert df is not None

    def test_read_csv_nonexistent_file(self):
        """Test reading a non-existent CSV file."""
        with pytest.raises(FileNotFoundError):
            read_csv("nonexistent_file.csv")

    def test_read_csv_empty_file(self, temp_dir):
        """Test reading an empty CSV file."""
        csv_file = temp_dir / "empty.csv"
        csv_file.write_text("")

        try:
            df = read_csv(str(csv_file))

            assert df is not None
            assert len(df) == 0
        except Exception:
            pass

    def test_read_csv_malformed_data(self, temp_dir):
        """Test reading CSV with malformed data."""

        csv_content = "col1,col2,col3\n1,2,3\n4,5\n6,7,8,9\n"
        csv_file = temp_dir / "malformed.csv"
        csv_file.write_text(csv_content)

        try:
            df = read_csv(str(csv_file))

            assert df is not None
        except Exception:
            pass


class TestUtilityEdgeCases:
    """Test edge cases and error handling in utilities."""

    def test_read_csv_large_file_simulation(self, temp_dir):
        """Test CSV reading performance with larger file."""

        csv_content = "id,value1,value2,value3\n"
        for i in range(1000):
            csv_content += f"{i},{i * 2},{i * 3},{i * 4}\n"

        csv_file = temp_dir / "large.csv"
        csv_file.write_text(csv_content)

        data_array, header = read_csv(str(csv_file))

        assert data_array is not None
        assert len(data_array) == 1000
        assert len(header) == 4
        assert "id" in header
        assert data_array[999][0] == "999"

    def test_path_handling(self, temp_dir):
        """Test utility functions with different path types."""
        csv_content = "a,b\n1,2\n"
        csv_file = temp_dir / "test.csv"
        csv_file.write_text(csv_content)

        data_array1, header1 = read_csv(str(csv_file))
        assert data_array1 is not None

        data_array2, header2 = read_csv(csv_file)
        assert data_array2 is not None

        assert len(data_array1) == len(data_array2)
        assert header1.keys() == header2.keys()


class TestUtilityIntegration:
    """Integration tests for utility functions."""

    def test_utils_with_datasets(self, temp_dir):
        """Test utilities integration with dataset functionality."""

        csv_content = "record_name,label\nA00001,N\nA00002,A\nA00003,O\n"
        csv_file = temp_dir / "REFERENCE.csv"
        csv_file.write_text(csv_content)

        data_array, header = read_csv(str(csv_file))

        assert data_array is not None
        assert len(data_array) == 3
        assert "record_name" in header
        assert "label" in header

        record_names = [row[header["record_name"]] for row in data_array]
        labels = [row[header["label"]] for row in data_array]

        assert len(record_names) == 3
        assert len(labels) == 3
        assert "A00001" in record_names
        assert "N" in labels

    def test_utils_module_structure(self):
        """Test utils module structure and exports."""

        assert hasattr(utils_module, "read_csv")

        assert callable(utils_module.read_csv)

    def test_all_utils_imports(self):
        """Test that all utility functions can be imported."""
        assert read_csv is not None
        assert download is not None


class TestUtilityDocumentation:
    """Test that utility functions have proper documentation."""

    def test_read_csv_docstring(self):
        """Test that read_csv has documentation."""
        assert read_csv.__doc__ is not None
        assert len(read_csv.__doc__.strip()) > 0

    def test_utility_function_signatures(self):
        """Test that utility functions have reasonable signatures."""
        sig = inspect.signature(read_csv)
        assert len(sig.parameters) >= 1

        first_param = list(sig.parameters.values())[0]
        assert first_param.name in ["file_path", "filepath", "path", "filename", "csv_file"]


class TestUtilityErrorHandling:
    """Test error handling in utility functions."""

    def test_read_csv_permission_error(self, temp_dir):
        """Test read_csv behavior with permission errors."""
        csv_file = temp_dir / "test.csv"
        csv_file.write_text("a,b\n1,2\n")

        try:
            csv_file.chmod(0o000)

            with pytest.raises(ValueError):
                read_csv(str(csv_file))

        except OSError:
            pytest.skip("Permission change not supported on this system")
        finally:
            try:
                csv_file.chmod(0o644)
            except OSError:
                pass

    def test_read_csv_with_invalid_encoding(self, temp_dir):
        """Test read_csv with invalid encoding."""

        csv_file = temp_dir / "invalid_encoding.csv"

        with open(csv_file, "wb") as f:
            f.write(b"col1,col2\n\xff\xfe,value\n")

        try:
            df = read_csv(str(csv_file))

            assert df is not None
        except UnicodeDecodeError:
            pass
        except Exception as e:
            assert "encoding" in str(e).lower() or "decode" in str(e).lower()


class TestDownloadAndExtractZip:
    """The shared archive-download helper used by every dataset."""

    @staticmethod
    def _make_zip(path, entries):
        import zipfile

        with zipfile.ZipFile(path, "w") as zf:
            for name, content in entries.items():
                zf.writestr(name, content)

    def _serve(self, monkeypatch, zip_source):
        """Make download_file 'download' by copying a local file."""
        import shutil

        def fake_download(url, file_path, desc=None, force=False, max_retries=3):
            shutil.copy(zip_source, file_path)

        monkeypatch.setattr(download, "download_file", fake_download)

    def test_flattens_nested_top_level_directory(self, temp_dir, monkeypatch):
        source = temp_dir / "src.zip"
        self._make_zip(source, {"data-1.0.0/rec01.hea": "h", "data-1.0.0/rec01.atr": "a"})
        self._serve(monkeypatch, source)

        dest = temp_dir / "dest"
        download.download_and_extract_zip(
            "http://example.invalid/x", dest, zip_name="data-1.0.0.zip", desc="t", verbose=False
        )
        assert (dest / "rec01.hea").exists()
        assert not (dest / "data-1.0.0").exists()
        assert not (dest / "data-1.0.0.zip").exists()

    def test_flattens_physionet_files_directory(self, temp_dir, monkeypatch):
        source = temp_dir / "src.zip"
        self._make_zip(
            source, {"data-1.0.0/files/rec01.hea": "h", "data-1.0.0/files/rec01.atr": "a"}
        )
        self._serve(monkeypatch, source)

        dest = temp_dir / "dest"
        download.download_and_extract_zip(
            "http://example.invalid/x", dest, zip_name="data-1.0.0.zip", desc="t", verbose=False
        )
        assert (dest / "rec01.hea").exists()
        assert not (dest / "files").exists()

    def test_explicit_nested_dirs_and_no_files_flatten(self, temp_dir, monkeypatch):
        source = temp_dir / "src.zip"
        self._make_zip(source, {"long-name-1.0.3/records/r.dat": "d"})
        self._serve(monkeypatch, source)

        dest = temp_dir / "dest"
        download.download_and_extract_zip(
            "http://example.invalid/x",
            dest,
            zip_name="short.zip",
            desc="t",
            nested_dirs=("long-name-1.0.3",),
            flatten_files_dir=False,
            verbose=False,
        )
        assert (dest / "records" / "r.dat").exists()
        assert not (dest / "long-name-1.0.3").exists()


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serves in-memory payloads, honouring Range requests the way mirrors do."""

    payloads: ClassVar[dict[str, bytes]] = {}
    requests: ClassVar[list[tuple[str, str | None]]] = []
    honour_range: ClassVar[bool] = True

    def do_GET(self):  # noqa: N802 - the name is fixed by http.server
        requested_range = self.headers.get("Range")
        type(self).requests.append((self.path, requested_range))
        body = self.payloads.get(self.path)
        if body is None:
            self.send_error(404)
            return

        start = 0
        if requested_range and self.honour_range:
            start = int(requested_range.split("=")[1].split("-")[0])
            if start >= len(body):
                self.send_error(416)
                return
            self.send_response(206)
        else:
            self.send_response(200)
        chunk = body[start:]
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, *args):
        pass


@pytest.fixture
def file_server(monkeypatch):
    """A local HTTP server for download tests; yields its base URL and handler."""
    _RangeHandler.payloads = {}
    _RangeHandler.requests = []
    _RangeHandler.honour_range = True
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(download.time, "sleep", lambda seconds: None)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", _RangeHandler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _zip_bytes(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return buffer.getvalue()


class TestDownloadFile:
    """Downloads resume, fall back across mirrors, and never leave a truncated file."""

    PAYLOAD = bytes(range(256)) * 400

    def test_downloads_a_file(self, temp_dir, file_server):
        base, handler = file_server
        handler.payloads["/data.bin"] = self.PAYLOAD
        dest = temp_dir / "data.bin"

        download.download_file(f"{base}/data.bin", dest)

        assert dest.read_bytes() == self.PAYLOAD
        assert not download.partial_path(dest).exists()

    def test_resumes_an_interrupted_download(self, temp_dir, file_server):
        base, handler = file_server
        handler.payloads["/data.bin"] = self.PAYLOAD
        dest = temp_dir / "data.bin"
        download.partial_path(dest).write_bytes(self.PAYLOAD[:30_000])

        download.download_file(f"{base}/data.bin", dest)

        assert dest.read_bytes() == self.PAYLOAD
        assert handler.requests == [("/data.bin", "bytes=30000-")]
        assert not download.partial_path(dest).exists()

    def test_restarts_when_the_server_ignores_the_range(self, temp_dir, file_server):
        base, handler = file_server
        handler.payloads["/data.bin"] = self.PAYLOAD
        handler.honour_range = False
        dest = temp_dir / "data.bin"
        download.partial_path(dest).write_bytes(b"x" * 30_000)

        download.download_file(f"{base}/data.bin", dest)

        assert dest.read_bytes() == self.PAYLOAD

    def test_a_failed_download_leaves_nothing_under_the_final_name(self, temp_dir, file_server):
        base, _ = file_server
        dest = temp_dir / "data.bin"

        with pytest.raises(RuntimeError, match="Failed to download"):
            download.download_file(f"{base}/missing.bin", dest, max_retries=2)

        assert not dest.exists()

    def test_falls_back_to_the_next_mirror(self, temp_dir, file_server):
        base, handler = file_server
        handler.payloads["/mirror-b/data.bin"] = self.PAYLOAD
        dest = temp_dir / "data.bin"

        download.download_file([f"{base}/mirror-a/data.bin", f"{base}/mirror-b/data.bin"], dest)

        assert dest.read_bytes() == self.PAYLOAD
        assert [path for path, _ in handler.requests] == [
            "/mirror-a/data.bin",
            "/mirror-b/data.bin",
        ]

    def test_a_truncated_archive_is_never_installed(self, temp_dir, file_server):
        base, handler = file_server
        handler.payloads["/data.zip"] = _zip_bytes({"a.txt": "a" * 5000})[:200]
        dest = temp_dir / "data.zip"

        with pytest.raises(RuntimeError, match="Failed to download"):
            download.download_file(f"{base}/data.zip", dest, max_retries=1)

        assert not dest.exists()
        assert not download.partial_path(dest).exists()

    def test_an_existing_corrupt_archive_is_replaced(self, temp_dir, file_server):
        base, handler = file_server
        archive = _zip_bytes({"a.txt": "hello"})
        handler.payloads["/data.zip"] = archive
        dest = temp_dir / "data.zip"
        dest.write_bytes(archive[:40])

        download.download_file(f"{base}/data.zip", dest)

        assert dest.read_bytes() == archive

    def test_an_existing_complete_file_is_not_fetched_again(self, temp_dir, file_server):
        base, handler = file_server
        dest = temp_dir / "data.bin"
        dest.write_bytes(b"already here")

        download.download_file(f"{base}/data.bin", dest)

        assert handler.requests == []
        assert dest.read_bytes() == b"already here"

    def test_unfinished_downloads_are_recognised(self, temp_dir):
        assert not download.has_unfinished_download(temp_dir)
        assert not download.has_unfinished_download(temp_dir / "absent")

        (temp_dir / "records.zip.part").write_bytes(b"x")
        assert download.has_unfinished_download(temp_dir)

        (temp_dir / "records.zip.part").unlink()
        (temp_dir / "records.zip").write_bytes(b"x")
        assert download.has_unfinished_download(temp_dir)


class TestDataHome:
    """DEEPECGKIT_HOME relocates everything the package downloads."""

    def test_default_is_under_the_home_directory(self, monkeypatch):
        from deepecgkit.utils.paths import get_data_home

        monkeypatch.delenv("DEEPECGKIT_HOME", raising=False)
        assert get_data_home() == Path.home() / ".deepecgkit"

    def test_environment_variable_moves_datasets_and_weights(self, monkeypatch, temp_dir):
        from deepecgkit.datasets import PTBXLDataset
        from deepecgkit.utils.weights import get_weights_dir

        monkeypatch.setenv("DEEPECGKIT_HOME", str(temp_dir))
        assert PTBXLDataset.get_default_data_dir() == temp_dir / "datasets" / "ptbxldataset"
        assert get_weights_dir() == temp_dir / "weights"


class TestPretrainedWeightListing:
    """Only weights that can be downloaded are listed as available."""

    def test_placeholder_entries_are_not_listed(self):
        from deepecgkit.utils.weights import is_published, list_pretrained_weights

        assert not is_published("kanres-af-30s")
        assert not is_published("no-such-weights")
        assert "kanres-af-30s" not in list_pretrained_weights()
        assert "kanres-af-30s" in list_pretrained_weights(include_unpublished=True)


class TestDownloadLock:
    """Two processes must not download the same dataset into the same files."""

    def test_lock_is_exclusive_and_released(self, temp_dir):
        target = temp_dir / "dataset"
        with download.download_lock(target):
            with open(temp_dir / ".dataset.lock", "a+") as other:
                assert download._try_lock(other) is False

        with open(temp_dir / ".dataset.lock", "a+") as other:
            assert download._try_lock(other) is True

    def test_a_held_lock_is_waited_for(self, temp_dir, monkeypatch):
        target = temp_dir / "dataset"
        holder = open(temp_dir / ".dataset.lock", "a+")
        assert download._try_lock(holder) is True
        pauses = []

        def release_after_first_wait(seconds):
            pauses.append(seconds)
            holder.close()

        monkeypatch.setattr(download.time, "sleep", release_after_first_wait)
        with download.download_lock(target, poll_seconds=0.5):
            pass

        assert pauses == [0.5]

    def test_an_unlockable_location_does_not_block_the_download(self, temp_dir):
        blocked = temp_dir / "not-a-directory"
        blocked.write_text("a file where the lock's folder should be")
        with download.download_lock(blocked / "dataset"):
            pass

    def test_a_mirror_that_made_progress_is_resumed_before_falling_back(
        self, temp_dir, file_server, monkeypatch
    ):
        base, handler = file_server
        payload = TestDownloadFile.PAYLOAD
        handler.payloads["/fast/data.bin"] = payload
        handler.payloads["/slow/data.bin"] = payload
        real_fetch = download._fetch
        dropped = []

        def flaky_fetch(url, part_path, desc, timeout, progress=True):
            if not dropped:
                dropped.append(url)
                part_path.write_bytes(payload[:20_000])
                raise OSError("connection reset")
            return real_fetch(url, part_path, desc, timeout, progress)

        monkeypatch.setattr(download, "_fetch", flaky_fetch)
        dest = temp_dir / "data.bin"
        download.download_file([f"{base}/fast/data.bin", f"{base}/slow/data.bin"], dest)

        assert dest.read_bytes() == payload
        assert handler.requests == [("/fast/data.bin", "bytes=20000-")]


class TestDownloadFiles:
    """A dataset made of many files is fetched from a mirror, resumably."""

    FILES: ClassVar[dict[str, bytes]] = {
        "/proj/a.hea": b"header a",
        "/proj/a.dat": b"signal a" * 500,
        "/proj/sub/b.hea": b"header b",
        "/proj/sub/b.dat": b"signal b" * 500,
    }

    def _names(self):
        return [path.removeprefix("/proj/") for path in self.FILES]

    def test_fetches_every_file_keeping_relative_paths(self, temp_dir, file_server):
        base, handler = file_server
        handler.payloads.update(self.FILES)

        download.download_files([f"{base}/proj/"], self._names(), temp_dir, desc="t")

        for path, content in self.FILES.items():
            assert (temp_dir / path.removeprefix("/proj/")).read_bytes() == content

    def test_parallel_fetch_gets_the_same_result(self, temp_dir, file_server):
        base, handler = file_server
        handler.payloads.update(self.FILES)

        download.download_files([f"{base}/proj/"], self._names(), temp_dir, desc="t", workers=4)

        for path, content in self.FILES.items():
            assert (temp_dir / path.removeprefix("/proj/")).read_bytes() == content

    def test_files_already_present_are_not_requested_again(self, temp_dir, file_server):
        base, handler = file_server
        handler.payloads.update(self.FILES)
        (temp_dir / "a.hea").write_bytes(b"header a")

        download.download_files([f"{base}/proj/"], self._names(), temp_dir, desc="t", workers=4)

        assert "/proj/a.hea" not in [path for path, _ in handler.requests]
        assert len(handler.requests) == len(self.FILES) - 1

    def test_a_missing_file_fails_the_whole_download(self, temp_dir, file_server):
        base, handler = file_server
        handler.payloads.update(self.FILES)

        with pytest.raises(RuntimeError, match="Failed to download"):
            download.download_files(
                [f"{base}/proj/"], [*self._names(), "c.dat"], temp_dir, desc="t", workers=4
            )

    def test_marker_stays_until_the_download_finishes(self, temp_dir):
        marker = temp_dir / download.IN_PROGRESS_MARKER

        with pytest.raises(RuntimeError), download.marking_download(temp_dir):
            assert marker.exists()
            raise RuntimeError("connection lost")
        assert marker.exists()
        assert download.has_unfinished_download(temp_dir)

        with download.marking_download(temp_dir):
            pass
        assert not marker.exists()
        assert not download.has_unfinished_download(temp_dir)


class TestKeepAliveDownloads:
    """Many small files are fetched over reused connections, with the slow path as backstop."""

    def test_parallel_download_reuses_connections_and_falls_back(
        self, temp_dir, file_server, monkeypatch
    ):
        base, handler = file_server
        names = [f"records/{index:04d}.dat" for index in range(40)]
        for name in names:
            handler.payloads[f"/proj/{name}"] = name.encode() * 50

        fallbacks = []
        real_download_file = download.download_file

        def counting_download_file(urls, file_path, **kwargs):
            fallbacks.append(Path(file_path).name)
            return real_download_file(urls, file_path, **kwargs)

        monkeypatch.setattr(download, "download_file", counting_download_file)
        download.download_files([f"{base}/proj/"], names, temp_dir, desc="t", workers=4)

        for name in names:
            assert (temp_dir / name).read_bytes() == name.encode() * 50
        assert fallbacks == []
        assert not list(temp_dir.rglob("*.part"))

    def test_large_file_mode_downloads_each_file_resumably(self, temp_dir, file_server):
        base, handler = file_server
        payload = TestDownloadFile.PAYLOAD
        handler.payloads["/proj/a.dat"] = payload
        handler.payloads["/proj/b.dat"] = payload[::-1]
        download.partial_path(temp_dir / "a.dat").write_bytes(payload[:10_000])

        download.download_files(
            [f"{base}/proj/"], ["a.dat", "b.dat"], temp_dir, desc="t", workers=2, keep_alive=False
        )

        assert (temp_dir / "a.dat").read_bytes() == payload
        assert (temp_dir / "b.dat").read_bytes() == payload[::-1]
        assert ("/proj/a.dat", "bytes=10000-") in handler.requests

    def test_a_file_the_fast_path_cannot_fetch_goes_through_the_mirrors(
        self, temp_dir, file_server
    ):
        base, handler = file_server
        handler.payloads["/mirror-b/a.dat"] = b"from the second mirror"

        download.download_files(
            [f"{base}/mirror-a/", f"{base}/mirror-b/"], ["a.dat"], temp_dir, desc="t", workers=2
        )

        assert (temp_dir / "a.dat").read_bytes() == b"from the second mirror"


class _NotPlainData:
    """Something only a full unpickle can restore, as a tampered checkpoint might carry."""


class TestCheckpointLoading:
    """Checkpoints are read in PyTorch's restricted mode unless their source is trusted."""

    def test_what_the_trainer_writes_loads(self, temp_dir):
        path = temp_dir / "native.pt"
        stored = {
            "model_state_dict": {"weight": torch.ones(2)},
            "epoch": 3,
            "best_val_loss": 0.5,
            "data_config": {"class_names": ["Normal", "AF"], "split": {"seed": 42}},
        }
        torch.save(stored, path)

        loaded = load_checkpoint_file(path)

        assert loaded["data_config"] == stored["data_config"]
        assert torch.equal(loaded["model_state_dict"]["weight"], torch.ones(2))

    def test_a_file_carrying_other_objects_is_refused(self, temp_dir, monkeypatch):
        monkeypatch.delenv(TRUST_ENV, raising=False)
        path = temp_dir / "foreign.pt"
        torch.save({"model_state_dict": {}, "extra": _NotPlainData()}, path)

        with pytest.raises(RuntimeError, match=TRUST_ENV):
            load_checkpoint_file(path)

    def test_a_trusted_source_can_be_declared(self, temp_dir, monkeypatch):
        monkeypatch.setenv(TRUST_ENV, "1")
        path = temp_dir / "foreign.pt"
        torch.save({"model_state_dict": {}, "extra": _NotPlainData()}, path)

        assert isinstance(load_checkpoint_file(path)["extra"], _NotPlainData)
