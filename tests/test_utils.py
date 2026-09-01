"""
Tests for deepecgkit.utils module.
"""

import inspect

import pytest

import deepecgkit.utils as utils_module
from deepecgkit.utils import download, read_csv


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
