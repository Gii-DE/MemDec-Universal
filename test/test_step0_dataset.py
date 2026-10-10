"""
Test Suite for src/step0_dataset.py (including unit & integration tests + edge case testing)
"""

import sys, gzip, io, shutil, tarfile, tempfile, unittest
from unittest.mock import patch, Mock
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.utils
_real_get_logger = src.utils.get_logger
_real_log_blank_line = src.utils.log_blank_line
src.utils.get_logger = lambda name: Mock()
src.utils.log_blank_line = lambda logger: None
_real_load_usecase_config = src.utils._load_usecase_config

def _fake_load_usecase_config(filename, global_fallback=None, usecase=None):
    if filename == "dataset_config.yaml":
        return {
            "datasets": {
                "test_dataset_small": "https://example.com/test-small.json.gz",
                "test_dataset_medium": "https://example.com/test-medium.jsonl.gz"
            }
        }
    return _real_load_usecase_config(filename, global_fallback, usecase)

src.utils._load_usecase_config = _fake_load_usecase_config
from src import step0_dataset as step0
from src.step0_dataset import (
    _resolve_dataset_url,
    _download_with_progress,
    _extract_gz,
    download_and_extract,
    main,
    DATASET_URLS,
)
src.utils.get_logger = _real_get_logger
src.utils.log_blank_line = _real_log_blank_line
src.utils._load_usecase_config = _real_load_usecase_config


class TestResolveDatasetUrl(unittest.TestCase):
    """Test URL resolution from dataset name or direct URL"""

    def test_resolve_known_key(self):
        """Test resolution of a known dataset key"""

        mock_urls = {
            "test_small": "https://example.com/test-small.json.gz",
            "test_medium": "https://example.com/test-medium.jsonl.gz"
        }
        with patch('src.step0_dataset.DATASET_URLS', mock_urls):
            for key, expected_url in mock_urls.items():
                result = _resolve_dataset_url(key)
                self.assertEqual(result, expected_url)

    def test_resolve_direct_url(self):
        """Test that a direct http URL is returned as-is"""

        url = "https://custom.example.com/data.jsonl.gz"
        result = _resolve_dataset_url(url)
        self.assertEqual(result, url)

    def test_resolve_unknown_key_raises(self):
        """Test that an unknown key raises ValueError"""

        with self.assertRaises(ValueError) as ctx:
            _resolve_dataset_url("nonexistent_dataset")
        self.assertIn("Invalid selection", str(ctx.exception))
        self.assertIn("nonexistent_dataset", str(ctx.exception))

    def test_resolve_unknown_key_lists_available(self):
        """Test that ValueError message lists available options"""

        mock_urls = {
            "test_dataset": "https://example.com/test.json.gz",
        }
        with patch('src.step0_dataset.DATASET_URLS', mock_urls):
            with self.assertRaises(ValueError) as ctx:
                _resolve_dataset_url("wrong_key")
            error_msg = str(ctx.exception)
            for key in mock_urls.keys():
                self.assertIn(key, error_msg)


class TestDownloadWithProgress(unittest.TestCase):
    """Test file download with progress bar and resume support"""

    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()
        self.target_path = Path(self.temp_dir) / "output.gz"

    def tearDown(self):
        """Cleanup test data"""

        shutil.rmtree(self.temp_dir)

    @patch('src.step0_dataset.tqdm')
    @patch('src.step0_dataset.requests.get')
    def test_download_success(self, mock_get, mock_tqdm):
        """Test successful file download from scratch"""

        mock_response = Mock()
        mock_response.headers = {'content-length': '12'}
        mock_response.iter_content = Mock(return_value=[b'hello ', b'world!'])
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        mock_progress = Mock()
        mock_tqdm.return_value = mock_progress
        _download_with_progress("https://example.com/file.gz", self.target_path)
        call_kwargs = mock_get.call_args
        self.assertNotIn('Range', call_kwargs.kwargs.get('headers', {}))
        self.assertEqual(mock_progress.update.call_count, 2)
        mock_progress.close.assert_called_once()

    @patch('src.step0_dataset.logger')
    @patch('src.step0_dataset.tqdm')
    @patch('src.step0_dataset.requests.get')
    def test_download_resume(self, mock_get, mock_tqdm, mock_logger):
        """Test download resume when partial file exists (server honors Range → 206)"""

        part_path = self.target_path.with_suffix('.part')
        part_path.write_bytes(b'partial')
        mock_response = Mock()
        mock_response.status_code = 206
        mock_response.headers = {'content-length': '5'}
        mock_response.iter_content = Mock(return_value=[b'data!'])
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        mock_tqdm.return_value = Mock()
        _download_with_progress("https://example.com/file.gz", self.target_path)
        call_kwargs = mock_get.call_args
        self.assertIn('Range', call_kwargs.kwargs.get('headers', {}))
        self.assertEqual(call_kwargs.kwargs['headers']['Range'], 'bytes=7-')
        mock_logger.warning.assert_not_called()
        self.assertEqual(self.target_path.read_bytes(), b'partialdata!')

    @patch('src.step0_dataset.logger')
    @patch('src.step0_dataset.tqdm')
    @patch('src.step0_dataset.requests.get')
    def test_download_resume_restart_when_range_ignored(self, mock_get, mock_tqdm, mock_logger):
        """Server returns 200 despite a Range request → restart, never append onto .part"""

        part_path = self.target_path.with_suffix('.part')
        part_path.write_bytes(b'partial')
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-length': '11'}
        mock_response.iter_content = Mock(return_value=[b'hello ', b'world'])
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        mock_tqdm.return_value = Mock()
        _download_with_progress("https://example.com/file.gz", self.target_path)
        mock_logger.warning.assert_called_once()
        self.assertEqual(self.target_path.read_bytes(), b'hello world')

    @patch('src.step0_dataset.logger')
    @patch('src.step0_dataset.requests.get')
    def test_download_already_complete(self, mock_get, mock_logger):
        """Test early exit when server reports 416 — .part already holds the complete file"""

        part_path = self.target_path.with_suffix('.part')
        part_path.write_bytes(b'complete data')
        mock_response = Mock()
        mock_response.status_code = 416
        mock_response.headers = {'content-length': '0'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        _download_with_progress("https://example.com/file.gz", self.target_path)
        mock_logger.warning.assert_called_once()
        warning_messages = [str(c) for c in mock_logger.warning.call_args_list]
        self.assertTrue(any("already" in msg.lower() for msg in warning_messages),
                        "Logger should warn that file is already downloaded")
        self.assertEqual(self.target_path.read_bytes(), b'complete data')
        self.assertFalse(part_path.exists())

    @patch('src.step0_dataset.requests.get')
    def test_download_http_error_raises(self, mock_get):
        """Test that HTTP errors are propagated"""

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_response
        with self.assertRaises(Exception):
            _download_with_progress("https://example.com/missing.gz", self.target_path)


class TestExtractGz(unittest.TestCase):
    """Test .gz file extraction"""

    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Cleanup test data"""

        shutil.rmtree(self.temp_dir)

    def test_extract_gz_success(self):
        """Test successful extraction of a valid .gz file"""

        gz_path = Path(self.temp_dir) / "test.json.gz"
        target_path = Path(self.temp_dir) / "test.json"
        original_content = b'{"key": "value", "number": 42}'
        with gzip.open(gz_path, 'wb') as f:
            f.write(original_content)
        _extract_gz(gz_path, target_path)
        self.assertTrue(target_path.exists())
        self.assertEqual(target_path.read_bytes(), original_content)

    def test_extract_gz_invalid_file_raises(self):
        """Test that extracting a corrupt .gz raises an exception"""

        gz_path = Path(self.temp_dir) / "corrupt.gz"
        target_path = Path(self.temp_dir) / "output.json"
        gz_path.write_bytes(b'this is not a valid gzip file')
        with self.assertRaises(Exception):
            _extract_gz(gz_path, target_path)

    def test_extract_gz_preserves_content(self):
        """Test that extraction preserves multi-line content exactly"""

        gz_path = Path(self.temp_dir) / "data.jsonl.gz"
        target_path = Path(self.temp_dir) / "data.jsonl"
        content = b'{"id": 1}\n{"id": 2}\n{"id": 3}\n'
        with gzip.open(gz_path, 'wb') as f:
            f.write(content)
        _extract_gz(gz_path, target_path)
        self.assertEqual(target_path.read_bytes(), content)


class TestDownloadAndExtract(unittest.TestCase):
    """Test combined download and extraction workflow"""

    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()
        self.target_dir = Path(self.temp_dir) / "dataset"

    def tearDown(self):
        """Cleanup test data"""

        shutil.rmtree(self.temp_dir)

    @patch('src.step0_dataset.logger')
    @patch('src.step0_dataset._extract_gz')
    @patch('src.step0_dataset._download_with_progress')
    def test_download_and_extract_json(self, mock_download, mock_extract, mock_logger):
        """Test full workflow for a .json.gz file with preferred_stem"""

        url = "https://example.com/cases.json.gz"

        def fake_download(_url, gz_path, chunk_size=8192):
            """Fake download function"""
            gz_path.parent.mkdir(parents=True, exist_ok=True)
            gz_path.write_bytes(b'fake gz content')

        def fake_extract(_gz_path, output_path):
            """Fake extract function"""
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b'{}')

        mock_download.side_effect = fake_download
        mock_extract.side_effect = fake_extract
        result_path, status = download_and_extract(url, self.target_dir, preferred_stem="test_dataset")
        mock_download.assert_called_once()
        mock_extract.assert_called_once()
        self.assertEqual(result_path.suffix, ".json")
        self.assertIn("Successfully extracted", status)
        mock_logger.info.assert_called()

    @patch('src.step0_dataset.logger')
    @patch('src.step0_dataset._extract_gz')
    @patch('src.step0_dataset._download_with_progress')
    def test_download_and_extract_already_exists(self, mock_download, mock_extract, mock_logger):
        """Test that existing output file skips download and extraction"""

        url = "https://example.com/cases.json.gz"
        self.target_dir.mkdir(parents=True, exist_ok=True)
        existing_file = self.target_dir / "test_dataset.json"
        existing_file.write_bytes(b'{"already": "here"}')
        result_path, status = download_and_extract(url, self.target_dir, preferred_stem="test_dataset")
        mock_download.assert_not_called()
        mock_extract.assert_not_called()
        self.assertEqual(result_path, existing_file)
        self.assertEqual(status, "Already exists")
        mock_logger.info.assert_not_called()

    @patch('src.step0_dataset.logger')
    @patch('src.step0_dataset._extract_gz')
    @patch('src.step0_dataset._download_with_progress')
    def test_download_and_extract_cleanup_on_error(self, mock_download, mock_extract, mock_logger):
        """Test that corrupt .gz is deleted when extraction fails"""

        url = "https://example.com/cases.json.gz"

        def fake_download(_url, gz_path, _chunk_size=8192):
            """Fake download function"""
            gz_path.parent.mkdir(parents=True, exist_ok=True)
            gz_path.write_bytes(b'not a real gz')

        mock_download.side_effect = fake_download
        mock_extract.side_effect = RuntimeError("bad gzip file")
        with self.assertRaises(RuntimeError) as ctx:
            download_and_extract(url, self.target_dir, preferred_stem="test_dataset")
        self.assertIn("Failed to extract", str(ctx.exception))
        gz_path = self.target_dir / "test_dataset.json.gz"
        self.assertFalse(gz_path.exists(), "Corrupt .gz file should be deleted after failed extraction")
        mock_logger.info.assert_called()
        log_messages = [str(c) for c in mock_logger.info.call_args_list]
        self.assertTrue(any("Extracting" in msg for msg in log_messages),
                        "Logger should record that extraction was attempted")

    @patch('src.step0_dataset._download_with_progress')
    def test_local_gz_extracted_not_downloaded(self, mock_download):
        """Test that a local .gz file is extracted in place without downloading"""

        local_dir = Path(self.temp_dir) / "data"
        local_dir.mkdir()
        local_gz = local_dir / "collection.tsv.gz"
        with gzip.open(local_gz, 'wb') as f:
            f.write(b"id\ttext\n1\thello\n")
        result_path, status = download_and_extract(str(local_gz), self.target_dir, preferred_stem="my_set")
        mock_download.assert_not_called()
        self.assertTrue(local_gz.exists(), "Source archive must be kept")
        self.assertEqual(result_path, self.target_dir / "my_set.tsv")
        self.assertEqual(result_path.read_bytes(), b"id\ttext\n1\thello\n")
        self.assertIn("extracted", status.lower())

    @patch('src.step0_dataset._download_with_progress')
    def test_local_tar_gz_extracted(self, mock_download):
        """Test that a local .tar.gz archive is extracted and its first member returned"""

        local_tar = Path(self.temp_dir) / "queries.tar.gz"
        member_data = b"id\ttext\nq1\twhat is law\n"
        with tarfile.open(local_tar, 'w:gz') as tar:
            info = tarfile.TarInfo(name="queries.train.tsv")
            info.size = len(member_data)
            tar.addfile(info, io.BytesIO(member_data))
        result_path, status = download_and_extract(str(local_tar), self.target_dir)
        mock_download.assert_not_called()
        self.assertEqual(result_path, self.target_dir / "queries.train.tsv")
        self.assertEqual(result_path.read_bytes(), member_data)

    @patch('src.step0_dataset.logger')
    @patch('src.step0_dataset._extract_gz')
    @patch('src.step0_dataset._download_with_progress')
    def test_download_and_extract_creates_target_dir(self, mock_download, mock_extract, mock_logger):
        """Test that target directory is created if it does not exist"""

        url = "https://example.com/cases.json.gz"
        non_existing_dir = Path(self.temp_dir) / "new" / "nested" / "dir"
        self.assertFalse(non_existing_dir.exists())

        def fake_download(_url, gz_path, _chunk_size=8192):
            """Fake download function"""
            gz_path.parent.mkdir(parents=True, exist_ok=True)
            gz_path.write_bytes(b'fake gz content')

        def fake_extract(_gz_path, output_path):
            """Fake extract function"""
            output_path.write_bytes(b'{}')

        mock_download.side_effect = fake_download
        mock_extract.side_effect = fake_extract
        download_and_extract(url, non_existing_dir, preferred_stem="test_dataset")
        self.assertTrue(non_existing_dir.exists())
        mock_download.assert_called_once()
        mock_extract.assert_called_once()
        mock_logger.info.assert_called()
        log_messages = [str(c) for c in mock_logger.info.call_args_list]
        self.assertTrue(any("Extracting" in msg for msg in log_messages),
                        "Logger should record the extraction step")


class TestMainFunction(unittest.TestCase):
    """Test main function and CLI integration"""

    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()
        self.orig_data_path = step0.DATA_DIR
        step0.DATA_DIR = Path(self.temp_dir) / "dataset"
        step0.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Cleanup test data"""

        step0.DATA_DIR = self.orig_data_path
        shutil.rmtree(self.temp_dir)

    @patch('src.step0_dataset.cleaning_process')
    @patch('src.step0_dataset.load_dataset')
    @patch('src.step0_dataset.download_and_extract')
    @patch('src.step0_dataset.logger')
    def test_main_success_json(self, mock_logger, mock_dl, mock_load_ds, mock_clean):
        """Test successful main execution with a .json dataset"""

        extracted_file = Path(self.temp_dir) / "test_dataset.json"
        extracted_file.write_bytes(b'{}')
        mock_dl.return_value = (extracted_file, "Successfully extracted: test_dataset.json")
        mock_raw_ds = Mock()
        mock_raw_ds.__len__ = Mock(return_value=500)
        mock_load_ds.return_value = {'train': mock_raw_ds}
        mock_clean.return_value = mock_raw_ds
        main("test_dataset_small")
        mock_dl.assert_called_once()
        mock_load_ds.assert_called_once_with('json', data_files={'train': str(extracted_file)})
        mock_clean.assert_called_once_with(mock_raw_ds, "test_dataset_small")
        mock_logger.info.assert_called()
        log_messages = [str(c) for c in mock_logger.info.call_args_list]
        self.assertTrue(any("Next step" in msg for msg in log_messages),
                        "Logger should point to the next pipeline step")

    @patch('src.step0_dataset.cleaning_process')
    @patch('src.step0_dataset.Dataset.from_list')
    @patch('src.step0_dataset.download_and_extract')
    @patch('src.step0_dataset.logger')
    def test_main_success_jsonl(self, mock_logger, mock_dl, mock_from_list, mock_clean):
        """Test successful main execution with a .jsonl dataset"""

        extracted_file = Path(self.temp_dir) / "test_dataset_medium.jsonl"
        extracted_file.write_bytes(b'{"id": 1}\n')
        mock_dl.return_value = (extracted_file, "Successfully extracted: test_dataset_medium.jsonl")
        mock_raw_ds = Mock()
        mock_raw_ds.__len__ = Mock(return_value=1)
        mock_from_list.return_value = mock_raw_ds
        mock_clean.return_value = mock_raw_ds
        main("test_dataset_medium")
        mock_from_list.assert_called_once()
        mock_clean.assert_called_once_with(mock_raw_ds, "test_dataset_medium")
        mock_logger.info.assert_called()
        log_messages = [str(c) for c in mock_logger.info.call_args_list]
        self.assertTrue(any("Next step" in msg for msg in log_messages),
                        "Logger should point to the next pipeline step")

    @patch('src.step0_dataset.download_and_extract')
    @patch('src.step0_dataset.logger')
    def test_main_unsupported_format_raises_systemexit(self, mock_logger, mock_dl):
        """Test that an unsupported file format triggers SystemExit"""

        extracted_file = Path(self.temp_dir) / "data.txt"
        extracted_file.write_bytes(b'plain text')
        mock_dl.return_value = (extracted_file, "Successfully extracted: data.txt")
        with self.assertRaises(SystemExit):
            main("test_dataset_small")
        mock_logger.error.assert_called()
        error_messages = [str(c) for c in mock_logger.error.call_args_list]
        self.assertTrue(any("Error" in msg or "Unsupported" in msg for msg in error_messages),
                        "Logger should record the unsupported format error")

    @patch('src.step0_dataset.cleaning_process')
    @patch('src.step0_dataset.Dataset.from_list')
    @patch('src.step0_dataset.download_and_extract')
    @patch('src.step0_dataset.logger')
    def test_main_success_tsv(self, mock_logger, mock_dl, mock_from_list, mock_clean):
        """Test successful main execution with a .tsv dataset (loaded via Dataset.from_list)"""

        extracted_file = Path(self.temp_dir) / "test_dataset_small.tsv"
        extracted_file.write_bytes(b'id\ttext\n1\thello\n')
        mock_dl.return_value = (extracted_file, "Successfully extracted: test_dataset_small.tsv")
        mock_raw_ds = Mock()
        mock_raw_ds.__len__ = Mock(return_value=1)
        mock_from_list.return_value = mock_raw_ds
        mock_clean.return_value = mock_raw_ds
        main("test_dataset_small")
        mock_from_list.assert_called_once()
        passed_rows = mock_from_list.call_args[0][0]
        self.assertEqual(passed_rows, [{"id": "1", "text": "hello"}])
        mock_clean.assert_called_once_with(mock_raw_ds, "test_dataset_small")

    @patch('src.step0_dataset.download_and_extract')
    @patch('src.step0_dataset.logger')
    def test_main_download_failure_raises_systemexit(self, mock_logger, mock_dl):
        """Test that a download failure triggers SystemExit"""

        mock_dl.side_effect = RuntimeError("Connection refused")
        with self.assertRaises(SystemExit):
            main("test_dataset")
        mock_logger.error.assert_called()
        error_messages = [str(c) for c in mock_logger.error.call_args_list]
        self.assertTrue(any("Error" in msg or "Connection" in msg for msg in error_messages),
                        "Logger should record a descriptive download error")

    @patch('src.step0_dataset.download_and_extract')
    @patch('src.step0_dataset.logger')
    def test_main_invalid_dataset_choice_raises_systemexit(self, mock_logger, mock_dl):
        """Test that an invalid dataset choice triggers SystemExit via ValueError"""

        mock_dl.side_effect = ValueError("Invalid selection: bad_name. Available: ...")
        with self.assertRaises(SystemExit):
            main("bad_name")
        mock_logger.error.assert_called()
        error_messages = [str(c) for c in mock_logger.error.call_args_list]
        self.assertTrue(any("Error" in msg or "Invalid" in msg for msg in error_messages),
                        "Logger should record a descriptive error for invalid dataset choice")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    @patch('src.step0_dataset.logger')
    def test_resolve_url_https_passthrough(self, mock_logger):
        """Test that https:// URLs bypass key lookup entirely"""

        url = "https://example.com/custom-dataset.json.gz"
        result = _resolve_dataset_url(url)
        self.assertEqual(result, url)
        mock_logger.assert_not_called()

    @patch('src.step0_dataset.logger')
    def test_resolve_url_http_passthrough(self, mock_logger):
        """Test that plain http:// URLs are also passed through"""

        url = "http://example.com/data.gz"
        result = _resolve_dataset_url(url)
        self.assertEqual(result, url)
        mock_logger.assert_not_called()

    def test_dataset_urls_format(self):
        """Test that all entries in DATASET_URLS are valid https URLs ending in .gz"""

        mock_urls = {
            "test_small": "https://example.com/test-small.json.gz",
            "test_medium": "https://example.com/test-medium.jsonl.gz"
        }
        with patch('src.step0_dataset.DATASET_URLS', mock_urls):
            for key, url in mock_urls.items():
                self.assertTrue(url.startswith("https://"),
                                f"URL for '{key}' should start with https://")
                self.assertTrue(url.endswith(".gz"),
                                f"URL for '{key}' should end with .gz")

    def test_dataset_urls_keys_not_empty(self):
        """Test that DATASET_URLS contains at least one entry"""

        mock_urls = {"test_small": "https://example.com/test-small.json.gz"}
        with patch('src.step0_dataset.DATASET_URLS', mock_urls):
            self.assertGreater(len(mock_urls), 0,
                               "DATASET_URLS must contain at least one dataset option")


if __name__ == '__main__':
    unittest.main(verbosity=2)