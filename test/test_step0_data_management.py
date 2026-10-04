"""
Test Suite for src/step0_data_management.py (including unit & integration tests + edge case testing)
"""

import sys, json, shutil, tempfile, unittest
from unittest.mock import patch, Mock
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.utils
_real_get_logger = src.utils.get_logger
_real_log_blank_line = src.utils.log_blank_line
_real_get_pipeline_value = src.utils.get_pipeline_value
src.utils.get_logger = lambda name: Mock()
src.utils.log_blank_line = lambda logger: None
src.utils.get_pipeline_value = lambda key, default=None: default if default is not None else "test_corpus"
from src import step0_data_management as step0
from src.step0_data_management import (
    load_index,
    save_index,
    process_datasets,
    main,
)
src.utils.get_logger = _real_get_logger
src.utils.log_blank_line = _real_log_blank_line
src.utils.get_pipeline_value = _real_get_pipeline_value


class TestLoadIndex(unittest.TestCase):
    """Test index file loading"""

    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()
        self.index_file = str(Path(self.temp_dir) / "index.json")

    def tearDown(self):
        """Cleanup test data"""

        shutil.rmtree(self.temp_dir)

    def test_load_index_valid_file(self):
        """Test loading a valid index file"""

        data = {"datasets": ["ds1", "ds2"]}
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        result = load_index(self.index_file)
        self.assertEqual(result, data)
        self.assertEqual(result["datasets"], ["ds1", "ds2"])

    @patch('src.step0_data_management.logger')
    def test_load_index_missing_file(self, mock_logger):
        """Test that a missing index file returns default empty index"""

        result = load_index("/nonexistent/path/index.json")
        self.assertEqual(result, {"datasets": []})
        mock_logger.warning.assert_called_once()
        warning_messages = [str(c) for c in mock_logger.warning.call_args_list]
        self.assertTrue(any("Could not load" in msg for msg in warning_messages),
                        "Logger should warn about missing index file")

    @patch('src.step0_data_management.logger')
    def test_load_index_corrupt_file(self, mock_logger):
        """Test that a corrupt JSON file returns default empty index"""

        with open(self.index_file, 'w') as f:
            f.write("{ not valid json !!!")
        result = load_index(self.index_file)
        self.assertEqual(result, {"datasets": []})
        mock_logger.warning.assert_called_once()

    def test_load_index_empty_datasets_list(self):
        """Test loading an index with an empty datasets list"""

        data = {"datasets": []}
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        result = load_index(self.index_file)
        self.assertEqual(result["datasets"], [])


class TestSaveIndex(unittest.TestCase):
    """Test index file saving"""

    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()
        self.index_file = str(Path(self.temp_dir) / "subdir" / "index.json")

    def tearDown(self):
        """Cleanup test data"""

        shutil.rmtree(self.temp_dir)

    @patch('src.step0_data_management.logger')
    def test_save_index_creates_dirs(self, mock_logger):
        """Test that save_index creates missing parent directories"""

        index = {"datasets": ["ds1"]}
        save_index(index, self.index_file)
        self.assertTrue(Path(self.index_file).exists())
        mock_logger.debug.assert_called_once()
        debug_messages = [str(c) for c in mock_logger.debug.call_args_list]
        self.assertTrue(any("saved" in msg.lower() for msg in debug_messages),
                        "Logger should confirm index was saved")

    @patch('src.step0_data_management.logger')
    def test_save_index_content_roundtrip(self, mock_logger):
        """Test that saved content can be read back correctly"""

        index = {"datasets": ["dataset/test_ds1", "dataset/test_ds2"]}
        save_index(index, self.index_file)
        with open(self.index_file, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        self.assertEqual(loaded, index)
        mock_logger.debug.assert_called_once()

    @patch('src.step0_data_management.logger')
    def test_save_index_unicode_preserved(self, mock_logger):
        """Test that Unicode characters (e.g. German umlauts) are preserved"""

        index = {"datasets": ["Ärger/über/Öl"]}
        save_index(index, self.index_file)
        with open(self.index_file, 'r', encoding='utf-8') as f:
            content = f.read()
        self.assertIn("Ärger", content)
        mock_logger.debug.assert_called_once()


class TestProcessDatasets(unittest.TestCase):
    """Test combined dataset processing logic"""

    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()
        self.output_path = str(Path(self.temp_dir) / "combined")
        self.index_file = str(Path(self.temp_dir) / "index.json")
        self.orig_project_root = step0.PROJECT_ROOT
        step0.PROJECT_ROOT = Path(self.temp_dir)

    def tearDown(self):
        """Cleanup test data"""

        step0.PROJECT_ROOT = self.orig_project_root
        shutil.rmtree(self.temp_dir)

    @patch('src.step0_data_management.logger')
    @patch('src.step0_data_management.save_index')
    @patch('src.step0_data_management.load_from_disk')
    @patch('src.step0_data_management.concatenate_datasets')
    def test_process_new_datasets(self, mock_concat, mock_load_disk, mock_save_index, mock_logger):
        """Test processing datasets not yet in the index"""

        ds_path = Path(self.temp_dir) / "dataset" / "test_dataset"
        ds_path.mkdir(parents=True, exist_ok=True)
        rel_path = str(ds_path.relative_to(step0.PROJECT_ROOT)).replace('\\', '/')
        mock_ds = Mock()
        mock_ds.__len__ = Mock(return_value=100)
        mock_ds.save_to_disk = lambda p: Path(p).mkdir(parents=True, exist_ok=True)
        mock_load_disk.return_value = mock_ds
        mock_concat.return_value = mock_ds
        with patch('src.step0_data_management.os.path.exists', return_value=False):
            result = process_datasets([rel_path], self.output_path, self.index_file)
        self.assertIsNotNone(result)
        mock_save_index.assert_called_once()
        mock_logger.info.assert_called()

    @patch('src.step0_data_management.logger')
    @patch('src.step0_data_management.load_from_disk')
    def test_process_already_indexed_dataset(self, mock_load_disk, mock_logger):
        """Test that already-indexed datasets are skipped"""

        index = {"datasets": ["dataset/test_dataset"]}
        with open(self.index_file, 'w') as f:
            json.dump(index, f)
        with patch('src.step0_data_management.os.path.exists', return_value=True):
            existing_ds = Mock()
            existing_ds.__len__ = Mock(return_value=50)
            mock_load_disk.return_value = existing_ds
            result = process_datasets(["dataset/test_dataset"], self.output_path, self.index_file)
        self.assertIsNotNone(result)
        log_messages = [str(c) for c in mock_logger.info.call_args_list]
        self.assertTrue(any("already in index" in msg for msg in log_messages),
                        "Logger should note that dataset was already indexed")

    @patch('src.step0_data_management.logger')
    def test_process_no_datasets_no_existing_output(self, mock_logger):
        """Test that None is returned when no datasets found and no output exists"""

        with patch('src.step0_data_management.os.path.exists', return_value=False):
            result = process_datasets(["dataset/missing"], self.output_path, self.index_file)
        self.assertIsNone(result)
        log_messages = [str(c) for c in mock_logger.info.call_args_list]
        self.assertTrue(any("No datasets found" in msg for msg in log_messages),
                        "Logger should note no datasets were found")

    @patch('src.step0_data_management.logger')
    @patch('src.step0_data_management.load_from_disk')
    def test_process_no_new_datasets_loads_existing(self, mock_load_disk, mock_logger):
        """Test that existing combined dataset is returned when nothing new to add"""

        index = {"datasets": ["dataset/test_dataset"]}
        with open(self.index_file, 'w') as f:
            json.dump(index, f)
        existing_combined = Mock()
        existing_combined.__len__ = Mock(return_value=200)
        mock_load_disk.return_value = existing_combined
        with patch('src.step0_data_management.os.path.exists', return_value=True):
            result = process_datasets(["dataset/test_dataset"], self.output_path, self.index_file)
        self.assertIs(result, existing_combined)
        log_messages = [str(c) for c in mock_logger.info.call_args_list]
        self.assertTrue(any("No new datasets" in msg for msg in log_messages),
                        "Logger should confirm no new datasets were added")

    @patch('src.step0_data_management.logger')
    @patch('src.step0_data_management.load_from_disk')
    def test_process_dataset_not_found_on_disk(self, mock_load_disk, mock_logger):
        """Test that a dataset path that does not exist on disk is skipped"""

        with patch('src.step0_data_management.os.path.exists', return_value=False):
            result = process_datasets(["dataset/nonexistent"], self.output_path, self.index_file)
        self.assertIsNone(result)
        mock_load_disk.assert_not_called()
        warning_messages = [str(c) for c in mock_logger.warning.call_args_list]
        self.assertTrue(any("not found" in msg for msg in warning_messages),
                        "Logger should warn about missing dataset path")

    @patch('src.step0_data_management.logger')
    @patch('src.step0_data_management.save_index')
    @patch('src.step0_data_management.concatenate_datasets')
    @patch('src.step0_data_management.load_from_disk')
    def test_process_combines_multiple_datasets(self, mock_load_disk, mock_concat, mock_save_index, mock_logger):
        """Test that multiple new datasets are concatenated correctly"""

        ds_path1 = Path(self.temp_dir) / "dataset" / "ds1"
        ds_path2 = Path(self.temp_dir) / "dataset" / "ds2"
        ds_path1.mkdir(parents=True, exist_ok=True)
        ds_path2.mkdir(parents=True, exist_ok=True)
        rel1 = str(ds_path1.relative_to(step0.PROJECT_ROOT)).replace('\\', '/')
        rel2 = str(ds_path2.relative_to(step0.PROJECT_ROOT)).replace('\\', '/')
        mock_ds1, mock_combined = Mock(), Mock()
        mock_ds1.__len__ = Mock(return_value=100)
        mock_ds1.save_to_disk = lambda p: Path(p).mkdir(parents=True, exist_ok=True)
        mock_combined.__len__ = Mock(return_value=300)
        mock_combined.save_to_disk = lambda p: Path(p).mkdir(parents=True, exist_ok=True)
        mock_load_disk.return_value = mock_ds1
        mock_concat.return_value = mock_combined
        with patch('src.step0_data_management.os.path.exists', return_value=False):
            result = process_datasets([rel1, rel2], self.output_path, self.index_file)
        self.assertIsNotNone(result)
        mock_concat.assert_called_once()
        mock_save_index.assert_called_once()
        saved_index = mock_save_index.call_args[0][0]
        self.assertIn(rel1, saved_index["datasets"])
        self.assertIn(rel2, saved_index["datasets"])
        mock_logger.info.assert_called()
        log_messages = [str(c) for c in mock_logger.info.call_args_list]
        self.assertTrue(any("Successfully updated" in msg for msg in log_messages),
                        "Logger should confirm combined dataset was updated")


class TestMainFunction(unittest.TestCase):
    """Test main function and CLI integration"""

    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()
        self.orig_data_dir = step0.DATA_DIR
        self.orig_project_root = step0.PROJECT_ROOT
        step0.PROJECT_ROOT = Path(self.temp_dir)
        step0.DATA_DIR = Path(self.temp_dir) / "dataset"
        step0.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Cleanup test data"""

        step0.DATA_DIR = self.orig_data_dir
        step0.PROJECT_ROOT = self.orig_project_root
        shutil.rmtree(self.temp_dir)

    @patch('src.step0_data_management.process_datasets')
    @patch('src.step0_data_management.logger')
    def test_main_success(self, mock_logger, mock_process):
        """Test successful main execution with valid dataset paths"""

        ds_dir = step0.DATA_DIR / "test_dataset"
        ds_dir.mkdir(parents=True, exist_ok=True)
        mock_combined = Mock()
        mock_combined.__len__ = Mock(return_value=500)
        mock_process.return_value = mock_combined
        output_dir = str(Path(self.temp_dir) / "output")
        main(datasets=["test_dataset"], output_dir=output_dir)
        mock_process.assert_called_once()
        mock_logger.info.assert_called()
        log_messages = [str(c) for c in mock_logger.info.call_args_list]
        self.assertTrue(any("Next step" in msg for msg in log_messages),
                        "Logger should point to the next pipeline step")

    @patch('src.step0_data_management.process_datasets')
    @patch('src.step0_data_management.logger')
    def test_main_logs_dataset_list(self, mock_logger, mock_process):
        """Test that main logs each resolved dataset path"""

        ds_dir = step0.DATA_DIR / "test_dataset2"
        ds_dir.mkdir(parents=True, exist_ok=True)
        mock_combined = Mock()
        mock_combined.__len__ = Mock(return_value=1000)
        mock_process.return_value = mock_combined
        output_dir = str(Path(self.temp_dir) / "output")
        main(datasets=["test_dataset2"], output_dir=output_dir)
        log_messages = [str(c) for c in mock_logger.info.call_args_list]
        self.assertTrue(any("test_dataset2" in msg for msg in log_messages),
                        "Logger should list the resolved dataset path")

    @patch('src.step0_data_management.logger')
    def test_main_no_valid_datasets_raises_systemexit(self, mock_logger):
        """Test that missing datasets trigger SystemExit"""

        with self.assertRaises(SystemExit):
            main(datasets=["nonexistent_dataset"], output_dir=str(Path(self.temp_dir) / "output"))
        mock_logger.error.assert_called()
        error_messages = [str(c) for c in mock_logger.error.call_args_list]
        self.assertTrue(any("Error" in msg for msg in error_messages),
                        "Logger should record the error before SystemExit")

    @patch('src.step0_data_management.process_datasets')
    @patch('src.step0_data_management.logger')
    def test_main_process_error_raises_systemexit(self, mock_logger, mock_process):
        """Test that an error in process_datasets triggers SystemExit"""

        ds_dir = step0.DATA_DIR / "test_dataset"
        ds_dir.mkdir(parents=True, exist_ok=True)
        mock_process.side_effect = RuntimeError("Disk full")
        output_dir = str(Path(self.temp_dir) / "output")
        with self.assertRaises(SystemExit):
            main(datasets=["test_dataset"], output_dir=output_dir)
        mock_logger.error.assert_called()
        error_messages = [str(c) for c in mock_logger.error.call_args_list]
        self.assertTrue(any("Error" in msg or "Disk full" in msg for msg in error_messages),
                        "Logger should record the processing error")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()
        self.index_file = str(Path(self.temp_dir) / "index.json")
        self.orig_project_root = step0.PROJECT_ROOT
        step0.PROJECT_ROOT = Path(self.temp_dir)

    def tearDown(self):
        """Cleanup test data"""

        step0.PROJECT_ROOT = self.orig_project_root
        shutil.rmtree(self.temp_dir)

    @patch('src.step0_data_management.logger')
    def test_save_and_load_index_roundtrip(self, mock_logger):
        """Test full save → load roundtrip preserves data exactly"""

        original = {"datasets": ["dataset/a", "dataset/b", "dataset/c"]}
        save_index(original, self.index_file)
        result = load_index(self.index_file)
        self.assertEqual(original, result)
        mock_logger.debug.assert_called_once()
        mock_logger.warning.assert_not_called()

    @patch('src.step0_data_management.logger')
    def test_load_index_returns_default_on_empty_file(self, mock_logger):
        """Test that a completely empty file returns the default index"""

        with open(self.index_file, 'w') as f:
            f.write("")
        result = load_index(self.index_file)
        self.assertEqual(result, {"datasets": []})
        mock_logger.warning.assert_called_once()

    @patch('src.step0_data_management.logger')
    def test_process_datasets_raises_when_load_fails_for_all(self, mock_logger):
        """Test that None is returned and a warning is logged when load_from_disk fails"""

        ds_path = Path(self.temp_dir) / "dataset" / "broken"
        ds_path.mkdir(parents=True, exist_ok=True)
        rel_path = str(ds_path.relative_to(step0.PROJECT_ROOT)).replace('\\', '/')
        with patch('src.step0_data_management.load_from_disk', side_effect=Exception("corrupt")), \
             patch.object(Path, 'exists', return_value=True), \
             patch('src.step0_data_management.os.path.exists', return_value=False):
            result = process_datasets([rel_path], str(Path(self.temp_dir) / "out"),
                                      str(Path(self.temp_dir) / "idx.json"))
        self.assertIsNone(result)
        warning_messages = [str(c) for c in mock_logger.warning.call_args_list]
        self.assertTrue(any("Could not load" in msg for msg in warning_messages),
                        "Logger should warn about each failed dataset load")

    @patch('src.step0_data_management.logger')
    def test_process_datasets_runtime_error_on_corrupt_combined(self, mock_logger):
        """Test RuntimeError when existing combined dataset is unloadable but index lists sources"""

        index = {"datasets": ["dataset/indexed_ds"]}
        with open(self.index_file, 'w') as f:
            json.dump(index, f)
        output_path = str(Path(self.temp_dir) / "combined")
        with patch('src.step0_data_management.os.path.exists', return_value=True), \
             patch('src.step0_data_management.load_from_disk', side_effect=Exception("corrupt")):
            with self.assertRaises(RuntimeError) as ctx:
                process_datasets(["dataset/indexed_ds"], output_path, self.index_file)
        self.assertIn("previously merged", str(ctx.exception))
        mock_logger.error.assert_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)