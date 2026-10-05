"""
Test Suite for src/step2_tokenization.py (including unit & integration tests + edge case testing)
"""

import os, sys, json, shutil, tempfile, unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.step2_tokenization as step2
from src.step2_tokenization import (
    initialize_tokenizer,
    tokenize_batch,
    load_cleaned_dataset,
    tokenize_dataset,
    save_train_test_json,
    main,
)


class TestTokenizerInitialization(unittest.TestCase):
    """Test tokenizer initialization"""
    
    @patch('src.step2_tokenization.AutoTokenizer')
    def test_initialize_tokenizer_default(self, mock_auto_tok):
        """Test default tokenizer initialization with EOS as pad token"""
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_auto_tok.from_pretrained.return_value = mock_tokenizer
        tokenizer = initialize_tokenizer("gemma2")
        mock_auto_tok.from_pretrained.assert_called_once_with("gemma2", trust_remote_code=True)
        self.assertIs(tokenizer, mock_tokenizer)
        self.assertEqual(tokenizer.pad_token, "</s>")

    @patch('src.step2_tokenization.AutoTokenizer')
    def test_initialize_tokenizer_with_existing_pad(self, mock_auto_tok):
        """Test tokenizer with existing pad token"""
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_auto_tok.from_pretrained.return_value = mock_tokenizer
        tokenizer = initialize_tokenizer("llama")
        mock_auto_tok.from_pretrained.assert_called_once_with("llama", trust_remote_code=True)
        self.assertIs(tokenizer, mock_tokenizer)
        self.assertEqual(tokenizer.pad_token, "[PAD]")


class TestTokenizeBatch(unittest.TestCase):
    """Test per-batch tokenization used by dataset.map"""

    def test_tokenize_batch_adds_labels_and_pads(self):
        """Labels mirror input_ids; sequences padded and truncated to max_length"""

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": [[10, 11]],
            "attention_mask": [[1, 1]],
        }
        batch = {"text": ["some text"]}
        result = tokenize_batch(batch, mock_tokenizer)
        mock_tokenizer.assert_called_once_with(
            ["some text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_attention_mask=True,
        )
        self.assertEqual(result["input_ids"], [[10, 11]])
        self.assertEqual(result["labels"], [[10, 11]])
        self.assertIsNot(result["labels"], result["input_ids"])

    def test_tokenize_batch_respects_custom_max_length(self):
        """Custom max_length is forwarded to the tokenizer"""

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [[1]], "attention_mask": [[1]]}
        tokenize_batch({"text": ["x"]}, mock_tokenizer, max_length=128)
        self.assertEqual(mock_tokenizer.call_args.kwargs["max_length"], 128)

    def test_tokenize_batch_empty(self):
        """Empty batch is tokenized with empty input and returns empty fields"""

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [], "attention_mask": []}
        result = tokenize_batch({"text": []}, mock_tokenizer)
        mock_tokenizer.assert_called_once()
        self.assertEqual(result, {"input_ids": [], "attention_mask": [], "labels": []})


class TestDatasetLoading(unittest.TestCase):
    """Test dataset loading with tests and error handling"""
    
    def setUp(self):
        """Setup test data"""
        
        temp_test_dir = step2.PROJECT_ROOT / "temp_test"
        temp_test_dir.mkdir(exist_ok=True)
        self.temp_dir = tempfile.mkdtemp(dir=temp_test_dir)
        self.dataset_path = os.path.join(self.temp_dir, "test_dataset")
        os.makedirs(self.dataset_path, exist_ok=True)
    
    def tearDown(self):
        """Cleanup test data"""
        
        shutil.rmtree(self.temp_dir)
        temp_test_dir = step2.PROJECT_ROOT / "temp_test"
        if temp_test_dir.exists():
            shutil.rmtree(temp_test_dir)

    @patch('src.step2_tokenization.logger')
    @patch('src.step2_tokenization.load_from_disk')
    def test_load_cleaned_dataset_success(self, mock_load, mock_logger):
        """Test successful dataset loading"""
        
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.__len__.return_value = 3
        mock_load.return_value = mock_dataset
        result = load_cleaned_dataset(self.dataset_path)
        mock_load.assert_called_once_with(self.dataset_path)
        self.assertIs(result, mock_dataset)
        mock_logger.info.assert_called()

    @patch('src.step2_tokenization.logger')
    @patch('src.step2_tokenization.load_from_disk')
    def test_load_cleaned_dataset_missing_text_column(self, mock_load, mock_logger):
        """Test error when dataset lacks required text column"""
        
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["wrong_column"]
        mock_dataset.__len__.return_value = 3
        mock_load.return_value = mock_dataset
        with self.assertRaises(ValueError) as ctx:
            load_cleaned_dataset(self.dataset_path)
        self.assertIn("Dataset must have 'text' column from step1", str(ctx.exception))
        mock_logger.error.assert_called()

    @patch('src.step2_tokenization.logger')
    @patch('src.step2_tokenization.load_from_disk')
    def test_load_cleaned_dataset_general_error(self, mock_load, mock_logger):
        """Test general dataset loading error"""
        
        mock_load.side_effect = Exception("Dataset not found")
        with self.assertRaises(Exception):
            load_cleaned_dataset(self.dataset_path)
        mock_logger.error.assert_called()


class TestTokenization(unittest.TestCase):
    """Test dataset tokenization function"""
    
    def setUp(self):
        """Setup test data"""
        
        self.dataset = MagicMock()
        self.dataset.column_names = ["text"]
        temp_test_dir = step2.PROJECT_ROOT / "temp_test"
        temp_test_dir.mkdir(exist_ok=True)
        self.temp_dir = tempfile.mkdtemp(dir=temp_test_dir)
        self.output_path = Path(self.temp_dir) / "outputs"
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Cleanup test data"""
        
        shutil.rmtree(self.temp_dir)
        temp_test_dir = step2.PROJECT_ROOT / "temp_test"
        if temp_test_dir.exists():
            shutil.rmtree(temp_test_dir)
    
    @patch('src.step2_tokenization.logger')
    def test_tokenize_dataset_success(self, mock_logger):
        """Test successful tokenization"""
        
        mock_tokenizer = MagicMock()
        tokenized = MagicMock()
        self.dataset.map.return_value = tokenized
        result = tokenize_dataset(self.dataset, mock_tokenizer, num_workers=1, batch_size=2)
        self.dataset.map.assert_called_once()
        self.assertIs(result, tokenized)
        mock_logger.info.assert_called()
    
    @patch('src.step2_tokenization.logger')
    def test_tokenize_dataset_uses_defaults(self, mock_logger):
        """Test tokenization with default parameters"""
        
        mock_tokenizer = MagicMock()
        tokenized = MagicMock()
        self.dataset.map.return_value = tokenized
        result = tokenize_dataset(self.dataset, mock_tokenizer)
        call_kwargs = self.dataset.map.call_args.kwargs
        self.assertEqual(call_kwargs["num_proc"], 2)
        self.assertEqual(call_kwargs["batch_size"], 3000)
        self.assertIs(result, tokenized)
        mock_logger.info.assert_called()

    @patch('src.step2_tokenization.logger')
    def test_tokenize_dataset_with_arrow_saving(self, mock_logger):
        """Test tokenization with Arrow format saving"""
        
        mock_dataset = MagicMock()
        mock_tokenizer = MagicMock()
        arrow_dir = self.output_path / "arrow_data"
        self.assertFalse(arrow_dir.exists())
        tokenized_dataset = MagicMock()
        mock_dataset.map.return_value = tokenized_dataset
        result = tokenize_dataset(
            mock_dataset, 
            mock_tokenizer, 
            output_path=self.output_path
        )
        mock_dataset.map.assert_called_once()
        tokenized_dataset.save_to_disk.assert_called_once_with(str(arrow_dir))
        mock_logger.info.assert_called()
        self.assertEqual(result, tokenized_dataset)


class TestDatasetSaving(unittest.TestCase):
    """Test dataset saving with Arrow format and JSON creation"""
    
    def setUp(self):
        """Setup test data"""
        
        temp_test_dir = step2.PROJECT_ROOT / "temp_test"
        temp_test_dir.mkdir(exist_ok=True)
        self.temp_dir = tempfile.mkdtemp(dir=temp_test_dir)
        self.output_path = Path(self.temp_dir) / "outputs"
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Cleanup test data"""
        
        shutil.rmtree(self.temp_dir)
        temp_test_dir = step2.PROJECT_ROOT / "temp_test"
        if temp_test_dir.exists():
            shutil.rmtree(temp_test_dir)

    @patch('src.step2_tokenization.logger')
    def test_save_train_test_json_existing(self, mock_logger):
        """Test skipping JSON creation when train/test files already exist"""
        
        tokenized_dataset = MagicMock()
        tokenized_dataset.train_test_split = MagicMock()
        train_file = self.output_path / "train.json"
        test_file = self.output_path / "test.json"
        state_file = self.output_path / "dataset_metadata.json"
        meta = {
            "train_file": str(train_file),
            "test_file": str(test_file),
            "train_samples": 10,
            "test_samples": 2,
            "total_samples": 12,
            "cleaned_dataset": "dummy",
            "tokenizer_model": "dummy",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        for p in [train_file, test_file, state_file]:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(meta, f)
        tokenizer = MagicMock()
        result = save_train_test_json(
            tokenized_dataset,
            str(self.output_path),
            "cleaned_path",
            tokenizer,
            train_test_split=0.8,
        )
        tokenized_dataset.train_test_split.assert_not_called()
        self.assertEqual(result["train_file"], str(train_file))
        self.assertEqual(result["test_file"], str(test_file))
        mock_logger.info.assert_called()

    @patch('src.step2_tokenization.logger')
    def test_save_train_test_json_creates_files(self, mock_logger):
        """Test creating train/test JSON files with data splitting and metadata"""
        
        tokenized_dataset = MagicMock()
        train_subset = MagicMock()
        test_subset = MagicMock()
        train_subset.__len__.return_value = 3
        test_subset.__len__.return_value = 1
        
        def _getitem_train(_):
            """Mock getitem for train subset"""
            return {
                "input_ids": [[1, 2], [3, 4], [5, 6]],
                "attention_mask": [[1, 1], [1, 1], [1, 1]],
            }
        
        def _getitem_test(_):
            """Mock getitem for test subset"""
            return {
                "input_ids": [[7, 8]],
                "attention_mask": [[1, 1]],
            }
        
        train_subset.__getitem__.side_effect = _getitem_train
        test_subset.__getitem__.side_effect = _getitem_test
        tokenized_dataset.train_test_split.return_value = {
            "train": train_subset,
            "test": test_subset,
        }
        tokenizer = MagicMock()
        tokenizer.name_or_path = "test-tokenizer"
        result = save_train_test_json(
            tokenized_dataset,
            str(self.output_path),
            "cleaned_path",
            tokenizer,
            train_test_split=0.75,
        )
        self.assertTrue((self.output_path / "train.json").exists())
        self.assertTrue((self.output_path / "test.json").exists())
        self.assertTrue((self.output_path / "dataset_metadata.json").exists())
        self.assertEqual(result["train_samples"], 3)
        self.assertEqual(result["test_samples"], 1)
        self.assertFalse(Path(result["train_file"]).is_absolute())
        self.assertFalse(Path(result["test_file"]).is_absolute())
        self.assertNotIn("\\", result["train_file"])
        self.assertTrue(result["train_file"].startswith("temp_test/"))
        self.assertTrue(result["test_file"].startswith("temp_test/"))
        self.assertEqual(result["cleaned_dataset"], "cleaned_path")
        with open(self.output_path / "dataset_metadata.json", "r", encoding="utf-8") as f:
            disk_meta = json.load(f)
        self.assertFalse(Path(disk_meta["train_file"]).is_absolute())
        tokenizer.save_pretrained.assert_called_once()
        mock_logger.info.assert_called()


class TestMainFunction(unittest.TestCase):
    """Test main function"""
    
    def setUp(self):
        """Setup test data"""
        
        temp_test_dir = step2.PROJECT_ROOT / "temp_test"
        temp_test_dir.mkdir(exist_ok=True)
        self.temp_dir = tempfile.mkdtemp(dir=temp_test_dir)
        self.data_dir = Path(self.temp_dir) / "dataset"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._orig_data_dir = None
        self._orig_data_dir = step2.DATA_DIR
        step2.DATA_DIR = self.data_dir
    
    def tearDown(self):
        """Cleanup test data"""
        
        if self._orig_data_dir is not None:
            step2.DATA_DIR = self._orig_data_dir
        shutil.rmtree(self.temp_dir)
        temp_test_dir = step2.PROJECT_ROOT / "temp_test"
        if temp_test_dir.exists():
            shutil.rmtree(temp_test_dir)

    @patch('src.step2_tokenization.cleanup_temp_files')
    @patch('src.step2_tokenization.save_train_test_json')
    @patch('src.step2_tokenization.tokenize_dataset')
    @patch('src.step2_tokenization.initialize_tokenizer')
    @patch('src.step2_tokenization.load_cleaned_dataset')
    @patch('src.step2_tokenization.logger')
    def test_main_success(
        self,
        mock_logger,
        mock_load,
        mock_init_tok,
        mock_tokenize,
        mock_save_json,
        mock_cleanup,
    ):
        """Test successful execution"""
        
        cleaned_name = "test_dataset"
        dataset = MagicMock()
        dataset.__len__.return_value = 1
        mock_load.return_value = dataset
        tokenizer = MagicMock()
        tokenizer.name_or_path = "test-model"
        mock_init_tok.return_value = tokenizer
        tokenized_dataset = MagicMock()
        tokenized_dataset.__len__.return_value = 1
        mock_tokenize.return_value = tokenized_dataset
        mock_save_json.return_value = {
            "train_samples": 1,
            "test_samples": 0,
        }
        main(dataset_cleaned=cleaned_name, base_model="test-model", num_workers=1, train_test_split=0.8)
        mock_load.assert_called_once()
        mock_init_tok.assert_called_once_with("test-model")
        mock_tokenize.assert_called_once()
        mock_save_json.assert_called_once()
        mock_logger.info.assert_called()
        mock_cleanup.assert_called_once()

    @patch('src.step2_tokenization.cleanup_temp_files')
    @patch('src.step2_tokenization.load_cleaned_dataset')
    @patch('src.step2_tokenization.logger')
    def test_main_dataset_error(self, mock_logger, mock_load, mock_cleanup):
        """Test main function with dataset error"""

        mock_load.side_effect = Exception("Dataset not found")
        with self.assertRaises(Exception):
            main(dataset_cleaned="missing", base_model="test-model")
        mock_cleanup.assert_called_once()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    @patch('src.step2_tokenization.logger')
    def test_empty_paths(self, mock_logger):
        """Test empty string paths"""

        with patch('src.step2_tokenization.DATA_DIR', Path('/test')):
            with self.assertRaises(FileNotFoundError):
                main(dataset_cleaned="", base_model="")
        mock_logger.error.assert_called()

    @patch('src.step2_tokenization.logger')
    def test_main_missing_dataset_cleaned_raises_systemexit(self, mock_logger):
        """Test that a missing dataset_cleaned name aborts with SystemExit"""

        with self.assertRaises(SystemExit):
            main(dataset_cleaned=None, base_model="test-model")
        mock_logger.error.assert_called()
        error_messages = [str(c) for c in mock_logger.error.call_args_list]
        self.assertTrue(any("dataset" in msg.lower() for msg in error_messages),
                        "Logger should report the missing dataset name")

    @patch('sys.argv', ['step2_tokenization'])
    @patch('src.step2_tokenization.CORPUS_NAME', None)
    @patch('src.step2_tokenization.get_pipeline_value', lambda key, default=None: default)
    def test_parse_args_missing_dataset_cleaned_errors(self):
        """Test parser.error (SystemExit) when no cleaned dataset can be resolved"""

        from src.step2_tokenization import parse_arguments
        with self.assertRaises(SystemExit):
            parse_arguments()

    @patch('src.step2_tokenization.initialize_tokenizer')
    @patch('src.step2_tokenization.save_train_test_json')
    @patch('src.step2_tokenization.tokenize_dataset')
    @patch('src.step2_tokenization.load_cleaned_dataset')
    @patch('src.step2_tokenization.logger')
    def test_existing_tokenized_data(self, mock_logger, mock_load, mock_tokenize, mock_save_json, mock_init_tok):
        """Test existing tokenized dataset handling"""

        temp_dir = tempfile.mkdtemp()
        orig_data_dir = step2.DATA_DIR
        try:
            data_dir = Path(temp_dir) / "dataset"
            data_dir.mkdir()
            cleaned_dir = data_dir / "existing_tokenized"
            cleaned_dir.mkdir()
            mock_dataset = MagicMock()
            mock_dataset.column_names = ["text"]
            mock_dataset.__len__.return_value = 10
            mock_load.return_value = mock_dataset
            tokenized_dir = data_dir / "existing_tokenized_tokenized-gemma3"
            tokenized_dir.mkdir()
            metadata_file = tokenized_dir / "dataset_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump({"train_samples": 100, "test_samples": 25}, f)
            with patch('src.step2_tokenization.PROJECT_ROOT', Path(temp_dir)):
                step2.DATA_DIR = data_dir
                mock_init_tok.return_value = MagicMock()
                mock_tokenize.return_value = MagicMock()
                mock_save_json.return_value = {"train_samples": 100, "test_samples": 25}
                main(dataset_cleaned="existing_tokenized", base_model="gemma3")
                mock_logger.info.assert_called()
                mock_load.assert_not_called()
                mock_init_tok.assert_not_called()
                mock_tokenize.assert_not_called()
                mock_save_json.assert_not_called()

        finally:
            step2.DATA_DIR = orig_data_dir
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main(verbosity=2)