"""
Test Suite for src/step1_cleaning.py (including unit & integration tests + edge case testing)
"""

import sys, shutil, tempfile, unittest, hashlib
from unittest.mock import patch, Mock, MagicMock, ANY
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import src.utils
_real_get_logger = src.utils.get_logger
_real_log_blank_line = src.utils.log_blank_line
src.utils.get_logger = lambda name: Mock()
src.utils.log_blank_line = lambda logger: None
from src import step1_cleaning as step1
from src.step1_cleaning import (
    clean_html,
    remove_urls_emails,
    fix_encoding_errors,
    normalize_text,
    is_valid_text,
    extract_field,
    cleaning_process,
    get_field_mapping,
    get_dataset_name_from_config,
    main,
)
src.utils.get_logger = _real_get_logger
src.utils.log_blank_line = _real_log_blank_line


class TestCleaningHelpers(unittest.TestCase):
    """Test cleaning helper functions"""

    def test_clean_html_success(self):
        """Test successful HTML cleaning"""

        html = "<p>Hello <b>world</b>!</p> <div>Extra</div>"
        result = clean_html(html)
        self.assertEqual(result, "Hello world! Extra")

    def test_clean_html_invalid(self):
        """Test HTML cleaning with invalid input"""

        invalid = "<invalid>>>" * 1000
        result = clean_html(invalid)
        self.assertEqual(result, invalid)

    def test_remove_urls_emails(self):
        """Test URL and email removal"""

        text = "Visit https://example.com or www.test.de, email@domain.com"
        result = remove_urls_emails(text)
        self.assertEqual(result, "Visit  or , ")

    def test_fix_encoding_errors(self):
        """Test encoding error fixing"""

        text = "\ufffd\x00invalid\u200b chars\x07"
        result = fix_encoding_errors(text)
        self.assertNotIn('\ufffd', result)
        self.assertNotIn('\x00', result)
        self.assertEqual(len(result), 14)

    def test_normalize_text_basic(self):
        """Test basic text normalization"""

        text = "  Multiple   spaces\xa0\u200b with\u200b zero\u200b width "
        result = normalize_text(text)
        self.assertEqual(result, "Multiple spaces with zero width")

    def test_normalize_text_abbreviations(self):
        """Test abbreviation normalization (A. B. C. -> A.B.C.)"""

        text = "See A. B. C. D. E.F. G.H.I."
        result = normalize_text(text)
        self.assertIn("A.B.C.", result)
        self.assertIn("D.E.F.", result)
        self.assertIn("G.H.I.", result)

    def test_normalize_text_numbers(self):
        """Test number patterns like 123 A. B. C.: 456"""

        text = "123 A. B. C. : 456"
        result = normalize_text(text)
        self.assertEqual(result, "123A.B.C.:456")

    def test_is_valid_text_good(self):
        """Test valid text acceptance"""

        text = "This is a valid test sentence with enough alphabetic characters."
        self.assertTrue(is_valid_text(text))

    def test_is_valid_text_too_short(self):
        """Test rejection of too short text"""

        text = "short"
        self.assertFalse(is_valid_text(text))

    def test_is_valid_text_low_alpha_ratio(self):
        """Test rejection of low alphabet ratio"""

        text = "!!!??? 12345 .,. "
        self.assertFalse(is_valid_text(text, min_len=10))

    def test_is_valid_text_high_repetition(self):
        """Test rejection of highly repetitive text"""

        text = "repeat " * 30
        self.assertFalse(is_valid_text(text))

class TestFieldExtraction(unittest.TestCase):
    """Test extract_field and get_field_mapping helpers"""

    def test_extract_field_concatenates_all_mapped_fields(self):
        """All present fields mapped to 'text' are joined with blank lines"""

        mapping = {"text": ["text", "content"]}
        row = {"content": "Inhalt", "text": "Text"}
        self.assertEqual(extract_field(row, mapping, "text"), "Text\n\nInhalt")

    def test_extract_field_falls_back_to_second_key(self):
        """Missing first key falls through to the next candidate"""

        mapping = {"text": ["text", "content"]}
        row = {"content": "Inhalt"}
        self.assertEqual(extract_field(row, mapping, "text"), "Inhalt")

    def test_extract_field_meta_returns_first_match(self):
        """Meta fields return the first non-empty candidate (order as configured)"""

        mapping = {"meta_category": ["category", "court"]}
        row = {"court": "BGH", "category": "Zivil"}
        self.assertEqual(extract_field(row, mapping, "meta_category"), "Zivil")

    def test_extract_field_missing_returns_default(self):
        """No matching key returns the default (None unless overridden)"""

        mapping = {"text": ["text", "content"]}
        self.assertIsNone(extract_field({"a": 1}, mapping, "text"))
        self.assertEqual(extract_field({"a": 1}, mapping, "text", default=""), "")

    def test_extract_field_skips_none_and_empty_values(self):
        """Empty strings and None values do not contribute to concatenation"""

        mapping = {"text": ["a", "b", "c"]}
        row = {"a": "", "b": None, "c": "wert"}
        self.assertEqual(extract_field(row, mapping, "text"), "wert")

    def test_extract_field_matches_whitespace_padded_keys(self):
        """Example keys are stripped before matching mapped field names"""

        mapping = {"text": ["text"]}
        row = {" text ": "padded"}
        self.assertEqual(extract_field(row, mapping, "text"), "padded")

    @patch('src.step1_cleaning._load_usecase_config')
    def test_get_field_mapping_from_local_files(self, mock_load_config):
        """Mapping is read from the local_files section for a known dataset"""

        mock_load_config.return_value = {
            "local_files": {"my_ds": {"field_mapping": {"text": ["body"], "meta": ["src"]}}}
        }
        result = get_field_mapping(dataset_name="my_ds")
        self.assertEqual(result, {"text": ["body"], "meta": ["src"]})

    @patch('src.step1_cleaning._load_usecase_config')
    def test_get_field_mapping_fallback_default(self, mock_load_config):
        """Unknown dataset falls back to default text/meta mapping"""

        mock_load_config.return_value = {}
        result = get_field_mapping(dataset_name="nonexistent_ds")
        self.assertEqual(result["text"], ["markdown_content", "content", "text"])
        self.assertIn("meta_category", result)
        self.assertIn("id", result)


class TestDatasetNameGeneration(unittest.TestCase):
    """Test dataset name generation from HF config"""

    @patch('src.step1_cleaning._load_usecase_config')
    def test_get_dataset_name_from_config_known(self, mock_load_config):
        """Test known configurations"""
        
        mock_load_config.return_value = {
            "huggingface": {
                "dataset_name": "test/dataset",
                "config_mappings": {
                    "config-large": {"output_name": "test_large"},
                    "config-medium": {"output_name": "test_medium"},
                    "config-small": {"output_name": "test_small"}
                },
                "default_config": "config-medium"
            }
        }
        self.assertEqual(get_dataset_name_from_config("config-large"), "test_large")
        self.assertEqual(get_dataset_name_from_config("config-medium"), "test_medium")
        self.assertEqual(get_dataset_name_from_config("config-small"), "test_small")

    def test_get_dataset_name_from_config_unknown(self):
        """Test unknown configuration fallback"""

        self.assertEqual(get_dataset_name_from_config("unknown-config"), "unknown")

class TestCleaningProcess(unittest.TestCase):
    """Test full cleaning process function"""

    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "dataset"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.orig_data_dir = step1.DATA_DIR
        step1.DATA_DIR = self.data_dir

    def tearDown(self):
        """Cleanup test data"""
        step1.DATA_DIR = self.orig_data_dir
        shutil.rmtree(self.temp_dir)

    @patch('src.step1_cleaning.logger')
    @patch('src.step1_cleaning.Dataset.from_list')
    @patch('src.step1_cleaning.load_from_disk')
    @patch('src.step1_cleaning.concatenate_datasets')
    @patch('src.step1_cleaning.shutil.rmtree')
    def test_cleaning_process_small_dataset(
        self, mock_rmtree, mock_concat, mock_load, mock_from_list, mock_logger
    ):
        """Test cleaning process with small valid dataset"""

        mock_example1 = {"content": "<p>Valid text 1</p>", "court": "TestCourt", "date": "2022-01-01"}
        mock_example2 = {"content": "<p>Valid text 2</p>", "id": 123}
        mock_ds = MagicMock()
        mock_ds.__len__ = Mock(return_value=2)
        mock_ds.__getitem__ = Mock(side_effect=lambda i: [mock_example1, mock_example2][i])
        mock_ds.__iter__ = Mock(return_value=iter([mock_example1, mock_example2]))
        mock_chunk_ds = Mock()
        mock_from_list.return_value = mock_chunk_ds
        mock_chunk_ds.save_to_disk = lambda p: Path(p).mkdir(parents=True, exist_ok=True)
        final_ds = Mock()
        final_ds.__len__ = Mock(return_value=2)
        # First load (chunk retrieval) fails -> cleaning runs; second load returns merged dataset
        mock_load.side_effect = [Exception("no chunk yet"), final_ds]
        result = cleaning_process(mock_ds, "test_dataset")
        self.assertIs(result, final_ds)
        mock_logger.info.assert_called()
        mock_concat.assert_not_called()
        mock_rmtree.assert_called_once()
        self.assertIn("chunk_0", str(mock_rmtree.call_args[0][0]))
        

    @patch('src.step1_cleaning.logger')
    def test_cleaning_process_all_invalid(self, mock_logger):
        """Test when all examples are filtered out"""

        mock_example = {"content": "too short"}
        mock_ds = MagicMock()
        mock_ds.__len__ = Mock(return_value=1)
        mock_ds.__getitem__ = Mock(return_value=mock_example)
        mock_ds.__iter__ = Mock(side_effect=[iter([mock_example])])
        with self.assertRaises(RuntimeError) as ctx:
            cleaning_process(mock_ds, "test_empty")
        self.assertIn("No valid entries", str(ctx.exception))
        mock_logger.info.assert_called()

class TestMainFunction(unittest.TestCase):
    """Test main function and CLI integration"""

    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()
        self.orig_data_dir = step1.DATA_DIR
        step1.DATA_DIR = Path(self.temp_dir) / "dataset"
        step1.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Cleanup test data"""

        step1.DATA_DIR = self.orig_data_dir
        shutil.rmtree(self.temp_dir)

    @patch('src.step1_cleaning.load_dataset')
    @patch('src.step1_cleaning.cleaning_process')
    @patch('src.step1_cleaning.logger')
    @patch('src.step1_cleaning._load_usecase_config')
    @patch('argparse.ArgumentParser')
    def test_main_success(self, mock_parser, mock_load_config, mock_logger, mock_cleaning, mock_load_ds):
        """Test successful main execution"""

        mock_load_config.return_value = {
            "huggingface": {
                "dataset_name": "test/dataset",
                "default_config": "test-config"
            }
        }
        mock_ds = Mock()
        mock_ds.__len__ = Mock(return_value=1000)
        mock_load_ds.return_value = {"train": mock_ds}
        mock_cleaning.return_value = mock_ds
        mock_args = Mock()
        mock_args.config = "test-config"
        mock_parser.return_value.parse_args.return_value = mock_args
        main("test-config")
        mock_load_ds.assert_called_once_with("test/dataset", "test-config", split="train", token=ANY)
        mock_cleaning.assert_called_once()
        mock_logger.info.assert_called()

    @patch('argparse.ArgumentParser')
    @patch('src.step1_cleaning.logger')
    def test_main_dataset_load_error(self, mock_logger, mock_parser):
        """Test main function with dataset loading error"""

        mock_args = Mock()
        mock_args.config = "invalid-config"
        mock_parser.return_value.parse_args.return_value = mock_args
        with patch('src.step1_cleaning.load_dataset') as mock_load:
            mock_load.side_effect = Exception("Authentication required")
            with self.assertRaises(Exception):
                main("invalid-config")
        mock_logger.error.assert_called()

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    @patch('src.step1_cleaning.logger')
    def test_normalize_text_empty(self, mock_logger):
        """Test empty string handling"""

        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")
        mock_logger.assert_not_called()

    @patch('src.step1_cleaning.logger')
    def test_hash_collision_detection(self, mock_logger):
        """Test duplicate detection via hashing"""

        text1 = "exact same text"
        text2 = "exact same text"
        h1 = hashlib.blake2b(text1.encode("utf-8"), digest_size=16).hexdigest()
        h2 = hashlib.blake2b(text2.encode("utf-8"), digest_size=16).hexdigest()
        self.assertEqual(h1, h2)
        mock_logger.assert_not_called()

    def test_german_umlaut_handling(self):
        """Test German umlaut preservation"""

        text = "Müllerstraße  ä ö ü Ä Ö Ü ß"
        normalized = normalize_text(text)
        self.assertIn("ä", normalized)
        self.assertIn("ö", normalized)  
        self.assertIn("ü", normalized)
        self.assertIn("Ä", normalized)
        self.assertIn("Ö", normalized)  
        self.assertIn("Ü", normalized)
        self.assertIn("ß", normalized)


if __name__ == '__main__':
    unittest.main(verbosity=2)