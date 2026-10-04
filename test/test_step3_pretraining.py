"""
Test Suite for src/step3_pretraining.py (including unit & integration tests + edge case testing)
"""

import os, sys, shutil, tempfile, unittest, pyarrow as pa
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.step3_pretraining import (
    PretrainConfig,
    build_datastore_labels,
    _validate_arrow_file,
    create_knn_datastore,
    main
)


class TestPretrainConfig(unittest.TestCase):
    """Test PretrainConfig dataclass"""
    
    def test_default_config_creation(self):
        """Test default values match source config"""

        config = PretrainConfig()
        self.assertEqual(config.batch_size, PretrainConfig.batch_size)
        self.assertEqual(config.seed, PretrainConfig.seed)
        self.assertEqual(config.max_length, PretrainConfig.max_length)
        self.assertEqual(config.ncentroids, PretrainConfig.ncentroids)
        self.assertEqual(config.code_size, PretrainConfig.code_size)
        self.assertEqual(config.probe, 8)
        self.assertIsNone(config.base_model)
    
    def test_custom_config_creation(self):
        """Test custom values override defaults"""

        config = PretrainConfig(
            base_model="test/model",
            batch_size=8,
            max_length=1024,
            ncentroids=4096
        )
        self.assertEqual(config.base_model, "test/model")
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.max_length, 1024)
        self.assertEqual(config.ncentroids, 4096)
    
    def test_config_attributes(self):
        """Test all attributes exist"""

        config = PretrainConfig()
        expected_attrs = [
            'tokenized_data_path', 'knn_datastore_path', 'base_model',
            'batch_size', 'seed', 'max_length',
            'ncentroids', 'code_size', 'probe', 'num_keys_to_add_at_a_time'
        ]
        for attr in expected_attrs:
            self.assertTrue(hasattr(config, attr), f"Missing attribute: {attr}")


class TestBuildDatastoreLabels(unittest.TestCase):
    """Test datastore label construction (only real tokens become entries)"""

    def test_right_padding_masks_padding_and_first_token(self):
        """Right-padded sequence: first real token masked (no real predecessor)"""

        labels = build_datastore_labels([5, 6, 7, 0, 0], [1, 1, 1, 0, 0])
        self.assertEqual(labels, [-100, 6, 7, -100, -100])

    def test_left_padding_masks_first_real_token(self):
        """With left padding, first real token is masked (its key is padding)"""

        labels = build_datastore_labels([0, 0, 5, 6, 7], [0, 0, 1, 1, 1])
        self.assertEqual(labels, [-100, -100, -100, 6, 7])

    def test_no_padding_keeps_all_but_first(self):
        """Full real sequence: first token masked (no preceding key)"""

        labels = build_datastore_labels([5, 6, 7], [1, 1, 1])
        self.assertEqual(labels, [-100, 6, 7])

    def test_all_padding_all_masked(self):
        """All-padding input produces all -100 labels"""

        labels = build_datastore_labels([0, 0, 0], [0, 0, 0])
        self.assertEqual(labels, [-100, -100, -100])


class TestValidateArrowFile(unittest.TestCase):
    """Test Arrow IPC datastore file validation"""

    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Cleanup test data"""

        shutil.rmtree(self.temp_dir)

    def test_missing_file_returns_none(self):
        """Non-existent file returns None"""

        self.assertIsNone(_validate_arrow_file(
            os.path.join(self.temp_dir, "nope.arrow"), 4))

    def test_empty_file_returns_none(self):
        """Zero-byte file returns None"""

        path = os.path.join(self.temp_dir, "empty.arrow")
        open(path, 'wb').close()
        self.assertIsNone(_validate_arrow_file(path, 4))

    def test_missing_eos_marker_returns_none(self):
        """File without EOS sentinel bytes returns None"""

        path = os.path.join(self.temp_dir, "bad.arrow")
        with open(path, 'wb') as f:
            f.write(b'not-terminated-properly-data')
        self.assertIsNone(_validate_arrow_file(path, 4))

    def test_valid_stream_returns_row_count(self):
        """Well-formed Arrow stream with correct schema returns its row count"""

        path = os.path.join(self.temp_dir, "dstore.arrow")
        schema = pa.schema([
            ('keys', pa.list_(pa.float32(), 4)),
            ('vals', pa.int64()),
        ])
        with pa.OSFile(path, 'wb') as sink:
            with pa.ipc.new_stream(sink, schema) as writer:
                batch = pa.record_batch(
                    [[ [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0] ],
                     [10, 11]],
                    schema=schema,
                )
                writer.write_batch(batch)
        self.assertEqual(_validate_arrow_file(path, 4), 2)

    def test_wrong_dimension_returns_none(self):
        """Valid stream with wrong 'keys' fixed-size dimension returns None"""

        path = os.path.join(self.temp_dir, "dstore_wrong_dim.arrow")
        schema = pa.schema([
            ('keys', pa.list_(pa.float32(), 8)),
            ('vals', pa.int64()),
        ])
        with pa.OSFile(path, 'wb') as sink:
            with pa.ipc.new_stream(sink, schema) as writer:
                batch = pa.record_batch(
                    [[[0.0] * 8], [10]],
                    schema=schema,
                )
                writer.write_batch(batch)
        self.assertIsNone(_validate_arrow_file(path, 4))


class TestCreateKNNDatastore(unittest.TestCase):
    """Test KNN datastore creation"""
    
    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()
        self.tokenized_dir = os.path.join(self.temp_dir, "tokenized")
        os.makedirs(os.path.join(self.tokenized_dir, "arrow_data"))
        self.config = PretrainConfig(
            base_model="test/model",
            tokenized_data_path=self.tokenized_dir,
            knn_datastore_path=self.temp_dir,
            batch_size=2
        )
    
    def tearDown(self):
        """Cleanup test data"""

        shutil.rmtree(self.temp_dir)
    
    @patch('src.step3_pretraining.logger')
    @patch('src.step3_pretraining.load_from_disk')
    @patch('src.step3_pretraining.setup_device')
    @patch('src.step3_pretraining.Accelerator')
    @patch('src.step3_pretraining.AutoConfig')
    @patch('src.step3_pretraining.AutoModelForCausalLM')
    @patch('src.step3_pretraining.KNNSaverMulti')
    def test_create_knn_datastore_basic(self, mock_knn_saver, mock_model, mock_config, mock_accelerator, mock_setup_device, mock_load, mock_logger):
        """Test basic datastore creation"""

        os.makedirs(self.config.knn_datastore_path, exist_ok=True)
        mock_setup_device.return_value = MagicMock(type="cpu")
        mock_config_instance = MagicMock()
        mock_config_instance.max_position_embeddings = 2048
        mock_config_instance.sliding_window = 512
        mock_config.from_pretrained.return_value = mock_config_instance
        mock_model_instance = MagicMock()
        mock_model_instance.config.hidden_size = 768
        mock_model_instance.config.max_position_embeddings = 2048
        mock_model_instance.config.sliding_window = 512
        mock_model_instance.device = "cpu"
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_saver_instance = MagicMock()
        mock_saver_instance._get_arrow_file_path.return_value = os.path.join(
            self.temp_dir, "dstore_test_768.arrow")
        mock_knn_saver.return_value = mock_saver_instance
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 2
        mock_dataset.__getitem__.return_value = {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 1]],
        }
        mock_load.return_value = mock_dataset
        create_knn_datastore(self.config)
        mock_setup_device.assert_called_once()
        mock_load.assert_called_once_with(str(Path(self.tokenized_dir).resolve() / "arrow_data"))
        self.assertIs(mock_saver_instance.model, mock_model_instance)
        mock_saver_instance.break_into.assert_called_once_with(mock_model_instance)
        mock_saver_instance.break_out.assert_called_once()
        mock_model_instance.assert_called_once()
        mock_saver_instance.build_index.assert_called_once()
        mock_logger.info.assert_called()
    
    @patch('src.step3_pretraining.logger')
    @patch('src.step3_pretraining.load_from_disk')
    @patch('src.step3_pretraining.setup_device')
    @patch('src.step3_pretraining.Accelerator')
    @patch('src.step3_pretraining.AutoConfig')
    @patch('src.step3_pretraining.AutoModelForCausalLM')
    @patch('src.step3_pretraining.KNNSaverMulti')
    def test_create_knn_datastore_with_model(self, mock_knn_saver, mock_model, mock_config, mock_accelerator, mock_setup_device, mock_load, mock_logger):
        """Test datastore creation with model"""

        mock_setup_device.return_value = MagicMock(type="cpu")
        mock_config_instance = MagicMock()
        mock_config_instance.max_position_embeddings = 2048
        mock_config_instance.sliding_window = 512
        mock_config.from_pretrained.return_value = mock_config_instance
        mock_model_instance = MagicMock()
        mock_model_instance.config.hidden_size = 768
        mock_model_instance.config.max_position_embeddings = 2048
        mock_model_instance.config.sliding_window = 512
        mock_model_instance.device = "cpu"
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_saver_instance = MagicMock()
        mock_saver_instance._get_arrow_file_path.return_value = os.path.join(
            self.temp_dir, "dstore_test_768.arrow")
        mock_knn_saver.return_value = mock_saver_instance
        mock_load.return_value = []
        create_knn_datastore(self.config)
        mock_model.from_pretrained.assert_called_once()
        self.assertEqual(mock_model.from_pretrained.call_args.args[0], "test/model")
        mock_model_instance.eval.assert_called_once()
        mock_knn_saver.assert_called_once()
        self.assertEqual(mock_knn_saver.call_args.kwargs["dimension"], 768)
        self.assertEqual(mock_knn_saver.call_args.kwargs["dstore_dir"], self.config.knn_datastore_path)
        self.assertFalse(mock_knn_saver.call_args.kwargs["knn_gpu"])
        mock_logger.info.assert_called()
    
    @patch('src.step3_pretraining.get_index_path')
    @patch('src.step3_pretraining.logger')
    @patch('src.step3_pretraining.pa')
    @patch('src.step3_pretraining.load_from_disk')
    @patch('src.step3_pretraining.setup_device')
    @patch('src.step3_pretraining.Accelerator')
    @patch('src.step3_pretraining.AutoConfig')
    @patch('src.step3_pretraining.AutoModelForCausalLM')
    @patch('src.step3_pretraining.KNNSaverMulti')
    def test_create_knn_datastore_existing_files(self, mock_knn_saver, mock_model, mock_config, mock_accelerator, mock_setup_device, mock_load, mock_pa, mock_logger, mock_get_index_path):
        """Test with existing canonical datastore + index files (skip recreation)"""

        arrow_file = os.path.join(self.temp_dir, "dstore_test_768.arrow")
        faiss_file = os.path.join(self.temp_dir, "index_test_768.faiss")
        with open(arrow_file, 'wb') as f:
            f.write(b"mockdata" + b'\xff\xff\xff\xff\x00\x00\x00\x00')
        with open(faiss_file, 'wb') as f:
            f.write(b"mock index")
        mock_setup_device.return_value = MagicMock(type="cpu")
        mock_config_instance = MagicMock()
        mock_config_instance.max_position_embeddings = 2048
        mock_config_instance.sliding_window = 512
        mock_config.from_pretrained.return_value = mock_config_instance
        mock_model_instance = MagicMock()
        mock_model_instance.config.hidden_size = 768
        mock_model_instance.config.max_position_embeddings = 2048
        mock_model_instance.config.sliding_window = 512
        mock_model_instance.device = "cpu"
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_saver_instance = MagicMock()
        mock_saver_instance._get_arrow_file_path.return_value = arrow_file
        mock_knn_saver.return_value = mock_saver_instance
        mock_get_index_path.return_value = faiss_file
        mock_key_field = MagicMock()
        mock_key_field.type.list_size = 768
        mock_reader = MagicMock()
        mock_reader.schema.names = ['keys', 'vals']
        mock_reader.schema.field.return_value = mock_key_field
        mock_reader.read_next_batch.side_effect = [MagicMock(num_rows=5), StopIteration]
        mock_pa.ipc.open_stream.return_value = mock_reader
        mock_pa.types.is_fixed_size_list.return_value = True
        create_knn_datastore(self.config)
        mock_pa.OSFile.assert_called_once_with(arrow_file, 'rb')
        mock_pa.ipc.open_stream.assert_called_once_with(mock_pa.OSFile.return_value)
        mock_load.assert_not_called()
        mock_saver_instance.build_index.assert_not_called()
        mock_saver_instance.break_into.assert_not_called()
        mock_saver_instance.break_out.assert_not_called()
        mock_logger.info.assert_called()

    @patch('src.step3_pretraining.get_index_path')
    @patch('src.step3_pretraining.logger')
    @patch('src.step3_pretraining.load_from_disk')
    @patch('src.step3_pretraining.setup_device')
    @patch('src.step3_pretraining.Accelerator')
    @patch('src.step3_pretraining.AutoConfig')
    @patch('src.step3_pretraining.AutoModelForCausalLM')
    @patch('src.step3_pretraining.KNNSaverMulti')
    def test_create_knn_datastore_unlink_failure_raises(self, mock_knn_saver, mock_model, mock_config, mock_accelerator, mock_setup_device, mock_load, mock_logger, mock_get_index_path):
        """Test that OSError propagates when an unusable datastore file cannot be removed"""

        bad_arrow = os.path.join(self.temp_dir, "dstore_test_768.arrow")
        open(bad_arrow, 'wb').close()  # 0-byte file -> fails validation
        mock_setup_device.return_value = MagicMock(type="cpu")
        mock_config_instance = MagicMock()
        mock_config_instance.max_position_embeddings = 2048
        mock_config_instance.sliding_window = 512
        mock_config.from_pretrained.return_value = mock_config_instance
        mock_model_instance = MagicMock()
        mock_model_instance.config.hidden_size = 768
        mock_model_instance.device = "cpu"
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_saver_instance = MagicMock()
        mock_saver_instance._get_arrow_file_path.return_value = bad_arrow
        mock_knn_saver.return_value = mock_saver_instance
        mock_get_index_path.return_value = os.path.join(self.temp_dir, "index_test_768.index")
        with patch('os.unlink', side_effect=OSError("locked by another process")):
            with self.assertRaises(OSError):
                create_knn_datastore(self.config)
        mock_saver_instance.break_into.assert_not_called()
        mock_load.assert_not_called()


class TestMainFunction(unittest.TestCase):
    """Test main function"""
    
    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()
        self.arrow_dir = os.path.join(self.temp_dir, "arrow_data")
        os.makedirs(self.arrow_dir)
        self.config = PretrainConfig(
            base_model="test/model",
            tokenized_data_path=self.temp_dir,
            knn_datastore_path=os.path.join(self.temp_dir, "knowledge_base")
        )
    
    def tearDown(self):
        """Cleanup test data"""

        shutil.rmtree(self.temp_dir)
    
    @patch('src.step3_pretraining.create_knn_datastore')
    @patch('src.step3_pretraining.logger')
    def test_main_success(self, mock_logger, mock_create_knn):
        """Test successful execution"""

        result = main(self.config)
        self.assertEqual(result, 0)
        mock_create_knn.assert_called_once_with(self.config)
        mock_logger.info.assert_called()
    
    @patch('src.step3_pretraining.logger')
    def test_main_missing_arrow_data(self, mock_logger):
        """Test missing arrow data directory"""

        shutil.rmtree(self.arrow_dir)
        result = main(self.config)
        self.assertEqual(result, 1)
        mock_logger.error.assert_called()
    
    @patch('src.step3_pretraining.create_knn_datastore')
    @patch('src.step3_pretraining.logger')
    def test_main_keyboard_interrupt(self, mock_logger, mock_create_knn):
        """Test keyboard interrupt handling"""

        mock_create_knn.side_effect = KeyboardInterrupt()
        result = main(self.config)
        self.assertEqual(result, 1)
        mock_logger.warning.assert_called()
    
    @patch('src.step3_pretraining.create_knn_datastore')
    @patch('src.step3_pretraining.logger')
    def test_main_general_exception(self, mock_logger, mock_create_knn):
        """Test general exception handling"""

        mock_create_knn.side_effect = Exception("Test error")
        result = main(self.config)
        self.assertEqual(result, 1)
        mock_logger.error.assert_called()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_config_with_extreme_values(self):
        """Test config with extreme parameter values"""

        config = PretrainConfig(
            batch_size=0,
            max_length=0,
            ncentroids=0,
            code_size=0
        )
        self.assertIsInstance(config, PretrainConfig)
    
    def test_empty_string_paths(self):
        """Test config with empty string paths"""
        
        config = PretrainConfig(
            tokenized_data_path="",
            knn_datastore_path=""
        )
        self.assertEqual(config.tokenized_data_path, "")
        self.assertEqual(config.knn_datastore_path, "")
    
    def test_device_selection(self):
        """Test device is not stored in config (selected at runtime via setup_device)"""

        config = PretrainConfig()
        self.assertFalse(hasattr(config, 'device'))


if __name__ == '__main__':
    unittest.main(verbosity=2)