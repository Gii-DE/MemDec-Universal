"""
Test Suite for src/step4_training.py (including unit & integration tests + edge case testing)
"""

import os, sys, json, shutil, tempfile, unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.step4_training import (
    TrainingConfig, 
    setup_environment, 
    prepare_dataset, 
    setup_training_args,
    validate_dataset,
    main
)


class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig dataclass"""
    
    def test_default_config_creation(self):
        """Test default values match source config"""

        config = TrainingConfig()
        self.assertEqual(config.max_steps, TrainingConfig.max_steps)
        self.assertEqual(config.base_model, TrainingConfig.base_model)
        self.assertEqual(config.batch_size, TrainingConfig.batch_size)
        self.assertEqual(config.learning_rate, TrainingConfig.learning_rate)
        self.assertEqual(config.k_neighbors, TrainingConfig.k_neighbors)
        self.assertIsNone(config.checkpoint_dir)
    
    def test_custom_config_creation(self):
        """Test custom values override defaults"""

        config = TrainingConfig(
            max_steps=1000,
            base_model="custom/model",
            learning_rate=1e-4
        )
        self.assertEqual(config.max_steps, 1000)
        self.assertEqual(config.base_model, "custom/model")
        self.assertEqual(config.learning_rate, 1e-4)
    
    def test_config_attributes(self):
        """Test all attributes exist"""

        config = TrainingConfig()
        expected_attrs = [
            'max_steps', 'tokenized_data_path', 'knn_datastore_path',
            'output_dir', 'base_model', 'checkpoint_dir', 'batch_size',
            'learning_rate', 'weight_decay', 'warmup_steps', 'num_train_epochs',
            'lr_scheduler_type', 'block_size', 'k_neighbors', 'alpha', 'lmbda',
            'gradient_accumulation_steps', 'seed', 'checkpointing_steps',
            'per_device_train_batch_size', 'no_unsloth'
        ]
        for attr in expected_attrs:
            self.assertTrue(hasattr(config, attr), f"Missing attribute: {attr}")


class TestSetupEnvironment(unittest.TestCase):
    """Test environment setup"""
    
    @patch('src.step4_training.logger', create=True)
    def test_setup_environment_basic(self, mock_logger):
        """Test env variables"""

        config = TrainingConfig(
            knn_datastore_path="/test/path",
            k_neighbors=5,
            block_size=1024
        )
        setup_environment(config)
        self.assertEqual(os.environ.get("KNN_DATASTORE_PATH"), "/test/path")
        self.assertEqual(os.environ.get("K_NEIGHBORS"), "5")
        self.assertEqual(os.environ.get("BLOCK_SIZE"), "1024")
        mock_logger.info.assert_called()


class TestPrepareDataset(unittest.TestCase):
    """Test dataset preparation"""
    
    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()
        self.test_dataset_path = os.path.join(self.temp_dir, "test_dataset")
        os.makedirs(self.test_dataset_path)
        self.train_file = os.path.join(self.test_dataset_path, "train.json")
        with open(self.train_file, 'w') as f:
            json.dump([
                {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
                {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]}
            ], f)
    
    def tearDown(self):
        """Cleanup test data"""

        shutil.rmtree(self.temp_dir)
    
    def test_prepare_dataset_success(self):
        """Test successful preparation"""

        config = TrainingConfig(tokenized_data_path=self.test_dataset_path)
        result = prepare_dataset(config)
        self.assertEqual(result, self.train_file)
        self.assertTrue(os.path.exists(self.train_file))
    
    @patch('src.step4_training.logger', create=True)
    def test_prepare_dataset_missing_columns(self, mock_logger):
        """Test missing columns error"""

        with open(self.train_file, 'w') as f:
            json.dump([{"wrong_column": [1, 2, 3]}], f)
        config = TrainingConfig(tokenized_data_path=self.test_dataset_path)
        with self.assertRaises(ValueError) as context:
            prepare_dataset(config)
        self.assertIn("Missing required columns", str(context.exception))
        mock_logger.error.assert_called()
    
    @patch('src.step4_training.logger', create=True)
    def test_prepare_dataset_file_not_found(self, mock_logger):
        """Test file not found error"""

        config = TrainingConfig(tokenized_data_path=os.path.join(self.temp_dir, "non_existent"))
        with self.assertRaises(Exception):
            prepare_dataset(config)
        mock_logger.error.assert_called()


class TestSetupTrainingArgs(unittest.TestCase):
    """Test training arguments setup"""
    
    @patch('src.step4_training.get_knowledge_base_paths')
    @patch('src.step4_training.logger', create=True)
    def test_setup_training_args_basic(self, mock_logger, mock_kb_paths):
        """Test basic arguments"""

        mock_kb_paths.return_value = {'dstore': '/test/dstore', 'index': '/test/index'}
        config = TrainingConfig(
            base_model="test/model",
            learning_rate=1e-4,
            max_steps=100
        )
        train_file = "/test/train.json"
        args = setup_training_args(config, train_file)
        self.assertIn("--model_name_or_path=test/model", args)
        self.assertIn("--train_file=/test/train.json", args)
        self.assertIn("--learning_rate=0.0001", args)
        self.assertIn("--max_train_steps=100", args)
    
    @patch('src.step4_training.get_knowledge_base_paths')
    @patch('src.step4_training.logger', create=True)
    def test_setup_training_args_with_checkpoint(self, mock_logger, mock_kb_paths):
        """Test checkpoint resume"""

        mock_kb_paths.return_value = {'dstore': '/test/dstore', 'index': '/test/index'}
        config = TrainingConfig(
            checkpoint_dir="step_50",
            output_dir="/test/outputs"
        )
        train_file = "/test/train.json"
        with patch('os.path.exists', return_value=True):
            args = setup_training_args(config, train_file)
        self.assertIn("--resume_from_checkpoint", " ".join(args))

    @patch('src.step4_training.get_knowledge_base_paths')
    @patch('src.step4_training.logger', create=True)
    def test_setup_training_args_no_checkpoint(self, mock_logger, mock_kb_paths):
        """Test no checkpoint"""

        mock_kb_paths.return_value = {'dstore': '/test/dstore', 'index': '/test/index'}
        config = TrainingConfig(checkpoint_dir=None)
        train_file = "/test/train.json"
        with patch('os.path.exists', return_value=False):
            args = setup_training_args(config, train_file)
        args_str = " ".join(args)
        self.assertNotIn("--resume_from_checkpoint", args_str)

    @patch('src.step4_training.get_knowledge_base_paths')
    @patch('src.step4_training.logger', create=True)
    def test_setup_training_args_no_unsloth_flag(self, mock_logger, mock_kb_paths):
        """Test --no_unsloth flag is appended only when config.no_unsloth is True"""

        mock_kb_paths.return_value = {'dstore': '/test/dstore', 'index': '/test/index'}
        train_file = "/test/train.json"
        args_on = setup_training_args(TrainingConfig(no_unsloth=True), train_file)
        self.assertIn("--no_unsloth", args_on)
        args_off = setup_training_args(TrainingConfig(no_unsloth=False), train_file)
        self.assertNotIn("--no_unsloth", args_off)

    @patch('src.step4_training.get_knowledge_base_paths')
    @patch('src.step4_training.logger', create=True)
    def test_setup_training_args_num_train_epochs(self, mock_logger, mock_kb_paths):
        """Test custom num_train_epochs is passed through to the training args"""

        mock_kb_paths.return_value = {'dstore': '/test/dstore', 'index': '/test/index'}
        args = setup_training_args(TrainingConfig(num_train_epochs=25), "/test/train.json")
        self.assertIn("--num_train_epochs=25", args)


class TestValidateDataset(unittest.TestCase):
    """Test dataset validation"""
    
    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()
        self.dataset_file = os.path.join(self.temp_dir, "dataset.json")
        with open(self.dataset_file, 'w') as f:
            json.dump([
                {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [4, 5, 6]},
                {"input_ids": [7, 8, 9], "attention_mask": [1, 1, 1], "labels": [10, 11, 12]}
            ], f)
    
    def tearDown(self):
        """Cleanup test data"""

        shutil.rmtree(self.temp_dir)
    
    @patch('src.step4_training.logger', create=True)
    @patch('src.step4_training.load_from_disk')
    def test_validate_dataset_success(self, mock_load, mock_logger):
        """Test successful dataset validation"""

        mock_dataset = MagicMock()
        mock_dataset.column_names = {'input_ids', 'attention_mask', 'labels'}
        mock_dataset.__len__.return_value = 100
        mock_load.return_value = mock_dataset
        result = validate_dataset(self.temp_dir)
        self.assertEqual(result, 100)
        mock_dataset.set_format.assert_called_once()
        mock_logger.info.assert_called()
    
    @patch('src.step4_training.logger', create=True)
    @patch('src.step4_training.load_from_disk')
    def test_validate_dataset_missing_columns(self, mock_load, mock_logger):
        """Test missing columns error"""

        mock_dataset = MagicMock()
        mock_dataset.column_names = {'input_ids', 'attention_mask'}
        mock_load.return_value = mock_dataset
        with self.assertRaises(ValueError) as context:
            validate_dataset(self.temp_dir)
        self.assertIn("Missing required columns", str(context.exception))
        mock_logger.info.assert_called()


class TestMainFunction(unittest.TestCase):
    """Test main function"""
    
    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()
        self.config = TrainingConfig(
            max_steps=1,
            tokenized_data_path=self.temp_dir,
            output_dir=os.path.join(self.temp_dir, "outputs"),
            base_model="gemma3"
        )
        train_file = os.path.join(self.temp_dir, "train.json")
        with open(train_file, 'w') as f:
            json.dump([
                {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [4, 5, 6]}
            ], f)
    
    def tearDown(self):
        """Cleanup test data"""

        shutil.rmtree(self.temp_dir)

    @patch('src.step4_training.get_knowledge_base_paths')
    @patch('src.step4_training.logger', create=True)
    @patch('src.step4_training.train_memdec_main')
    @patch('src.step4_training.parse_args')
    @patch('src.step4_training.tqdm')
    @patch('src.step4_training.gc.collect')
    @patch('src.step4_training.torch.cuda.is_available')
    def test_main_success(self, mock_cuda, mock_gc, mock_tqdm, mock_parse_args, mock_train_main, mock_logger, mock_kb_paths):
        """Test successful execution"""

        mock_kb_paths.return_value = {'dstore': '/test/dstore', 'index': '/test/index'}
        mock_cuda.return_value = False
        mock_args = MagicMock()
        mock_parse_args.return_value = mock_args
        mock_progress_bar = MagicMock()
        mock_tqdm.return_value = mock_progress_bar
        mock_train_main.return_value = None
        try:
            main(self.config)
        except Exception as e:
            self.fail(f"main() raised {e} unexpectedly!")
        mock_parse_args.assert_called_once()
        mock_train_main.assert_called_once_with(progress_bar=mock_progress_bar)
        mock_gc.assert_called()
        mock_logger.info.assert_called()
    
    @patch('src.step4_training.logger', create=True)
    def test_main_dataset_error(self, mock_logger):
        """Test main function with dataset error"""

        config = TrainingConfig(
            tokenized_data_path=os.path.join(self.temp_dir, "non_existent"),
            output_dir=os.path.join(self.temp_dir, "outputs"),
            max_steps=1
        )
        with self.assertRaises(Exception):
            main(config)
        mock_logger.error.assert_called()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_config_with_extreme_values(self):
        """Test config with extreme parameter values"""

        config = TrainingConfig(
            max_steps=0,
            learning_rate=0.0,
            batch_size=-1,
            k_neighbors=0
        )
        self.assertIsInstance(config, TrainingConfig)
    
    def test_empty_string_paths(self):
        """Test config with empty string paths"""
        
        config = TrainingConfig(
            tokenized_data_path="",
            knn_datastore_path="",
            output_dir=""
        )
        self.assertEqual(config.tokenized_data_path, "")
        self.assertEqual(config.knn_datastore_path, "")
        self.assertEqual(config.output_dir, "")


if __name__ == '__main__':
    unittest.main(verbosity=2)