"""
Test Suite for src/step5_evaluation.py (including unit & integration tests + edge case testing)
"""

import sys, torch, unittest, math
from unittest.mock import patch, MagicMock
from pathlib import Path
from datasets import Dataset as HFDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
import src.utils
from src.step5_evaluation import (
    EvalConfig, resolve_checkpoint_path, _resolve_named_path,
    load_eval_dataset, joint_evaluate, load_models, main
)


class TestEvalConfig(unittest.TestCase):
    """Test EvalConfig dataclass functionality"""

    def test_default_config_creation(self):
        """Test default values match source config"""

        config = EvalConfig()
        self.assertEqual(config.split, EvalConfig.split)
        self.assertEqual(config.max_examples, EvalConfig.max_examples)
        self.assertEqual(config.lmbda, EvalConfig.lmbda)
        self.assertEqual(config.knn_temp, EvalConfig.knn_temp)
        self.assertEqual(config.batch_size, EvalConfig.batch_size)
        self.assertEqual(config.seed, EvalConfig.seed)
        self.assertEqual(config.save_results, EvalConfig.save_results)
        self.assertEqual(config.compare_with_base, EvalConfig.compare_with_base)

    def test_custom_config_creation(self):
        """Test custom values override defaults"""

        config = EvalConfig(
            base_model="custom/model",
            split="train",
            max_examples=100,
            lmbda=0.5,
            batch_size=2,
            save_results=False,
        )
        self.assertEqual(config.base_model, "custom/model")
        self.assertEqual(config.split, "train")
        self.assertEqual(config.max_examples, 100)
        self.assertEqual(config.lmbda, 0.5)
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.save_results, False)

    def test_config_attributes(self):
        """Test all attributes exist"""

        config = EvalConfig()
        expected_attrs = [
            'tokenized_data_path', 'base_model', 'checkpoint_dir',
            'output_dir', 'results_dir', 'split', 'max_examples',
            'lmbda', 'knn_temp', 'batch_size', 'seed',
            'save_results', 'compare_with_base'
        ]
        for attr in expected_attrs:
            self.assertTrue(hasattr(config, attr))


class TestResolveCheckpointPath(unittest.TestCase):
    """Test checkpoint path resolution functionality"""

    def test_none_input(self):
        """Test that None input returns None"""

        result = resolve_checkpoint_path(None, "/some/output")
        self.assertIsNone(result)

    def test_absolute_path_exists(self):
        """Test absolute path that exists"""

        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.is_absolute', return_value=True):
                with patch('pathlib.Path.__str__', return_value="/absolute/path"):
                    result = resolve_checkpoint_path("/absolute/path", "/output")
                    self.assertEqual(result, "/absolute/path")

    def test_absolute_path_not_exists(self):
        """Test absolute path that doesn't exist"""

        with patch('pathlib.Path.exists', return_value=False):
            with patch('pathlib.Path.is_absolute', return_value=True):
                result = resolve_checkpoint_path("/absolute/path", "/output")
                self.assertIsNone(result)

    def test_latest_keyword(self):
        """Test 'latest' keyword resolution"""

        def make_dir(name_str):
            """Create a mock directory"""
            d = MagicMock()
            d.name = name_str
            d.is_dir.return_value = True
            d.__str__ = MagicMock(return_value=f"/output/{name_str}")
            return d

        mock_dirs = [make_dir("step_100"), make_dir("step_200"), make_dir("checkpoint_50")]
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.iterdir', return_value=mock_dirs):
                with patch('src.step5_evaluation.logger'):
                    result = resolve_checkpoint_path("latest", "/output")
                    self.assertIsNotNone(result)
                    self.assertIn("step_200", result)

    def test_latest_no_checkpoints(self):
        """Test 'latest' with no checkpoints found"""

        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.iterdir', return_value=[]):
                result = resolve_checkpoint_path("latest", "/output")
                self.assertIsNone(result)

    def test_relative_to_project_root(self):
        """Test relative path resolved to project root"""

        with patch('src.utils.PROJECT_ROOT', Path("/project")):
            with patch('pathlib.Path.exists', return_value=True):
                result = resolve_checkpoint_path("relative/path", "/output")
                self.assertEqual(result, str(Path("/project/relative/path")))

    def test_relative_to_output_dir(self):
        """Test relative path resolved to output directory"""

        with patch('src.utils.PROJECT_ROOT', Path("/project")):
            with patch('pathlib.Path.exists') as mock_exists:
                mock_exists.side_effect = [False, True]
                result = resolve_checkpoint_path("relative/path", "/output")
                self.assertEqual(result, str(Path("/output/relative/path")))


class TestResolveNamedPath(unittest.TestCase):
    """Test named path resolution functionality (shared resolver for tokenized data)"""

    def test_none_input(self):
        """Test that None input returns None"""

        result = _resolve_named_path(None, src.utils.DATA_DIR, "tokenized-data")
        self.assertIsNone(result)

    def test_absolute_path_exists(self):
        """Test absolute path that exists"""

        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.is_absolute', return_value=True):
                with patch('pathlib.Path.__str__', return_value="/absolute/path"):
                    result = _resolve_named_path("/absolute/path", src.utils.DATA_DIR, "tokenized-data")
                    self.assertEqual(result, "/absolute/path")

    def test_absolute_path_not_exists(self):
        """Test absolute path that doesn't exist"""

        with patch('pathlib.Path.exists', return_value=False):
            with patch('pathlib.Path.is_absolute', return_value=True):
                result = _resolve_named_path("/absolute/path", src.utils.DATA_DIR, "tokenized-data")
                self.assertIsNone(result)

    def test_relative_in_project_root(self):
        """Test relative path found in PROJECT_ROOT (checked first)"""

        with patch('src.utils.DATA_DIR', Path("/data")):
            with patch('src.utils.PROJECT_ROOT', Path("/project")):
                with patch('pathlib.Path.exists') as mock_exists:
                    mock_exists.side_effect = [True, False]
                    result = _resolve_named_path("relative/path", src.utils.DATA_DIR, "tokenized-data")
                    self.assertEqual(result, str(Path("/project/relative/path")))

    def test_relative_in_data_dir(self):
        """Test relative path found in DATA_DIR (fallback after project root)"""

        with patch('src.utils.DATA_DIR', Path("/data")):
            with patch('src.utils.PROJECT_ROOT', Path("/project")):
                with patch('pathlib.Path.exists') as mock_exists:
                    mock_exists.side_effect = [False, True]
                    result = _resolve_named_path("relative/path", src.utils.DATA_DIR, "tokenized-data")
                    self.assertEqual(result, str(Path("/data/relative/path")))

    def test_not_found_anywhere(self):
        """Test path not found anywhere"""

        with patch('src.utils.DATA_DIR', Path("/data")):
            with patch('src.utils.PROJECT_ROOT', Path("/project")):
                with patch('pathlib.Path.exists', return_value=False):
                    result = _resolve_named_path("relative/path", src.utils.DATA_DIR, "tokenized-data")
                    self.assertIsNone(result)


class TestLoadEvalDataset(unittest.TestCase):
    """Test dataset loading functionality"""

    def test_path_not_exists(self):
        """Test FileNotFoundError when path doesn't exist"""

        config = EvalConfig(tokenized_data_path="/nonexistent/path")
        with patch('pathlib.Path.exists', return_value=False):
            with self.assertRaises(FileNotFoundError):
                load_eval_dataset(config)

    @patch('src.step5_evaluation.initialize_tokenizer')
    def test_json_file_loading(self, mock_init_tok):
        """Test loading from JSON files (labels are built dynamically, padding is masked)"""
        
        mock_init_tok.return_value.decode.return_value = "text without separator"
        mock_init_tok.return_value.pad_token_id = 0
        config = EvalConfig(tokenized_data_path="/valid/path", base_model="test/model", split="test")
        with patch('pathlib.Path.exists', side_effect=[True, True]):
            with patch('builtins.open', create=True):
                with patch('json.load', return_value=[
                    {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]},
                    {"input_ids": [3, 0], "attention_mask": [1, 0], "labels": [3, 0]}
                ]):
                    result = load_eval_dataset(config)
                    self.assertIsInstance(result, HFDataset)
                    self.assertEqual(len(result), 2)
                    self.assertEqual(result[0]["labels"], [1, 2])
                    self.assertEqual(result[1]["labels"], [3, -100])
        mock_init_tok.assert_called_once_with("test/model")

    def test_json_file_not_found(self):
        """Test FileNotFoundError when JSON file doesn't exist"""

        config = EvalConfig(tokenized_data_path="/valid/path", split="test")
        with patch('pathlib.Path.exists', side_effect=[True, False]):
            with self.assertRaises(FileNotFoundError) as cm:
                load_eval_dataset(config)
            self.assertIn("JSON file not found", str(cm.exception))

    @patch('src.step5_evaluation.initialize_tokenizer')
    def test_missing_required_columns(self, mock_init_tok):
        """Test KeyError when required columns are missing (labels are no longer required)"""

        config = EvalConfig(tokenized_data_path="/valid/path", split="test")
        with patch('pathlib.Path.exists', side_effect=[True, True]):
            with patch('builtins.open', create=True):
                with patch('json.load', return_value=[
                    {"input_ids": [1, 2], "labels": [1, 2]}
                ]):
                    with self.assertRaises(KeyError) as cm:
                        load_eval_dataset(config)
                    self.assertIn("'attention_mask'", str(cm.exception))

    @patch('src.step5_evaluation.initialize_tokenizer')
    def test_max_examples_limit(self, mock_init_tok):
        """Test dataset is limited when max_examples is set"""

        mock_init_tok.return_value.decode.return_value = "text without separator"
        mock_init_tok.return_value.pad_token_id = 0
        config = EvalConfig(tokenized_data_path="/valid/path", max_examples=2)
        with patch('pathlib.Path.exists', side_effect=[True, True]):
            with patch('builtins.open', create=True):
                with patch('json.load', return_value=[
                    {"input_ids": [1], "attention_mask": [1], "labels": [1]},
                    {"input_ids": [2], "attention_mask": [1], "labels": [2]},
                    {"input_ids": [3], "attention_mask": [1], "labels": [3]},
                    {"input_ids": [4], "attention_mask": [1], "labels": [4]},
                ]):
                    result = load_eval_dataset(config)
                    self.assertEqual(len(result), 2)


class TestJointEvaluate(unittest.TestCase):
    """Test joint evaluation functionality"""

    def test_no_tokens(self):
        """Test when no tokens are available"""

        mock_lm_logits = torch.randn(1, 10, 1000)
        mock_knn_logits = torch.randn(1, 10, 1000)
        mock_batch = {"labels": torch.tensor([[-100] * 10])}
        result = joint_evaluate(
            mock_lm_logits, mock_knn_logits, mock_batch, lmbda=0.3
        )
        self.assertEqual(result[2], 0)

    def test_returns_tuple_and_counts_unmasked(self):
        """Test tuple shape and that unmasked shifted labels are counted"""

        seq_len = 20
        mock_lm_logits = torch.randn(1, seq_len, 1000)
        mock_knn_logits = torch.randn(1, seq_len, 1000)
        mock_batch = {"labels": torch.tensor([[1] * seq_len])}
        result = joint_evaluate(
            mock_lm_logits, mock_knn_logits, mock_batch, lmbda=0.3
        )
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], torch.Tensor)
        self.assertIsInstance(result[1], torch.Tensor)
        self.assertIsInstance(result[2], int)
        self.assertEqual(result[2], seq_len - 1)

    def test_matches_log_softmax_reference(self):
        """Test NLL sums equal the manual log_softmax + logaddexp computation"""

        torch.manual_seed(0)
        lmbda = 0.3
        mock_lm_logits = torch.randn(1, 6, 50)
        mock_knn_logits = torch.randn(1, 6, 50)
        labels = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mock_batch = {"labels": labels}
        joint_nll, lm_nll, cnt = joint_evaluate(
            mock_lm_logits, mock_knn_logits, mock_batch, lmbda
        )
        shift_labels = labels[:, 1:]
        lm_lp  = torch.log_softmax(mock_lm_logits[:, :-1],  dim=-1)
        knn_lp = torch.log_softmax(mock_knn_logits[:, :-1], dim=-1)
        exp_lm = -lm_lp.gather(-1, shift_labels.unsqueeze(-1)).sum()
        exp_joint = -torch.logaddexp(
            lm_lp + math.log(1 - lmbda), knn_lp + math.log(lmbda)
        ).gather(-1, shift_labels.unsqueeze(-1)).sum()
        self.assertAlmostEqual(lm_nll.item(),    exp_lm.item(),    places=4)
        self.assertAlmostEqual(joint_nll.item(), exp_joint.item(), places=4)
        self.assertEqual(cnt, 5)


class TestLoadModels(unittest.TestCase):
    """Test model loading functionality"""

    @patch('src.step5_evaluation.validate_checkpoint_model_type')
    @patch('src.step5_evaluation.AutoConfig')
    @patch('src.step5_evaluation.AutoModelForCausalLM.from_pretrained')
    @patch('src.step5_evaluation.setup_device')
    @patch('src.step5_evaluation.initialize_tokenizer')
    def test_load_models_success(self, mock_tokenizer, mock_device, mock_from_pretrained, mock_auto_config, mock_validate):
        """Test successful model loading"""

        mock_device.return_value = MagicMock(type="cpu")
        mock_base_model = MagicMock()
        mock_knn_model = MagicMock()
        mock_base_model.to = MagicMock(return_value=mock_base_model)
        mock_knn_model.to = MagicMock(return_value=mock_knn_model)
        mock_from_pretrained.side_effect = [mock_base_model, mock_knn_model]
        mock_tokenizer.return_value = MagicMock(__len__=MagicMock(return_value=1000))
        config = EvalConfig(base_model="test/model", checkpoint_dir="test/checkpoint")
        result = load_models(config, mock_tokenizer.return_value)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], mock_base_model)
        self.assertEqual(result[1], mock_knn_model)
        mock_base_model.eval.assert_called_once()
        mock_knn_model.eval.assert_called_once()
        mock_validate.assert_called_once()

    @patch('src.step5_evaluation.validate_checkpoint_model_type')
    @patch('src.step5_evaluation.AutoConfig')
    @patch('src.step5_evaluation.AutoModelForCausalLM.from_pretrained')
    @patch('src.step5_evaluation.setup_device')
    @patch('src.step5_evaluation.initialize_tokenizer')
    def test_load_models_fallback_to_base(self, mock_tokenizer, mock_device, mock_from_pretrained, mock_auto_config, mock_validate):
        """Test that exception is raised when checkpoint loading fails (no silent fallback when checkpoint_dir is set)"""

        mock_device.return_value = MagicMock(type="cpu")
        mock_base_model = MagicMock()
        mock_base_model.to = MagicMock(return_value=mock_base_model)
        mock_from_pretrained.side_effect = [mock_base_model, Exception("Checkpoint failed")]
        mock_tokenizer.return_value = MagicMock(__len__=MagicMock(return_value=1000))
        config = EvalConfig(base_model="test/model", checkpoint_dir="test/checkpoint")
        with self.assertRaises(Exception) as cm:
            load_models(config, mock_tokenizer.return_value)
        self.assertIn("Checkpoint failed", str(cm.exception))

    @patch('src.step5_evaluation.validate_checkpoint_model_type')
    @patch('src.step5_evaluation.AutoConfig')
    @patch('src.step5_evaluation.AutoModelForCausalLM.from_pretrained')
    @patch('src.step5_evaluation.setup_device')
    @patch('src.step5_evaluation.initialize_tokenizer')
    def test_load_models_no_checkpoint(self, mock_tokenizer, mock_device, mock_from_pretrained, mock_auto_config, mock_validate):
        """Test loading without checkpoint (uses base model for both)"""

        mock_device.return_value = MagicMock(type="cpu")
        mock_base_model = MagicMock()
        mock_base_model.to = MagicMock(return_value=mock_base_model)
        mock_from_pretrained.return_value = mock_base_model
        mock_tokenizer.return_value = MagicMock(__len__=MagicMock(return_value=1000))
        config = EvalConfig(base_model="test/model", checkpoint_dir=None)
        result = load_models(config, mock_tokenizer.return_value)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], mock_base_model)
        self.assertEqual(result[1], mock_base_model)
        mock_validate.assert_not_called()


class TestMainFunction(unittest.TestCase):
    """Test main evaluation function"""

    @patch('src.step5_evaluation.torch.cuda.is_available', return_value=False)
    @patch('src.step5_evaluation.gc.collect')
    @patch('src.step5_evaluation.torch.cuda.empty_cache')
    @patch('src.step5_evaluation.load_models')
    @patch('src.step5_evaluation.initialize_tokenizer')
    @patch('src.step5_evaluation.DataLoader')
    @patch('src.step5_evaluation.load_eval_dataset')
    @patch('src.step5_evaluation.tqdm')
    @patch('src.step5_evaluation.joint_evaluate')
    @patch('src.step5_evaluation.log_blank_line')
    @patch('src.step5_evaluation.logger')
    def test_main_success(self, mock_logger, mock_log_blank, mock_joint_eval, mock_tqdm,
                          mock_load_dataset, mock_dataloader, mock_tokenizer, mock_load_models,
                          mock_empty_cache, mock_collect, mock_cuda_available):
        """Test successful main execution - verifies all pipeline steps are called correctly"""

        cpu_device = torch.device("cpu")
        mock_load_dataset.return_value = MagicMock()
        mock_dataloader_instance = MagicMock()
        mock_dataloader_instance.__len__ = MagicMock(return_value=1)
        mock_dataloader.return_value = mock_dataloader_instance
        mock_tokenizer.return_value = MagicMock()
        mock_base_model = MagicMock()
        mock_knn_model  = MagicMock()
        mock_load_models.return_value = (mock_base_model, mock_knn_model)
        mock_param = MagicMock()
        mock_param.device = cpu_device
        mock_base_model.parameters.return_value = iter([mock_param])
        mock_base_model.return_value = MagicMock(logits=torch.randn(1, 3, 100))
        mock_knn_model.return_value  = MagicMock(logits=torch.randn(1, 3, 100))
        mock_batch = {
            "input_ids":      torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "labels":         torch.tensor([[1, 2, 3]]),
        }
        mock_tqdm.return_value = [mock_batch]
        mock_joint_eval.return_value = (torch.tensor(5.0), torch.tensor(6.0), 3)
        mock_log_blank.return_value = None
        mock_logger.info    = MagicMock()
        mock_logger.warning = MagicMock()
        mock_logger.error   = MagicMock()
        mock_logger.debug   = MagicMock()
        mock_cuda_available.return_value = False
        mock_collect.return_value        = None
        mock_empty_cache.return_value    = None

        config = EvalConfig(
            tokenized_data_path="/valid/path",
            base_model="test/model",
            batch_size=1,
            max_examples=None,
            save_results=False,
        )
        result = main(config)
        self.assertIsInstance(result, dict)
        self.assertIn("base_ppl",  result)
        self.assertIn("joint_ppl", result)
        self.assertIn("ppl_delta", result)
        self.assertIn("n_tokens",  result)
        self.assertEqual(result["n_tokens"], 3)
        mock_load_dataset.assert_called_once_with(config)
        mock_dataloader.assert_called_once()
        mock_tokenizer.assert_called_once_with(config.base_model)
        mock_load_models.assert_called_once_with(config, mock_tokenizer.return_value)
        mock_tqdm.assert_called_once()
        mock_joint_eval.assert_called_once()
        mock_log_blank.assert_called()


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""

    @patch('src.step5_evaluation.initialize_tokenizer')
    def test_empty_dataset(self, mock_init_tok):
        """Test handling of empty dataset"""

        config = EvalConfig(tokenized_data_path="/valid/path")
        with patch('pathlib.Path.exists', side_effect=[True, True]):
            with patch('builtins.open', create=True):
                with patch('json.load', return_value=[]):
                    result = load_eval_dataset(config)
                    self.assertEqual(len(result), 0)

    def test_joint_eval_counts_unmasked_only(self):
        """Test masked (-100) label positions are excluded from the token count"""

        mock_lm_logits = torch.randn(1, 5, 1000)
        mock_knn_logits = torch.randn(1, 5, 1000)
        mock_batch = {"labels": torch.tensor([[-100, 1, 2, -100, 3]])}
        result = joint_evaluate(
            mock_lm_logits, mock_knn_logits, mock_batch, lmbda=0.3
        )
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[2], int)
        self.assertEqual(result[2], 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)