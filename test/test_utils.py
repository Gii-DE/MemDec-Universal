"""
Test Suite for src/utils.py (including unit & integration tests + edge case testing)
"""

import sys, os, json, logging, shutil, tempfile, unittest, torch
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    log_blank_line,
    setup_device,
    cleanup_temp_files,
    get_default_model,
    set_default_model,
    get_model_config,
    get_model_full_name,
    extract_model_identifier,
    get_logger,
    resolve_base_model,
    resolve_checkpoint_path,
    _resolve_checkpoint_or_exit,
    _resolve_named_path,
    project_rel,
    _RelativePathFilter,
    PROJECT_ROOT,
    DATA_DIR,
)

class TestResolveCheckpointOrExit(unittest.TestCase):
    """Test _resolve_checkpoint_or_exit: CLI checkpoint guard with latest-fallback"""

    @patch('src.utils.resolve_checkpoint_path')
    def test_explicit_checkpoint_resolves(self, mock_resolve):
        """A valid --checkpoint value is passed through and returned"""

        mock_resolve.return_value = "/output/step_500"
        result = _resolve_checkpoint_or_exit("step_500", "/output")
        self.assertEqual(result, "/output/step_500")
        mock_resolve.assert_called_once_with("step_500", "/output")

    @patch('src.utils.resolve_checkpoint_path')
    def test_no_checkpoint_falls_back_to_latest(self, mock_resolve):
        """No --checkpoint triggers the 'latest' scan of the output dir"""

        mock_resolve.return_value = "/output/step_1000"
        result = _resolve_checkpoint_or_exit(None, "/output")
        self.assertEqual(result, "/output/step_1000")
        mock_resolve.assert_called_once_with("latest", "/output")

    @patch('src.utils.resolve_checkpoint_path')
    @patch('src.utils.logger')
    def test_explicit_checkpoint_not_found_exits(self, mock_logger, mock_resolve):
        """Unresolvable --checkpoint aborts with SystemExit and names the arg"""

        mock_resolve.return_value = None
        with self.assertRaises(SystemExit):
            _resolve_checkpoint_or_exit("step_999", "/output")
        error_messages = [str(c) for c in mock_logger.error.call_args_list]
        self.assertTrue(any("step_999" in msg for msg in error_messages))

    @patch('src.utils.resolve_checkpoint_path')
    @patch('src.utils.logger')
    def test_no_checkpoint_no_step_dirs_exits(self, mock_logger, mock_resolve):
        """No --checkpoint and no step_* found aborts with a setup hint"""

        mock_resolve.return_value = None
        with self.assertRaises(SystemExit):
            _resolve_checkpoint_or_exit(None, "/output")
        error_messages = [str(c) for c in mock_logger.error.call_args_list]
        self.assertTrue(any("step_*" in msg or "pipeline_config" in msg
                            for msg in error_messages))


class TestLogBlankLine(unittest.TestCase):
    """Test blank line insertion into logger streams"""

    def test_log_blank_line_uses_root_logger_if_none(self):
        """Test that passing None uses the root logger without raising"""

        try:
            log_blank_line(None)
        except Exception as e:
            self.fail(f"log_blank_line(None) raised unexpectedly: {e}")

    def test_log_blank_line_skips_empty_file_handler(self):
        """Test that blank line is skipped for empty log files"""

        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
            log_path = f.name
        try:
            mock_stream = Mock()
            mock_handler = Mock()
            mock_handler.stream = mock_stream
            mock_handler.baseFilename = log_path
            test_logger = logging.getLogger("test_blank_skip_empty")
            test_logger.addHandler(mock_handler)
            with patch('src.utils.os.path.exists', return_value=True), \
                 patch('src.utils.os.path.getsize', return_value=0):
                log_blank_line(test_logger)
            mock_stream.write.assert_not_called()
            test_logger.removeHandler(mock_handler)
        finally:
            Path(log_path).unlink(missing_ok=True)

    def test_log_blank_line_no_handlers(self):
        """Test that log_blank_line handles a logger with no handlers gracefully"""

        empty_logger = logging.getLogger("test_blank_no_handlers")
        empty_logger.handlers.clear()
        try:
            log_blank_line(empty_logger)
        except Exception as e:
            self.fail(f"log_blank_line with no handlers raised unexpectedly: {e}")

class TestSetupDevice(unittest.TestCase):
    """Test device selection for training"""

    @patch('src.utils.torch.cuda.is_available', return_value=False)
    def test_setup_device_cpu_when_no_gpu(self, _mock_cuda):
        """Test that CPU is selected when no GPU is available"""

        device = setup_device()
        self.assertEqual(str(device), "cpu")

    @patch('src.utils.torch.cuda.is_available', return_value=False)
    def test_setup_device_force_cpu(self, _mock_cuda):
        """Test that force_cpu=True always returns CPU device"""

        device = setup_device(force_cpu=True)
        self.assertEqual(str(device), "cpu")

    @patch('src.utils.torch.cuda.device_count', return_value=1)
    @patch('src.utils.torch.cuda.get_device_name', return_value="Tesla T4")
    @patch('src.utils.torch.cuda.get_device_properties')
    @patch('src.utils.torch.cuda.is_available', return_value=True)
    def test_setup_device_gpu_when_available(self, _mock_avail, mock_props, _mock_name, _mock_count):
        """Test that CUDA device is selected when GPU is available"""

        mock_props.return_value.total_memory = 8 * 1024 ** 3
        device = setup_device()
        self.assertEqual(str(device), "cuda")

    @patch('src.utils.torch.cuda.is_available', return_value=True)
    def test_setup_device_force_cpu_overrides_gpu(self, _mock_cuda):
        """Test that force_cpu=True overrides GPU availability"""

        device = setup_device(force_cpu=True)
        self.assertEqual(str(device), "cpu")

class TestGetDefaultModel(unittest.TestCase):
    """Test default model resolution from environment or config file"""

    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Cleanup test data"""

        shutil.rmtree(self.temp_dir)

    @patch('src.utils.os.getenv', return_value="llama")
    def test_get_default_model_from_env(self, _mock_env):
        """Test that environment variable takes priority over config file"""

        result = get_default_model()
        self.assertEqual(result, "llama")

    @patch('src.utils.os.getenv', return_value=None)
    def test_get_default_model_from_config_file(self, _mock_env):
        """Test that model is read from config file when env var is not set"""

        config_file = Path(self.temp_dir) / "model_config.json"
        config_file.write_text(json.dumps({"default_model": "phi"}))
        with patch('src.utils.MODEL_CONFIG_FILE', config_file):
            result = get_default_model()
        self.assertEqual(result, "phi")

    @patch('src.utils.os.getenv', return_value=None)
    def test_get_default_model_fallback(self, _mock_env):
        """Test that fallback returns a valid model when nothing is configured"""

        non_existing = Path(self.temp_dir) / "nonexistent.json"
        with patch('src.utils.MODEL_CONFIG_FILE', non_existing):
            result = get_default_model()
        valid_models = ["gemma3", "qwen3.5", "smollm3"]
        self.assertIn(result, valid_models)

    @patch('src.utils.os.getenv', return_value=None)
    def test_get_default_model_corrupt_config_falls_back(self, _mock_env):
        """Test that a corrupt config file falls back to a valid model"""

        config_file = Path(self.temp_dir) / "model_config.json"
        config_file.write_text("{ invalid json !!!")
        with patch('src.utils.MODEL_CONFIG_FILE', config_file):
            result = get_default_model()
        valid_models = ["gemma3", "qwen3.5", "smollm3"]
        self.assertIn(result, valid_models)

class TestSetDefaultModel(unittest.TestCase):
    """Test persisting the default model to config file"""

    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "model_config.json"

    def tearDown(self):
        """Cleanup test data"""

        shutil.rmtree(self.temp_dir)

    def test_set_default_model_writes_config(self):
        """Test that set_default_model writes the correct model to the config file"""

        with patch('src.utils.MODEL_CONFIG_FILE', self.config_file), \
             patch('src.utils.PROJECT_ROOT', Path(self.temp_dir)):
            set_default_model("gemma3")
        with open(self.config_file, 'r') as f:
            saved = json.load(f)
        self.assertEqual(saved["default_model"], "gemma3")

    def test_set_default_model_normalizes_to_lowercase(self):
        """Test that model name is stored in lowercase regardless of input case"""

        with patch('src.utils.MODEL_CONFIG_FILE', self.config_file), \
             patch('src.utils.PROJECT_ROOT', Path(self.temp_dir)):
            set_default_model("GEMMA3")
        with open(self.config_file, 'r') as f:
            saved = json.load(f)
        self.assertEqual(saved["default_model"], "gemma3")

    def test_set_default_model_invalid_raises_value_error(self):
        """Test that an invalid model name raises ValueError"""

        with self.assertRaises(ValueError) as ctx:
            set_default_model("gpt4")
        self.assertIn("gpt4", str(ctx.exception))

    def test_set_default_model_all_valid_models(self):
        """Test that all supported model names are accepted without error"""

        for model in ["gemma3", "qwen3.5", "smollm3"]:
            with patch('src.utils.MODEL_CONFIG_FILE', self.config_file), \
                 patch('src.utils.PROJECT_ROOT', Path(self.temp_dir)):
                try:
                    set_default_model(model)
                except ValueError:
                    self.fail(f"set_default_model('{model}') raised ValueError unexpectedly")

    def test_set_default_model_updates_env_file(self):
        """Test that MEMDEC_MODEL is written into the project .env file"""

        with patch('src.utils.MODEL_CONFIG_FILE', self.config_file), \
             patch('src.utils.PROJECT_ROOT', Path(self.temp_dir)):
            set_default_model("smollm3")
        env_file = Path(self.temp_dir) / ".env"
        self.assertTrue(env_file.exists())
        self.assertIn("MEMDEC_MODEL='smollm3'", env_file.read_text())

class TestGetModelConfig(unittest.TestCase):
    """Test model configuration lookup by name"""

    def test_get_model_config_gemma(self):
        """Test that gemma3 returns the correct model name and description"""

        config = get_model_config("gemma3")
        self.assertIn("gemma", config["name"].lower())
        self.assertIn("name", config)
        self.assertIn("description", config)

    def test_get_model_config_qwen3_5(self):
        """Test that qwen3.5 returns the correct model name and description"""

        config = get_model_config("qwen3.5")
        self.assertIn("qwen", config["name"].lower())

    def test_get_model_config_smollm3(self):
        """Test that smollm3 returns the correct model name and description"""

        config = get_model_config("smollm3")
        self.assertIn("smollm", config["name"].lower())

    def test_get_model_config_qwen(self):
        """Test that qwen returns the correct model name and description"""

        config = get_model_config("qwen3.5")
        self.assertIn("qwen", config["name"].lower())

    def test_get_model_config_invalid_raises_value_error(self):
        """Test that an unsupported model name raises ValueError"""

        with self.assertRaises(ValueError) as ctx:
            get_model_config("gpt4")
        self.assertIn("gpt4", str(ctx.exception))

    def test_get_model_config_case_insensitive(self):
        """Test that model name lookup is case-insensitive"""

        config_lower = get_model_config("gemma3")
        config_upper = get_model_config("GEMMA3")
        self.assertEqual(config_lower, config_upper)

    @patch('src.utils.get_default_model', return_value="gemma3")
    def test_get_model_config_uses_default_when_none(self, _mock_default):
        """Test that passing None as model_choice falls back to the default model"""

        config = get_model_config(None)
        self.assertIn("name", config)
        self.assertIn("gemma", config["name"].lower())
        _mock_default.assert_called_once()

class TestCleanupTempFiles(unittest.TestCase):
    """Test cleanup of temporary and cache files"""

    def setUp(self):
        """Setup test data"""

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Cleanup test data"""

        shutil.rmtree(self.temp_dir)

    def test_cleanup_removes_cache_files(self):
        """Test that cache-* files in dataset directories are deleted"""

        dataset_dir = Path(self.temp_dir) / "dataset" / "test_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "cache-abc123.arrow"
        cache_file.write_bytes(b"fake cache")
        with patch('src.utils.Path') as mock_path_cls, \
             patch('src.utils.shutil.rmtree'):
            def path_side_effect(arg):
                if arg == 'dataset':
                    return Path(self.temp_dir) / "dataset"
                return Path(arg)
            mock_path_cls.side_effect = path_side_effect
            mock_path_cls.home = Path.home
            count = cleanup_temp_files(logger=None)
        self.assertGreaterEqual(count, 0)

    def test_cleanup_returns_zero_when_no_dataset_dir(self):
        """Test that cleanup returns 0 when dataset directory does not exist"""

        with patch('src.utils.Path') as mock_path_cls, \
             patch('src.utils.shutil.rmtree'):
            def path_side_effect(arg):
                if arg == 'dataset':
                    return Path(self.temp_dir) / "nonexistent_dataset"
                return Path(arg)
            mock_path_cls.side_effect = path_side_effect
            mock_path_cls.home = Path.home
            count = cleanup_temp_files(logger=None)
        self.assertEqual(count, 0)

    def test_cleanup_logs_when_files_cleaned(self):
        """Test that cleanup logs a message when cache files are removed"""

        mock_logger = Mock()
        dataset_dir = Path(self.temp_dir) / "dataset" / "test_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        cache_file = dataset_dir / "cache-xyz.arrow"
        cache_file.write_bytes(b"data")
        with patch('src.utils.Path') as mock_path_cls, \
             patch('src.utils.shutil.rmtree'):
            def path_side_effect(arg):
                if arg == 'dataset':
                    return Path(self.temp_dir) / "dataset"
                return Path(arg)
            mock_path_cls.side_effect = path_side_effect
            mock_path_cls.home = Path.home
            count = cleanup_temp_files(logger=mock_logger)
        if count > 0:
            mock_logger.info.assert_called()

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def test_get_model_config_all_have_name_and_description(self):
        """Test that all valid models return dicts with 'name' and 'description' keys"""

        for model in ["gemma3", "qwen3.5", "smollm3"]:
            config = get_model_config(model)
            self.assertIn("name", config, f"Missing 'name' for model '{model}'")
            self.assertIn("description", config, f"Missing 'description' for model '{model}'")

    def test_set_and_get_default_model_roundtrip(self):
        """Test full set → get roundtrip preserves model name correctly"""

        temp_dir = tempfile.mkdtemp()
        try:
            config_file = Path(temp_dir) / "model_config.json"
            with patch('src.utils.MODEL_CONFIG_FILE', config_file), \
                 patch('src.utils.PROJECT_ROOT', Path(temp_dir)), \
                 patch('src.utils.os.getenv', return_value=None):
                set_default_model("qwen3.5")
                result = get_default_model()
            self.assertEqual(result, "qwen3.5")
        finally:
            shutil.rmtree(temp_dir)

    @patch('src.utils.torch.cuda.is_available', return_value=False)
    def test_setup_device_returns_torch_device(self, _mock_cuda):
        """Test that setup_device always returns a torch.device instance"""

        device = setup_device()
        self.assertIsInstance(device, torch.device)


class TestExtractModelIdentifier(unittest.TestCase):
    """Test keyword extraction from model names, variants and paths"""

    def test_exact_key_match(self):
        """Exact config keys resolve to themselves"""

        self.assertEqual(extract_model_identifier("gemma3"), "gemma3")
        self.assertEqual(extract_model_identifier("gemma3-1b"), "gemma3-1b")
        self.assertEqual(extract_model_identifier("qwen3.5-2b"), "qwen3.5-2b")

    def test_full_hf_name_match(self):
        """Full HuggingFace names resolve to their config key"""

        self.assertEqual(extract_model_identifier("unsloth/gemma-3-270m-it"), "gemma3")
        self.assertEqual(extract_model_identifier("unsloth/gemma-3-1b-it"), "gemma3-1b")
        self.assertEqual(extract_model_identifier("unsloth/SmolLM3-3B"), "smollm3")

    def test_normalized_variant_resolves_specific_key(self):
        """HF-style variants without org prefix resolve to the specific key, not the family"""

        # Regression: 'gemma-3-1b-it' previously fell through to 'gemma3'
        self.assertEqual(extract_model_identifier("gemma-3-1b-it"), "gemma3-1b")
        self.assertEqual(extract_model_identifier("qwen-3.5-2b"), "qwen3.5-2b")
        self.assertEqual(extract_model_identifier("SmolLM2-1.7B-Instruct"), "smollm2-1.7b")

    def test_family_key_still_resolves(self):
        """Family-level inputs still resolve to the family key"""

        self.assertEqual(extract_model_identifier("gemma-3-270m-it"), "gemma3")
        self.assertEqual(extract_model_identifier("unsloth/Qwen3.5-0.8B"), "qwen3.5")

    def test_unknown_returns_none_when_no_default(self):
        """Unresolvable input returns None when return_default=False"""

        self.assertIsNone(extract_model_identifier("totally-unrelated-xyz", return_default=False))

    def test_get_model_full_name_variant(self):
        """get_model_full_name resolves normalized variants to the canonical HF name"""

        self.assertEqual(get_model_full_name("gemma-3-1b-it"), "unsloth/gemma-3-1b-it")
        self.assertEqual(get_model_full_name("gemma3"), "unsloth/gemma-3-270m-it")

    def test_get_model_full_name_unknown_warns_and_falls_back(self):
        """Unmatched input (e.g. typo 'mollm2-1.7b') warns and falls back to the default model"""

        with patch('src.utils.logger') as mock_logger:
            result = get_model_full_name("mollm2-1.7b")
        self.assertEqual(result, get_model_config(get_default_model())["name"])
        mock_logger.warning.assert_called_once()
        self.assertIn("mollm2-1.7b", mock_logger.warning.call_args.args[0])


class TestResolveBaseModel(unittest.TestCase):
    """Test the unified base model resolution hierarchy:
    .env MEMDEC_MODEL > CLI > steps.<step>.base_model > models.default_model
    > auto-derived (tokenized data / checkpoint) > system default 'gemma3'
    """

    def test_env_overrides_everything(self):
        """.env MEMDEC_MODEL is the global user setting and must win over CLI,
        step config, pipeline default and auto-derived sources"""

        with patch.dict(os.environ, {"MEMDEC_MODEL": "smollm3"}), \
             patch('src.utils.get_pipeline_value') as mock_get_pipeline:
            result = resolve_base_model(
                cli_model="gemma3-1b",
                step_config_key="steps.step5_evaluation",
                tokenized_data="legal_corpus_tokenized-gemma3",
                checkpoint="step_500",
                output_dir="/outputs",
            )
        self.assertEqual(result, "unsloth/SmolLM3-3B")
        mock_get_pipeline.assert_not_called()

    @patch('src.utils.get_pipeline_value')
    def test_cli_overrides_config_when_env_unset(self, mock_get_pipeline):
        """CLI --model wins over step config and pipeline default once .env is unset"""

        env = {k: v for k, v in os.environ.items() if k != "MEMDEC_MODEL"}
        with patch.dict(os.environ, env, clear=True):
            result = resolve_base_model(
                cli_model="gemma3",
                step_config_key="steps.step5_evaluation",
                tokenized_data="legal_corpus_tokenized-smollm3",
            )
        self.assertEqual(result, "unsloth/gemma-3-270m-it")
        mock_get_pipeline.assert_not_called()

    @patch('src.utils.get_pipeline_value')
    def test_step_config_overrides_pipeline_default(self, mock_get_pipeline):
        """steps.<step>.base_model must win over pipeline.models.default_model
        and auto-derived sources once .env and CLI are unset"""

        def side_effect(key, default=None, usecase=None):
            if key == "steps.step5_evaluation.base_model":
                return "gemma3-1b"
            if key == "pipeline.models.default_model":
                return "smollm3"
            return default
        mock_get_pipeline.side_effect = side_effect
        env = {k: v for k, v in os.environ.items() if k != "MEMDEC_MODEL"}
        with patch.dict(os.environ, env, clear=True):
            result = resolve_base_model(
                step_config_key="steps.step5_evaluation",
                tokenized_data="legal_corpus_tokenized-smollm3",
            )
        self.assertEqual(result, "unsloth/gemma-3-1b-it")
        # the pipeline default is never consulted once the step config resolves
        mock_get_pipeline.assert_called_once_with("steps.step5_evaluation.base_model", None)

    @patch('src.utils._resolve_named_path')
    @patch('src.utils.get_dataset_tokenizer_model')
    @patch('src.utils.get_pipeline_value')
    def test_step_config_overrides_auto_derivation(self, mock_get_pipeline, mock_dataset_tokenizer, mock_resolve_tokenized):
        """steps.<step>.base_model must win over auto-derived sources (and over
        pipeline.models.default_model) once .env and CLI are unset"""

        def side_effect(key, default=None, usecase=None):
            if key == "steps.step5_evaluation.base_model":
                return "gemma3-1b"
            if key == "pipeline.models.default_model":
                return "smollm3"
            return default
        mock_get_pipeline.side_effect = side_effect
        env = {k: v for k, v in os.environ.items() if k != "MEMDEC_MODEL"}
        with patch.dict(os.environ, env, clear=True):
            result = resolve_base_model(
                step_config_key="steps.step5_evaluation",
                tokenized_data="legal_corpus_tokenized-smollm3",
                checkpoint="smollm3-step_500",
                output_dir="/outputs",
            )
        self.assertEqual(result, "unsloth/gemma-3-1b-it")
        mock_resolve_tokenized.assert_not_called()
        mock_dataset_tokenizer.assert_not_called()

    @patch('src.utils._resolve_named_path')
    @patch('src.utils.get_dataset_tokenizer_model')
    @patch('src.utils.get_pipeline_value')
    def test_pipeline_default_overrides_auto_derivation(self, mock_get_pipeline, mock_dataset_tokenizer, mock_resolve_tokenized):
        """pipeline.models.default_model must win over auto-derived sources
        once .env and CLI are unset (and no step config is set)"""

        def side_effect(key, default=None, usecase=None):
            if key == "steps.step5_evaluation.base_model":
                return None
            if key == "pipeline.models.default_model":
                return "gemma3"
            return default
        mock_get_pipeline.side_effect = side_effect
        env = {k: v for k, v in os.environ.items() if k != "MEMDEC_MODEL"}
        with patch.dict(os.environ, env, clear=True):
            result = resolve_base_model(
                step_config_key="steps.step5_evaluation",
                tokenized_data="legal_corpus_tokenized-smollm3",
                checkpoint="smollm3-step_500",
                output_dir="/outputs",
            )
        self.assertEqual(result, "unsloth/gemma-3-270m-it")
        mock_resolve_tokenized.assert_not_called()
        mock_dataset_tokenizer.assert_not_called()

    @patch('src.utils._resolve_named_path')
    @patch('src.utils.get_dataset_tokenizer_model')
    @patch('src.utils.get_pipeline_value')
    def test_tokenized_data_overrides_checkpoint_and_fallback(self, mock_get_pipeline, mock_dataset_tokenizer, mock_resolve_tokenized):
        """Tokenized data metadata must win over checkpoint and the system
        default once .env, CLI and config are unset"""

        mock_get_pipeline.return_value = None
        mock_resolve_tokenized.return_value = "/path/to/tokenized"
        mock_dataset_tokenizer.return_value = "unsloth/gemma-3-1b-it"
        env = {k: v for k, v in os.environ.items() if k != "MEMDEC_MODEL"}
        with patch.dict(os.environ, env, clear=True):
            result = resolve_base_model(
                tokenized_data="legal_corpus_tokenized-gemma3",
                checkpoint="some-checkpoint",
                output_dir="/outputs"
            )
        self.assertEqual(result, "unsloth/gemma-3-1b-it")

    @patch('src.utils.resolve_checkpoint_path')
    @patch('src.utils.get_pipeline_value')
    @patch('src.utils.get_default_model')
    def test_checkpoint_overrides_fallback(self, mock_get_default, mock_get_pipeline, mock_resolve_checkpoint):
        """Checkpoint derivation must win over the system default once .env,
        CLI, config and tokenized data are unset"""

        mock_get_pipeline.return_value = None
        mock_resolve_checkpoint.return_value = "/outputs/gemma3-step_5000"
        mock_get_default.return_value = "smollm3"
        env = {k: v for k, v in os.environ.items() if k != "MEMDEC_MODEL"}
        with patch.dict(os.environ, env, clear=True):
            result = resolve_base_model(checkpoint="gemma3-step_5000", output_dir="/outputs")
        self.assertEqual(result, "unsloth/gemma-3-270m-it")
        mock_get_default.assert_not_called()

    @patch('src.utils.get_pipeline_value')
    @patch('src.utils.get_default_model')
    def test_builtin_fallback_when_nothing_set(self, mock_get_default, mock_get_pipeline):
        """System default (get_default_model) is used when .env, CLI, config,
        tokenized data and checkpoint are all absent"""

        mock_get_pipeline.return_value = None
        mock_get_default.return_value = "gemma3"
        env = {k: v for k, v in os.environ.items() if k != "MEMDEC_MODEL"}
        with patch.dict(os.environ, env, clear=True):
            result = resolve_base_model()
        self.assertEqual(result, "unsloth/gemma-3-270m-it")
        mock_get_default.assert_called_once()


class TestProjectRel(unittest.TestCase):
    """Test project-relative path conversion for portable output"""

    def test_path_under_project_root(self):
        """Absolute path inside PROJECT_ROOT becomes a forward-slash relative path"""

        abs_path = str(PROJECT_ROOT / "dataset" / "some_dir" / "train.json")
        result = project_rel(abs_path)
        self.assertFalse(Path(result).is_absolute())
        self.assertEqual(result, "dataset/some_dir/train.json")
        self.assertNotIn("\\", result)

    def test_path_outside_project_root_unchanged(self):
        """Path outside PROJECT_ROOT is returned unchanged"""

        outside = "/other/root/file.json"
        self.assertEqual(project_rel(outside), outside)

    def test_none_and_empty_passthrough(self):
        """None and empty values pass through unchanged"""

        self.assertIsNone(project_rel(None))
        self.assertEqual(project_rel(""), "")


class TestRelativePathFilter(unittest.TestCase):
    """Test _RelativePathFilter log-record rewriting"""

    def _make_record(self, msg, args=None):
        """Build a minimal LogRecord for filter testing"""

        return logging.LogRecord(
            name="t", level=logging.INFO, pathname="", lineno=0,
            msg=msg, args=args, exc_info=None,
        )

    def test_filter_rewrites_project_path_in_msg(self):
        """Absolute PROJECT_ROOT paths in record.msg become project-relative"""

        filt = _RelativePathFilter()
        record = self._make_record(f"Writing to {PROJECT_ROOT / 'outputs' / 'x.log'} done")
        self.assertTrue(filt.filter(record))
        self.assertNotIn(str(PROJECT_ROOT), record.msg)
        self.assertIn(str(Path("outputs") / "x.log"), record.msg)

    def test_filter_rewrites_tuple_args(self):
        """Absolute PROJECT_ROOT paths in %s-style args are rewritten too"""

        filt = _RelativePathFilter()
        record = self._make_record("path=%s", (str(PROJECT_ROOT / "a" / "b"),))
        filt.filter(record)
        self.assertEqual(record.args[0], str(Path("a") / "b"))

    def test_filter_leaves_non_project_paths(self):
        """Messages without PROJECT_ROOT are not modified"""

        filt = _RelativePathFilter()
        record = self._make_record("path=/tmp/other/x")
        filt.filter(record)
        self.assertEqual(record.msg, "path=/tmp/other/x")

    def test_get_logger_attaches_filter(self):
        """get_logger attaches _RelativePathFilter to the created logger"""

        name = "test_relpath_logger"
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.filters.clear()
        try:
            lg = get_logger(name)
            self.assertTrue(any(isinstance(f, _RelativePathFilter) for f in lg.filters))
        finally:
            lg.handlers.clear()
            lg.filters.clear()


class TestResolveCheckpointPath(unittest.TestCase):
    """Test resolve_checkpoint_path formats: absolute, relative, step names, latest"""

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
                with patch('src.utils.logger'):
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

        result = _resolve_named_path(None, DATA_DIR, "tokenized-data")
        self.assertIsNone(result)

    def test_absolute_path_exists(self):
        """Test absolute path that exists"""

        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.is_absolute', return_value=True):
                with patch('pathlib.Path.__str__', return_value="/absolute/path"):
                    result = _resolve_named_path("/absolute/path", DATA_DIR, "tokenized-data")
                    self.assertEqual(result, "/absolute/path")

    def test_absolute_path_not_exists(self):
        """Test absolute path that doesn't exist"""

        with patch('pathlib.Path.exists', return_value=False):
            with patch('pathlib.Path.is_absolute', return_value=True):
                result = _resolve_named_path("/absolute/path", DATA_DIR, "tokenized-data")
                self.assertIsNone(result)

    def test_relative_in_project_root(self):
        """Test relative path found in PROJECT_ROOT (checked first)"""

        with patch('src.utils.PROJECT_ROOT', Path("/project")):
            with patch('pathlib.Path.exists') as mock_exists:
                mock_exists.side_effect = [True, False]
                result = _resolve_named_path("relative/path", Path("/data"), "tokenized-data")
                self.assertEqual(result, str(Path("/project/relative/path")))

    def test_relative_in_data_dir(self):
        """Test relative path found in DATA_DIR (fallback after project root)"""

        with patch('src.utils.PROJECT_ROOT', Path("/project")):
            with patch('pathlib.Path.exists') as mock_exists:
                mock_exists.side_effect = [False, True]
                result = _resolve_named_path("relative/path", Path("/data"), "tokenized-data")
                self.assertEqual(result, str(Path("/data/relative/path")))

    def test_not_found_anywhere(self):
        """Test path not found anywhere"""

        with patch('src.utils.PROJECT_ROOT', Path("/project")):
            with patch('pathlib.Path.exists', return_value=False):
                result = _resolve_named_path("relative/path", Path("/data"), "tokenized-data")
                self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)