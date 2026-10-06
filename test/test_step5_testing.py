"""
Test Suite for src/step5_testing.py (including unit & integration tests + edge case testing)
"""

import sys, shutil, tempfile, unittest, json, torch
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_mock_memory_decoder_module = MagicMock()
_mock_memory_decoder_cls    = type('MemoryDecoder', (object,), {})
_mock_memory_decoder_module.MemoryDecoder = _mock_memory_decoder_cls
sys.modules.setdefault("MemoryDecoder",              MagicMock())
sys.modules.setdefault("MemoryDecoder.demo",         MagicMock())
sys.modules.setdefault("MemoryDecoder.demo.memDec",  _mock_memory_decoder_module)

import src.step5_testing as step5
from src.step5_testing import (
    TestingConfig,
    load_models,
    generate_responses,
    test_legal_scenario,
    save_results,
    load_test_config,
    main,
)

LEGAL_TESTS = load_test_config()


class TestTestingConfig(unittest.TestCase):
    """Test TestingConfig dataclass defaults and overrides"""

    def test_default_values(self):
        """Test that default config values match source config"""

        config = TestingConfig()
        self.assertEqual(config.max_new_tokens, TestingConfig.max_new_tokens)
        self.assertEqual(config.lmbda, TestingConfig.lmbda)
        self.assertEqual(config.batch_size, TestingConfig.batch_size)
        self.assertTrue(config.save_results)
        self.assertTrue(config.compare_with_base)
        self.assertIsNone(config.base_model)
        self.assertIsNone(config.checkpoint_dir)

    def test_custom_values(self):
        """Test custom values override defaults"""

        config = TestingConfig(
            max_new_tokens=200,
            lmbda=0.7,
            compare_with_base=False,
            base_model="my-model",
        )
        self.assertEqual(config.max_new_tokens, 200)
        self.assertEqual(config.lmbda, 0.7)
        self.assertEqual(config.knn_temp, 1.0)
        self.assertFalse(config.compare_with_base)
        self.assertEqual(config.base_model, "my-model")

    def test_optional_checkpoint_path(self):
        """Test checkpoint_path is optional and defaults to None"""

        config = TestingConfig(base_model="gemma")
        self.assertIsNone(config.checkpoint_dir)

        config_with_ckpt = TestingConfig(base_model="gemma", checkpoint_dir="/some/path")
        self.assertEqual(config_with_ckpt.checkpoint_dir, "/some/path")


class TestLegalTestsStructure(unittest.TestCase):
    """Test the LEGAL_TESTS dictionary is correctly structured"""

    def test_all_expected_scenarios_present(self):
        """Test all expected scenario keys are present"""

        self.assertGreater(len(LEGAL_TESTS), 0, "LEGAL_TESTS must contain at least one scenario")

    def test_each_scenario_has_title_and_prompts(self):
        """Test each scenario contains required title and prompts fields"""

        for name, scenario in LEGAL_TESTS.items():
            with self.subTest(scenario=name):
                self.assertIn("title", scenario)
                self.assertIn("prompts", scenario)
                self.assertIsInstance(scenario["title"], str)
                self.assertIsInstance(scenario["prompts"], list)

    def test_each_scenario_has_four_prompts(self):
        """Test each scenario contains exactly four prompts"""

        for name, scenario in LEGAL_TESTS.items():
            with self.subTest(scenario=name):
                self.assertEqual(len(scenario["prompts"]), 4)

    def test_prompts_are_non_empty_strings(self):
        """Test all prompts are non-empty strings"""

        for name, scenario in LEGAL_TESTS.items():
            for prompt in scenario["prompts"]:
                with self.subTest(scenario=name, prompt=prompt[:30]):
                    self.assertIsInstance(prompt, str)
                    self.assertGreater(len(prompt.strip()), 0)


class TestSetupEnvironment(unittest.TestCase):
    """Test directory creation and logging (moved into main function)"""

    def setUp(self):
        """Setup test data"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Cleanup test data"""
        shutil.rmtree(self.temp_dir)

    @patch('src.step5_testing.logger')
    def test_main_creates_output_directories(self, mock_logger):
        """Test that main creates required directories"""

        results_dir = str(Path(self.temp_dir) / "results")
        kb_path = str(Path(self.temp_dir) / "knowledge_base")
        config = TestingConfig(
            base_model="my-model",
            results_dir=results_dir,
            knowledge_base_path=kb_path,
            save_results=False,
        )
        with patch('src.step5_testing.load_models') as mock_load:
            mock_load.return_value = (Mock(), Mock(), Mock(), Mock())
            with patch('src.step5_testing.test_legal_scenario') as mock_test:
                mock_test.return_value = {
                    "scenario_name": "test", "title": "Test", "prompts": [],
                    "memory_decoder_responses": [], "base_model_responses": [],
                    "source_checks": []
                }
                main(config)
                self.assertTrue(Path(results_dir).exists())
                self.assertTrue(Path(kb_path).exists())

    @patch('src.step5_testing.logger')
    def test_main_logs_config_values(self, mock_logger):
        """Test that all config keys are logged during main execution"""

        config = TestingConfig(
            base_model="my-model",
            results_dir=str(Path(self.temp_dir) / "results"),
            knowledge_base_path=str(Path(self.temp_dir) / "kb"),
            save_results=False,
        )
        with patch('src.step5_testing.load_models') as mock_load:
            mock_load.return_value = (Mock(), Mock(), Mock(), Mock())
            with patch('src.step5_testing.test_legal_scenario') as mock_test:
                mock_test.return_value = {
                    "scenario_name": "test", "title": "Test", "prompts": [],
                    "memory_decoder_responses": [], "base_model_responses": [],
                    "source_checks": []
                }
                main(config)
                logged_messages = [str(c) for c in mock_logger.info.call_args_list]
                self.assertTrue(any("max_new_tokens" in m for m in logged_messages))


class TestLoadModels(unittest.TestCase):
    """Test load_models function"""

    def _make_config(self, checkpoint_path=None):
        """Create a TestingConfig with default values"""

        return TestingConfig(
            base_model="my-model",
            checkpoint_dir=checkpoint_path,
            knowledge_base_path="/tmp/kb",
            output_dir="/tmp/outputs",
        )

    @patch('src.step5_testing.logger')
    @patch('src.step5_testing.validate_checkpoint_model_type')
    @patch('src.step5_testing.AutoConfig')
    @patch('src.step5_testing.MemoryDecoder')
    @patch('src.step5_testing.AutoModelForCausalLM.from_pretrained')
    @patch('src.step5_testing.initialize_tokenizer')
    @patch('src.step5_testing.setup_device')
    def test_load_models_with_checkpoint(
        self, mock_device, mock_tokenizer, mock_from_pretrained, mock_memdec, mock_auto_config, mock_validate, mock_logger
    ):
        """Test model loading succeeds when checkpoint_path is provided"""

        mock_device.return_value = Mock(type="cpu")
        mock_tokenizer.return_value = Mock(__len__=Mock(return_value=32000))
        mock_model = Mock()
        mock_model.parameters = Mock(return_value=iter([Mock(device=Mock())]))
        mock_from_pretrained.return_value = mock_model
        mock_memdec_instance = Mock()
        mock_memdec.return_value = mock_memdec_instance
        mock_memdec_instance.to = Mock(return_value=mock_memdec_instance)
        config = self._make_config(checkpoint_path="/path/to/checkpoint")
        tokenizer, base_lm, knn_gen, mem_dec = load_models(config)
        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(base_lm)
        self.assertIsNotNone(knn_gen)
        self.assertIsNotNone(mem_dec)
        mock_validate.assert_called_once()
        mock_logger.info.assert_called()

    @patch('src.step5_testing.logger')
    @patch('src.step5_testing.validate_checkpoint_model_type')
    @patch('src.step5_testing.AutoConfig')
    @patch('src.step5_testing.MemoryDecoder')
    @patch('src.step5_testing.AutoModelForCausalLM.from_pretrained')
    @patch('src.step5_testing.initialize_tokenizer')
    @patch('src.step5_testing.setup_device')
    def test_load_models_fallback_on_knn_error(
        self, mock_device, mock_tokenizer, mock_from_pretrained, mock_memdec, mock_auto_config, mock_validate, mock_logger
    ):
        """Test that exception propagates when checkpoint_path is set and knn loading fails"""
 
        mock_device.return_value = Mock(type="cpu")
        mock_tokenizer.return_value = Mock(__len__=Mock(return_value=32000))
        mock_base_model = Mock()
        mock_from_pretrained.side_effect = [mock_base_model, Exception("Checkpoint corrupt")]
        mock_memdec_instance = Mock()
        mock_memdec_instance.to = Mock(return_value=mock_memdec_instance)
        mock_memdec.return_value = mock_memdec_instance
        config = self._make_config(checkpoint_path="/bad/path")
        with self.assertRaises(Exception) as cm:
            load_models(config)
        self.assertIn("Checkpoint corrupt", str(cm.exception))
        mock_logger.error.assert_called()

    @patch('src.step5_testing.logger')
    @patch('src.step5_testing.AutoConfig')
    @patch('src.step5_testing.initialize_tokenizer')
    @patch('src.step5_testing.setup_device')
    def test_load_models_raises_on_base_model_failure(
        self, mock_device, mock_tokenizer, mock_auto_config, mock_logger
    ):
        """Test that an exception is raised when the base model itself fails to load"""
 
        mock_device.return_value = Mock(type="cpu")
        mock_tokenizer.return_value = Mock()
        config = self._make_config()
        with patch('src.step5_testing.AutoModelForCausalLM.from_pretrained') as mock_load:
            mock_load.side_effect = Exception("Model not found")
            with self.assertRaises(Exception) as cm:
                load_models(config)
            self.assertIn("Model not found", str(cm.exception))
        mock_logger.error.assert_called()


class TestGenerateResponse(unittest.TestCase):
    """Test generate_response function"""

    def _make_config(self, **kwargs):
        """Create a TestingConfig with default values"""
        defaults = dict(max_new_tokens=50)
        defaults.update(kwargs)
        return TestingConfig(**defaults)

    def _make_mock_inputs(self, input_length=3):
        """Build a mock tokenizer-output whose input_ids.shape[1] is a real int."""

        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = Mock(
            side_effect=lambda key: MagicMock(shape=(1, input_length)) if key == "input_ids" else MagicMock()
        )
        mock_inputs.to = Mock(return_value=mock_inputs)
        return mock_inputs

    def _make_mock_model(self, output_token_ids=None):
        """Build a plain (non-MemoryDecoder) mock model"""

        if output_token_ids is None:
            output_token_ids = [0, 1, 2, 3, 4]
        mock_model = Mock()
        mock_param = Mock()
        mock_param.device = "cpu"
        mock_model.parameters = Mock(return_value=iter([mock_param]))
        mock_model.generate = Mock(return_value=torch.tensor([output_token_ids]))
        return mock_model

    def test_generate_response_returns_decoded_new_tokens(self):
        """Test that only the newly generated tokens (after the prompt) are decoded and returned"""

        input_length  = 3
        decoded_new   = "Das Widerrufsrecht erlaubt..."
        mock_inputs   = self._make_mock_inputs(input_length)
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode = Mock(return_value=decoded_new)
        mock_model = self._make_mock_model(output_token_ids=[0, 1, 2, 3, 4])
        config = self._make_config()
        results = generate_responses(mock_model, mock_tokenizer, ["Was ist das Widerrufsrecht?"], config)
        mock_tokenizer.decode.assert_called_once()
        self.assertEqual(len(results), 1)
        self.assertIn(decoded_new, results[0])

    def test_generate_response_no_prefix_if_not_present(self):
        """Test response is returned as decoded text when it doesn't start with the prompt"""

        decoded = "Das ist eine Antwort."
        mock_inputs    = self._make_mock_inputs(input_length=3)
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode = Mock(return_value=decoded)
        mock_model = self._make_mock_model()
        config = self._make_config()
        results = generate_responses(mock_model, mock_tokenizer, ["Ein Verbraucher kauft online..."], config)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], str)
        self.assertIn("Das ist eine Antwort", results[0])

    def test_generate_response_passes_generation_params(self):
        """Test that generation parameters from config are forwarded to model.generate"""

        mock_inputs    = self._make_mock_inputs(input_length=1)
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.eos_token_id = 99
        mock_tokenizer.decode = Mock(return_value="response")
        mock_model = self._make_mock_model(output_token_ids=[0, 7])
        config = self._make_config(max_new_tokens=200)
        generate_responses(mock_model, mock_tokenizer, ["prompt"], config)
        mock_model.generate.assert_called_once()
        _, call_kwargs = mock_model.generate.call_args
        self.assertEqual(call_kwargs["max_new_tokens"], 200)
        self.assertFalse(call_kwargs["do_sample"])
        self.assertEqual(call_kwargs["pad_token_id"], 99)
        self.assertIn("repetition_penalty", call_kwargs)
        self.assertEqual(call_kwargs["repetition_penalty"], config.repetition_penalty)

    def test_generate_response_sampling_params(self):
        """Test that sampling parameters are forwarded when do_sample is enabled"""

        mock_inputs    = self._make_mock_inputs(input_length=1)
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.eos_token_id = 99
        mock_tokenizer.decode = Mock(return_value="response")
        mock_model = self._make_mock_model(output_token_ids=[0, 7])
        config = self._make_config(do_sample=True, temperature=0.8, top_p=0.9, top_k=40)
        generate_responses(mock_model, mock_tokenizer, ["prompt"], config)
        _, call_kwargs = mock_model.generate.call_args
        self.assertTrue(call_kwargs["do_sample"])
        self.assertEqual(call_kwargs["temperature"], 0.8)
        self.assertEqual(call_kwargs["top_p"], 0.9)
        self.assertEqual(call_kwargs["top_k"], 40)


class TestTestLegalScenario(unittest.TestCase):
    """Test test_legal_scenario function"""

    def _make_config(self, compare_with_base=True):
        """Create test config"""

        return TestingConfig(compare_with_base=compare_with_base)

    @patch('src.step5_testing.generate_responses')
    def test_returns_correct_structure(self, mock_generate):
        """Test that returned result dict has all required keys"""

        mock_generate.side_effect = lambda model, tok, prompts, cfg, name="m": ["Eine rechtliche Antwort."] * len(prompts)
        scenario = {"title": "Test Szenario", "prompts": ["Frage 1", "Frage 2"]}
        config = self._make_config()
        result = test_legal_scenario("test", scenario, Mock(), Mock(), Mock(), config)
        self.assertEqual(result["scenario_name"], "test")
        self.assertEqual(result["title"], "Test Szenario")
        self.assertEqual(len(result["prompts"]), 2)
        self.assertEqual(len(result["memory_decoder_responses"]), 2)
        self.assertEqual(len(result["base_model_responses"]), 2)
        self.assertIn("source_checks", result)
        self.assertEqual(len(result["source_checks"]), 2)
        self.assertIn("gen_time_s", result["source_checks"][0]["memory_decoder"])

    @patch('src.step5_testing.generate_responses')
    def test_skips_base_model_when_disabled(self, mock_generate):
        """Test base model is not called when compare_with_base is False"""

        mock_generate.side_effect = lambda model, tok, prompts, cfg, name="m": ["Antwort."] * len(prompts)
        scenario = {"title": "Szenario", "prompts": ["Frage"]}
        config = self._make_config(compare_with_base=False)
        mock_base = Mock()
        mock_memdec = Mock()
        result = test_legal_scenario("test", scenario, Mock(), mock_base, mock_memdec, config)
        self.assertEqual(result["base_model_responses"], [])
        self.assertIn("source_checks", result)
        mock_generate.assert_called_once()

    @patch('src.step5_testing.generate_responses')
    def test_generation_times_recorded_per_prompt(self, mock_generate):
        """Test that generation times are recorded for each prompt"""

        mock_generate.side_effect = lambda model, tok, prompts, cfg, name="m": ["Antwort."] * len(prompts)
        scenario = {"title": "Timing Test", "prompts": ["F1", "F2", "F3"]}
        config = self._make_config()
        result = test_legal_scenario("timing", scenario, Mock(), Mock(), Mock(), config)
        self.assertEqual(len(result["source_checks"]), 3)
        for entry in result["source_checks"]:
            self.assertIn("gen_time_s", entry["memory_decoder"])
            self.assertIn("gen_time_s", entry["base_model"])


class TestSaveResults(unittest.TestCase):
    """Test save_results function writes expected files"""

    def setUp(self):
        """Setup test data"""

        temp_test_dir = step5.PROJECT_ROOT / "temp_test"
        temp_test_dir.mkdir(exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(dir=temp_test_dir))

    def tearDown(self):
        """Cleanup test data"""
        
        shutil.rmtree(self.temp_dir)
        temp_test_dir = step5.PROJECT_ROOT / "temp_test"
        if temp_test_dir.exists():
            shutil.rmtree(temp_test_dir)

    def test_saves_json_with_config_and_results(self):
        """Test that detailed JSON file is created with config and results"""

        results_dir = self.temp_dir / "outputs" / "test_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        config = TestingConfig(results_dir=str(results_dir))
        results = [{
            "scenario_name": "tax",
            "title": "Steuerbescheid",
            "prompts": ["Frage?"],
            "memory_decoder_responses": ["Antwort."],
            "base_model_responses": ["Base Antwort."],
            "source_checks": [{"memory_decoder": {"gen_time_s": 0.5},
                               "base_model": {"gen_time_s": 0.8}}],
        }]
        save_results(results, config)
        output_files = list(results_dir.iterdir())
        json_files = [f for f in output_files if f.suffix == ".json"]
        self.assertEqual(len(json_files), 1)

    def test_json_contains_expected_structure(self):
        """Test that saved JSON file contains config and results keys"""

        results_dir = self.temp_dir / "outputs" / "test_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        config = TestingConfig(results_dir=str(results_dir))
        results = [{
            "scenario_name": "rental",
            "title": "Miete",
            "prompts": ["Frage?"],
            "memory_decoder_responses": ["Antwort."],
            "base_model_responses": [],
            "source_checks": [{"memory_decoder": {"gen_time_s": 1.2}}],
        }]
        save_results(results, config)
        json_file = next(results_dir.glob("test_results_*.json"))
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("config", data)
        self.assertIn("summary", data)
        self.assertIn("results", data)

    def test_config_paths_use_forward_slashes(self):
        """Test that config paths relative to PROJECT_ROOT use forward slashes (Windows-safe)"""

        results_dir = self.temp_dir / "outputs" / "test_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        kb_path = str(step5.PROJECT_ROOT / "knowledge_base")
        config = TestingConfig(
            results_dir=str(results_dir),
            knowledge_base_path=kb_path,
            output_dir=str(step5.PROJECT_ROOT / "outputs"),
        )
        results = [{
            "scenario_name": "employment",
            "title": "Kündigung",
            "prompts": ["Frage?"],
            "memory_decoder_responses": ["Antwort."],
            "base_model_responses": [],
            "source_checks": [{"memory_decoder": {"gen_time_s": 0.9}}],
        }]
        save_results(results, config)
        json_file = next(results_dir.glob("test_results_*.json"))
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        project_root_str = str(step5.PROJECT_ROOT)
        for key, value in data["config"].items():
            if isinstance(value, str) and project_root_str.replace("\\", "/") in value.replace("\\", "/"):
                self.assertNotIn("\\", value, msg=f"Backslash found in config key '{key}': {value}")

    def test_filename_includes_model_keyword(self):
        """Test the results filename embeds the resolved model keyword"""

        results_dir = self.temp_dir / "outputs" / "test_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        config = TestingConfig(
            results_dir=str(results_dir),
            base_model="unsloth/gemma-3-1b-it",
        )
        save_results([], config)
        json_file = next(results_dir.glob("test_results_*.json"))
        expected_kw = step5.extract_model_identifier(config.base_model) or "model"
        self.assertTrue(
            json_file.name.startswith(f"test_results_{expected_kw}_"),
            msg=f"Unexpected filename: {json_file.name}",
        )


class TestMainFunction(unittest.TestCase):
    """Test main function and overall orchestration"""

    def setUp(self):
        """Setup test data"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Cleanup test data"""
        shutil.rmtree(self.temp_dir)

    @patch('src.step5_testing.logger')
    @patch('src.step5_testing.test_legal_scenario')
    @patch('src.step5_testing.load_models')
    def test_main_runs_all_scenarios_by_default(
        self, mock_load, mock_test, mock_logger
    ):
        """Test that main iterates over all LEGAL_TESTS scenarios by default"""

        mock_load.return_value = (Mock(), Mock(), Mock(), Mock())
        mock_test.return_value = {
            "scenario_name": "x", "title": "X", "prompts": [],
            "memory_decoder_responses": [], "base_model_responses": [],
            "source_checks": []
        }
        config = TestingConfig(
            base_model="my-model",
            results_dir=self.temp_dir,
            knowledge_base_path=self.temp_dir,
            output_dir=self.temp_dir,
            save_results=False,
        )
        main(config)
        mock_load.assert_called_once_with(config)
        self.assertEqual(mock_test.call_count, len(LEGAL_TESTS))
        mock_logger.info.assert_called()

    @patch('src.step5_testing.logger')
    @patch('src.step5_testing.load_models')
    def test_main_raises_and_logs_on_model_load_failure(
        self, mock_load, mock_logger
    ):
        """Test main propagates exception when load_models raises"""

        mock_load.side_effect = RuntimeError("CUDA out of memory")
        config = TestingConfig(
            base_model="my-model",
            results_dir=self.temp_dir,
            knowledge_base_path=self.temp_dir,
            output_dir=self.temp_dir,
        )
        with self.assertRaises(RuntimeError) as cm:
            main(config)
        self.assertIn("CUDA out of memory", str(cm.exception))
        mock_logger.error.assert_called()

    @patch('src.step5_testing.logger')
    @patch('src.step5_testing.test_legal_scenario')
    @patch('src.step5_testing.load_models')
    def test_main_skips_failed_scenario_and_continues(
        self, mock_load, mock_test, mock_logger
    ):
        """Test that a failing scenario is skipped and remaining scenarios still run"""

        mock_load.return_value = (Mock(), Mock(), Mock(), Mock())
        success_result = {
            "scenario_name": "ok", "title": "OK", "prompts": [],
            "memory_decoder_responses": [], "base_model_responses": [],
            "source_checks": []
        }
        mock_test.side_effect = [
            Exception("Scenario failed"),
            *[success_result] * (len(LEGAL_TESTS) - 1)
        ]
        config = TestingConfig(
            base_model="mock-model",
            results_dir=self.temp_dir,
            knowledge_base_path=self.temp_dir,
            output_dir=self.temp_dir,
            save_results=False,
        )
        main(config)
        mock_logger.error.assert_called()
        self.assertEqual(mock_test.call_count, len(LEGAL_TESTS))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    @patch('src.step5_testing.generate_responses')
    def test_test_legal_scenario_single_prompt(self, mock_generate):
        """Test scenario with a single prompt runs without error"""

        mock_generate.side_effect = lambda model, tok, prompts, cfg, name="m": ["Antwort."] * len(prompts)
        scenario = {"title": "Single", "prompts": ["Nur eine Frage?"]}
        config = TestingConfig(compare_with_base=False)
        result = test_legal_scenario("single", scenario, Mock(), Mock(), Mock(), config)
        self.assertEqual(len(result["prompts"]), 1)
        self.assertEqual(len(result["memory_decoder_responses"]), 1)
        self.assertEqual(len(result["source_checks"]), 1)
        self.assertIn("gen_time_s", result["source_checks"][0]["memory_decoder"])

    def test_testing_config_serializable_via_dict(self):
        """Test that TestingConfig.__dict__ contains only JSON-safe primitive types"""

        config = TestingConfig(base_model="test-model", checkpoint_dir="/path")
        for key, value in config.__dict__.items():
            with self.subTest(key=key):
                self.assertIsInstance(value, (str, int, float, bool, type(None)))

    def test_legal_tests_german_prompts_encoded_correctly(self):
        """Test German prompts containing umlauts are proper Python strings"""

        for scenario_name, scenario in LEGAL_TESTS.items():
            for prompt in scenario["prompts"]:
                with self.subTest(scenario=scenario_name):
                    encoded = prompt.encode("utf-8")
                    self.assertEqual(encoded.decode("utf-8"), prompt)

    @patch('src.step5_testing.generate_responses')
    def test_generate_responses_called_once_per_model_per_prompt_batch(
        self, mock_generate
    ):
        """Test generate_responses is called once per model per prompt when batch_size=1"""

        mock_generate.side_effect = lambda model, tok, prompts, cfg, name="m": ["Antwort."] * len(prompts)
        scenario = {"title": "Doppelaufruf", "prompts": ["F1", "F2"]}
        config = TestingConfig(compare_with_base=True, batch_size=1)
        test_legal_scenario("double", scenario, Mock(), Mock(), Mock(), config)
        self.assertEqual(mock_generate.call_count, 4)


class TestSourceCheck(unittest.TestCase):
    """Test source-check helper functions"""

    def test_normalize_for_match_lowercases_and_collapses_whitespace(self):
        """Test _normalize_for_match lowercases and collapses whitespace"""

        from src.step5_testing import _normalize_for_match
        result = _normalize_for_match("  Hello   WORLD  ")
        self.assertEqual(result, "hello world")

    def test_normalize_for_match_handles_empty_string(self):
        """Test _normalize_for_match returns empty string for empty input"""

        from src.step5_testing import _normalize_for_match
        self.assertEqual(_normalize_for_match(""), "")

    def test_simple_source_check_disabled_when_no_sources(self):
        """Test _simple_source_check returns enabled=False when no sources found"""

        from src.step5_testing import _simple_source_check
        result = _simple_source_check("some response", "nonexistent_scenario")
        self.assertFalse(result.get("enabled", True))

    def test_simple_source_check_empty_response(self):
        """Test _simple_source_check returns zero scores for empty response"""

        from src.step5_testing import _simple_source_check
        result = _simple_source_check("", "withdrawals")
        self.assertEqual(result.get("max_fuzzy"), 0.0)
        self.assertEqual(result.get("max_overlap"), 0.0)

    def test_token_re_matches_legal_references(self):
        """Test _TOKEN_RE captures paragraph references and legal abbreviations"""

        from src.step5_testing import _TOKEN_RE
        tokens = set(_TOKEN_RE.findall("§355 bgb und §§312g bgb sowie kschg"))
        self.assertIn("§355", tokens)
        self.assertIn("§§312g", tokens)
        self.assertIn("bgb", tokens)
        self.assertIn("kschg", tokens)

    def test_source_check_result_keys_present(self):
        """Test _simple_source_check always returns required keys on valid input"""

        from src.step5_testing import _simple_source_check
        result = _simple_source_check("Eine Antwort über Widerruf.", "nonexistent_scenario")
        self.assertIn("enabled", result)


class TestSummarizeSourceChecks(unittest.TestCase):
    """Test summarize_source_checks aggregation"""

    def _mk_results(self):
        """Two scored prompts where memory_decoder wins fuzzy, splits overlap"""

        return [{
            "scenario_name": "s1",
            "source_checks": [
                {"memory_decoder": {"enabled": True, "max_fuzzy": 0.6, "max_overlap": 0.4},
                 "base_model":     {"enabled": True, "max_fuzzy": 0.4, "max_overlap": 0.5}},
                {"memory_decoder": {"enabled": True, "max_fuzzy": 0.2, "max_overlap": 0.2},
                 "base_model":     {"enabled": True, "max_fuzzy": 0.1, "max_overlap": 0.1}},
            ],
        }]

    def test_overall_means_wins_and_winner(self):
        """Test means, pairwise win counts and overall winner across scenarios"""

        from src.step5_testing import summarize_source_checks
        summary = summarize_source_checks(self._mk_results(), compare_with_base=True)
        overall = summary["overall"]
        self.assertEqual(overall["evaluated_pairs"], 2)
        self.assertAlmostEqual(overall["memory_decoder"]["mean_max_fuzzy"], 0.4)
        self.assertAlmostEqual(overall["memory_decoder"]["mean_max_overlap"], 0.3)
        self.assertAlmostEqual(overall["base_model"]["mean_max_fuzzy"], 0.25)
        self.assertEqual(overall["memory_decoder"]["fuzzy_wins"], 2)
        self.assertEqual(overall["memory_decoder"]["overlap_wins"], 1)
        self.assertEqual(overall["base_model"]["overlap_wins"], 1)
        self.assertEqual(overall["winner"], "memory_decoder")

    def test_by_scenario_and_winner_none_without_pairs(self):
        """Test per-scenario aggregation and winner=None when no pairwise comparison"""

        from src.step5_testing import summarize_source_checks
        summary = summarize_source_checks(self._mk_results(), compare_with_base=True)
        self.assertEqual(summary["by_scenario"]["s1"]["evaluated_pairs"], 2)
        no_pairs = [{"scenario_name": "s2", "source_checks": [
            {"memory_decoder": {"enabled": True, "max_fuzzy": 0.5, "max_overlap": 0.5}},
        ]}]
        summary = summarize_source_checks(no_pairs, compare_with_base=True)
        self.assertIsNone(summary["overall"]["winner"])

    def test_disabled_checks_are_skipped(self):
        """Test entries with enabled=False do not contribute to stats or pairs"""

        from src.step5_testing import summarize_source_checks
        results = [{"scenario_name": "s", "source_checks": [
            {"memory_decoder": {"enabled": False, "reason": "no_sources"},
             "base_model":     {"enabled": True, "max_fuzzy": 0.9, "max_overlap": 0.9}},
        ]}]
        summary = summarize_source_checks(results, compare_with_base=True)
        self.assertEqual(summary["overall"]["memory_decoder"]["n"], 0)
        self.assertEqual(summary["overall"]["base_model"]["n"], 1)
        self.assertEqual(summary["overall"]["evaluated_pairs"], 0)
        self.assertIsNone(summary["overall"]["winner"])

    def test_winner_none_when_not_comparing(self):
        """Test winner is None when compare_with_base is disabled"""

        from src.step5_testing import summarize_source_checks
        summary = summarize_source_checks(self._mk_results(), compare_with_base=False)
        self.assertIsNone(summary["overall"]["winner"])
        self.assertNotIn("base_model", summary["overall"])


if __name__ == '__main__':
    unittest.main(verbosity=2)