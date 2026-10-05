"""
Utility Functions (Reusable)
"""

import os, sys, logging, argparse, inspect, shutil, torch, json
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
from pathlib import Path
from typing import Dict, Optional, Any
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / '.env'
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


# --- CONFIG ---
logger = logging.getLogger("utils")
logger.addHandler(logging.NullHandler())
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
ENV_FILE = PROJECT_ROOT / ".env"
CONFIG_DIR = PROJECT_ROOT / "config"
MODEL_CONFIG_FILE = CONFIG_DIR / "model_config.json"
USECASE_DIR = CONFIG_DIR / "legal_usecase"
DATASET_CONFIG_FILE = CONFIG_DIR / "dataset_config.yaml"
PIPELINE_CONFIG_FILE = CONFIG_DIR / "pipeline_config.yaml"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# --- LOGGING SETUP ---
class _RelativePathFilter(logging.Filter):
    """Rewrite absolute PROJECT_ROOT paths in log messages to project-relative paths."""
    _ROOT = str(PROJECT_ROOT)

    def _rel(self, text: str) -> str:
        """
        Convert absolute path to relative path.
        
        Args:
            text: Absolute path string
            
        Returns:
            Relative path string
        """
        return (text.replace(self._ROOT + os.sep, "")
                    .replace(self._ROOT + "/", "")
                    .replace(self._ROOT, ""))

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records to replace absolute paths with relative paths.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to include the record, False to exclude it
        """
        if isinstance(record.msg, str) and self._ROOT in record.msg:
            record.msg = self._rel(record.msg)
        args = record.args
        if isinstance(args, dict):
            record.args = {k: self._rel(v) if isinstance(v, str) else v for k, v in args.items()}
        elif isinstance(args, tuple):
            record.args = tuple(self._rel(v) if isinstance(v, str) else v for v in args)
        elif isinstance(args, str) and self._ROOT in args:
            record.args = (self._rel(args),)
        return True


def get_logger(log_name: str | None = None) -> logging.Logger:
    """
    Create a logger writing to outputs/logs/<log_name>.log and stdout.
    When invoked from a test module, only the stdout handler is attached.
    
    Args:
        log_name: Name of log file (without .log extension)
                 Examples: 'preparation', 'training', 'inference'
    
    Returns:
        logging.Logger: Configured logger instance
    """
    if log_name is None:
        frame_info = inspect.stack()[1]
        module = inspect.getmodule(frame_info.frame)
        log_name = Path(module.__file__).stem if module and module.__file__ else "default"
    logger = logging.getLogger(log_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.addFilter(_RelativePathFilter())

    is_test = False
    for frame in inspect.stack():
        module = inspect.getmodule(frame.frame)
        if module and module.__file__:
            module_path = Path(module.__file__)
            if 'test' in module_path.parts or module_path.name.startswith('test_'):
                is_test = True
                break
    if not is_test:
        log_dir = Path(__file__).parent.parent / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{log_name}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    if hasattr(stream_handler.stream, 'reconfigure'):
        stream_handler.stream.reconfigure(encoding='utf-8')
    logger.addHandler(stream_handler)
    return logger


def log_blank_line(logger_obj: logging.Logger | None = None) -> None:
    """
    Insert a blank line in all handler outputs (without logging header).
    
    Args:
        logger_obj: Logger instance to use. If None, uses the root logger.
        
    Returns:
        None
    """
    target_logger = logger_obj or logging.getLogger()
    for handler in target_logger.handlers:
        stream = getattr(handler, "stream", None)
        if stream is not None:
            if hasattr(handler, "baseFilename"):
                log_path = handler.baseFilename
                if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
                    continue
            stream.write("\n")
            stream.flush()


# --- CONFIG LOADING ---
def get_active_usecase() -> str:
    """
    Get the active use-case from environment variable or default.
    
    Returns:
        Name of the active use-case (e.g., "legal_usecase")
    """
    return os.getenv("MEMDEC_USECASE", "legal_usecase")


def get_usecase_config_path(usecase: str = None) -> Path:
    """
    Get the config directory for a specific use-case.
    
    Args:
        usecase: Use-case name. If None, uses active use-case from env var.
        
    Returns:
        Path to the use-case config directory
    """
    if usecase is None:
        usecase = get_active_usecase()
    usecase_dir = CONFIG_DIR / usecase
    if usecase_dir.exists():
        return usecase_dir
    return CONFIG_DIR


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load a configuration file (YAML or JSON).
    
    Args:
        config_path: Path to the config file (YAML or JSON)
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ImportError: If required library is not installed
        ValueError: If file format is not supported
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    file_ext = config_path.suffix.lower()
    with open(config_path, 'r', encoding='utf-8') as f:
        if file_ext in ('.yaml', '.yml'):
            if not YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML is required to load YAML config files. "
                )
            return yaml.safe_load(f)
        elif file_ext == '.json':
            return json.load(f)
        else:
            raise ValueError(
                f"Unsupported config file format: {file_ext}. "
                f"Supported formats: .yaml, .yml, .json"
            )


def _load_usecase_config(filename: str, global_fallback: Optional[Path] = None, usecase: str = None) -> Dict[str, Any]:
    """
    Load a named YAML config from the use-case directory, falling back to a global file.

    Args:
        filename: Config file name inside the use-case directory (e.g. 'dataset_config.yaml')
        global_fallback: Global config path to use when no use-case file exists (None = required)
        usecase: Use-case name. If None, uses active use-case from env var.

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If neither the use-case file nor the global fallback exists
    """
    config_path = get_usecase_config_path(usecase) / filename
    if config_path.exists():
        return load_config(config_path)
    if global_fallback and global_fallback.exists():
        return load_config(global_fallback)
    raise FileNotFoundError(
        f"No '{filename}' found for use-case '{usecase}' or as global config"
    )


def get_pipeline_value(key_path: str, default: Any = None, usecase: str = None) -> Any:
    """
    Get a value from pipeline config using dot notation (e.g., "steps.step4_training.batch_size").
    
    Args:
        key_path: Dot-separated path to the config value
        default: Default value if key not found
        usecase: Use-case name. If None, uses active use-case from env var.
        
    Returns:
        The config value or default
    """
    try:
        config = _load_usecase_config("pipeline_config.yaml", PIPELINE_CONFIG_FILE, usecase)
        keys = key_path.split('.')
        value = config
        for key in keys:
            value = value[key]
        return value
    except (FileNotFoundError, KeyError, ImportError, TypeError):
        return default


def _resolve_named_path(arg: Optional[str], base_dir: Path, label: str) -> Optional[str]:
    """
    Resolve a path argument against absolute path, project root, then a base directory.

    Args:
        arg: User-supplied name/path (absolute path, project-relative, or base-relative)
        base_dir: Fallback directory to search (e.g. outputs/ or dataset/)
        label: Human-readable resource label used in warning messages

    Returns:
        Optional[str]: Resolved absolute path string or None if not found
    """
    if not arg:
        return None
    p = Path(arg)
    if p.is_absolute():
        if p.exists():
            return str(p)
        logger.warning(f"⚠️ {label} not found at absolute path: {p}")
        return None
    candidate_project = PROJECT_ROOT / p
    if candidate_project.exists():
        return str(candidate_project)
    candidate_base = base_dir / p
    if candidate_base.exists():
        return str(candidate_base)
    logger.warning(
        f"⚠️ Could not resolve {label} '{arg}'. "
        f"Tried: {candidate_project}, {candidate_base}"
    )
    return None


def resolve_checkpoint_path(checkpoint_arg: Optional[str], output_dir: str) -> Optional[str]:
    """
    Resolve checkpoint path from various input formats.
    
    Args:
        checkpoint_arg: Checkpoint identifier (absolute path, relative path, step name, or 'latest')
        output_dir: Base output directory for checkpoints
        
    Returns:
        Optional[str]: Resolved absolute path as string or None if not found
    """
    if not checkpoint_arg:
        return None
    output_path = Path(output_dir)
    if checkpoint_arg.lower() == "latest":
        candidates = (
            [d for d in output_path.iterdir()
             if d.is_dir() and ("checkpoint" in d.name or d.name.startswith("step_"))]
            if output_path.exists() else []
        )
        if not candidates:
            logger.warning(f"⚠️ No checkpoints found in {output_path}")
            return None

        def _step_num(d: Path) -> int:
            """Extract step number from directory name (supports '_' and '-')."""
            for sep in ["_", "-"]:
                parts = d.name.split(sep)
                if parts[-1].isdigit():
                    return int(parts[-1])
            return 0

        latest = max(candidates, key=_step_num)
        logger.info(f"🔍 'latest' resolved → {latest}")
        return str(latest)
    return _resolve_named_path(checkpoint_arg, output_path, "checkpoint")


# --- GLOBAL LLM-MODEL HANDLING ---
def get_default_model() -> str:
    """
    Get default model from env variable (.env), model_config.json, or fallback.

    Resolution order:
        1. MEMDEC_MODEL environment variable
        2. 'default_model' key in model_config.json
        3. Hard-coded fallback 'gemma3'

    Returns:
        Default model name (e.g., gemma3/gemma3-1b/qwen3.5/smollm3)
    """
    try:
        env_model = os.getenv("MEMDEC_MODEL")
        if env_model:
            return env_model.lower()
    except Exception:
        pass
    try:
        if MODEL_CONFIG_FILE.exists():
            config = load_config(MODEL_CONFIG_FILE)
            cfg_model = config.get("default_model")
            if cfg_model:
                return str(cfg_model).lower()
    except Exception:
        pass
    return "gemma3"


def set_default_model(model_name: str) -> None:
    """
    Set default model in model_config.json and the .env file.

    Args:
        model_name: Model keyword (must exist in model_config.json)

    Returns:
        None

    Raises:
        ValueError: If model_name is not valid
    """
    model_name = model_name.lower()
    valid_models = get_model_keywords()
    if model_name not in valid_models:
        raise ValueError(f"Invalid model: {model_name}. Choose from: {valid_models}")
    try:
        config = load_config(MODEL_CONFIG_FILE) if MODEL_CONFIG_FILE.exists() else {}
        config["default_model"] = model_name
        MODEL_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"❌ Failed to save config: {e}")
        raise
    try:
        env_file = PROJECT_ROOT / ".env"
        env_lines = []
        memdec_model_found = False
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                env_lines = f.readlines()
        for i, line in enumerate(env_lines):
            if line.strip().startswith("MEMDEC_MODEL="):
                env_lines[i] = f"MEMDEC_MODEL='{model_name}'\n"
                memdec_model_found = True
                break
        if not memdec_model_found:
            env_lines.append(f"MEMDEC_MODEL='{model_name}'\n")
        with open(env_file, 'w', encoding='utf-8') as f:
            f.writelines(env_lines)
    except Exception as e:
        print(f"⚠️ Failed to save .env (config file was updated): {e}")
    print(f"✅ Default model set to: {model_name}")


def get_model_config(model_choice: str = None) -> Dict[str, str]:
    """
    Get model configuration based on specified model choice or default.

    Args:
        model_choice: Model name (gemma3/qwen3.5/smollm3). If None, uses default.

    Returns:
        Dictionary containing model name and description

    Raises:
        ValueError: If model choice is not supported or config file is missing
    """
    if model_choice is None:
        model_choice = get_default_model()
    model_choice = model_choice.lower()
    if not MODEL_CONFIG_FILE.exists():
        raise ValueError(f"Model config file not found: {MODEL_CONFIG_FILE}")
    try:
        config = load_config(MODEL_CONFIG_FILE)
        models = config.get("models", {})
        if model_choice in models:
            return models[model_choice]
        available = list(models.keys())
        raise ValueError(f"Invalid model choice: {model_choice}. Available models: {available}")
    except Exception as e:
        if "Invalid model choice" in str(e):
            raise
        raise ValueError(f"Error loading model config: {e}")


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name to the canonical model keyword (e.g., gemma3/qwen3.5/smollm3).

    Args:
        model_name: Model name to normalize (e.g., "unsloth/gemma-3-270m-it")

    Returns:
        Normalized model keyword (e.g., "gemma3")
    """
    identifier = extract_model_identifier(model_name, return_default=False)
    if identifier:
        return identifier
    return model_name.lower().replace('unsloth/', '').replace('-it', '').replace('_', '').replace('.', '')


def get_knowledge_base_paths(model_choice: str = None, knowledge_base_dir: str = None) -> Dict[str, str]:
    """
    Get knowledge base file paths for a specific model.

    Args:
        model_choice: Model name (gemma3/qwen3.5/smollm3 or full HuggingFace name). If None, uses default.
        knowledge_base_dir: Custom knowledge base directory. If None, uses default.

    Returns:
        Dictionary with 'dstore' and 'index' file paths

    Raises:
        ValueError: If model choice is not supported or config file is missing
    """
    if model_choice is None:
        model_choice = get_default_model()
    model_choice = extract_model_identifier(model_choice).lower()
    if not MODEL_CONFIG_FILE.exists():
        raise ValueError(f"Model config file not found: {MODEL_CONFIG_FILE}")
    
    try:
        config = load_config(MODEL_CONFIG_FILE)
        models = config.get("models", {})
        if model_choice not in models:
            available = list(models.keys())
            raise ValueError(f"Invalid model choice: {model_choice}. Available models: {available}")
        model_data = models[model_choice]
        dstore = model_data.get("dstore")
        index = model_data.get("index")
        if not dstore or not index:
            raise ValueError(f"Knowledge base paths not defined for model: {model_choice}")
        if knowledge_base_dir is None:
            knowledge_base_dir = str(KNOWLEDGE_BASE_DIR)
        return {
            "dstore": str(Path(knowledge_base_dir) / dstore),
            "index": str(Path(knowledge_base_dir) / index)
        }
    except Exception as e:
        if "Invalid model choice" in str(e) or "Knowledge base paths not defined" in str(e):
            raise
        raise ValueError(f"Error loading knowledge base config: {e}")


def get_dataset_tokenizer_model(tokenized_data_path: str) -> Optional[str]:
    """
    Extract the tokenizer model from the dataset's metadata file.

    Args:
        tokenized_data_path: Path to the tokenized dataset directory

    Returns:
        Tokenizer model name if found, None otherwise
    """
    metadata_file = Path(tokenized_data_path) / "dataset_metadata.json"
    if not metadata_file.exists():
        metadata_file = Path(tokenized_data_path) / "state.json"
    if metadata_file.exists():
        try:
            metadata = load_config(metadata_file)
            return metadata.get('tokenizer_model')
        except Exception as e:
            if logger:
                logger.warning(f"Could not read metadata file: {e}")
    return None


def resolve_base_model(
    cli_model: Optional[str] = None,
    step_config_key: Optional[str] = None,
    tokenized_data: Optional[str] = None,
    checkpoint: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    Resolve the base model name according to the strict hierarchy:

    CLI > per-step pipeline config (base_model) > tokenized dataset metadata >
    checkpoint name > pipeline.models.default_model > env default (MEMDEC_MODEL)

    Auto-derivation from tokenized_data or checkpoint takes precedence over the global
    pipeline default: a base model that does not match the dataset tokenizer or checkpoint
    cannot consume them anyway. An explicit CLI or per-step config choice is never
    silently overridden by auto-derivation.

    Args:
        cli_model: Value passed via --model on the CLI (highest priority).
        step_config_key: Pipeline config key for the step (e.g. "steps.step5_evaluation").
        tokenized_data: Tokenized dataset name/path for auto-derivation.
        checkpoint: Checkpoint name/path for auto-derivation.
        output_dir: Output directory used to resolve relative checkpoint names.

    Returns:
        The canonical full HuggingFace model name for the resolved base model.

    Raises:
        ValueError: If no source can be resolved or the model choice is unsupported.
    """
    # 1. CLI override (highest priority)
    if cli_model:
        full_name = get_model_full_name(cli_model)
        logger.info(f"✅ Base model resolved from CLI: {full_name}")
        return full_name
    # 2. Per-step pipeline config
    if step_config_key:
        step_model = get_pipeline_value(f"{step_config_key}.base_model", None)
        if step_model:
            full_name = get_model_full_name(step_model)
            logger.info(f"✅ Base model resolved from {step_config_key}.base_model: {full_name}")
            return full_name
    # 3. Auto-derive from tokenized dataset metadata
    if tokenized_data:
        resolved_path = _resolve_named_path(tokenized_data, DATA_DIR, "tokenized-data")
        if resolved_path:
            metadata_model = get_dataset_tokenizer_model(resolved_path)
            if metadata_model:
                full_name = get_model_full_name(metadata_model)
                logger.info(f"✅ Base model auto-derived from tokenized data metadata: {full_name}")
                return full_name
        full_name = get_model_full_name(tokenized_data)
        logger.info(f"✅ Base model auto-derived from tokenized data name: {full_name}")
        return full_name
    # 4. Auto-derive from checkpoint (config.json first, then name heuristics)
    if checkpoint:
        resolved_checkpoint = resolve_checkpoint_path(checkpoint, output_dir) if output_dir else checkpoint
        model_source = None
        if resolved_checkpoint:
            config_file = Path(resolved_checkpoint) / "config.json"
            if config_file.exists():
                try:
                    ckpt_config = load_config(config_file)
                    model_source = ckpt_config.get("_name_or_path") or ckpt_config.get("model_type")
                except Exception:
                    pass
        for candidate in (model_source, resolved_checkpoint, checkpoint):
            if not candidate:
                continue
            keyword = extract_model_identifier(str(candidate), return_default=False)
            if keyword:
                full_name = get_model_full_name(keyword)
                logger.info(f"✅ Base model auto-derived from checkpoint '{checkpoint}': {full_name}")
                return full_name
        logger.warning(
            f"⚠️ Could not auto-derive base model from checkpoint '{checkpoint}' — "
            "falling back to pipeline/env default"
        )
    # 5. Pipeline-wide default model
    pipeline_default = get_pipeline_value("pipeline.models.default_model", None)
    if pipeline_default:
        full_name = get_model_full_name(pipeline_default)
        logger.info(f"✅ Base model resolved from pipeline.models.default_model: {full_name}")
        return full_name
    # 6. Env / hard-coded default (lowest priority)
    default_model = get_default_model()
    full_name = get_model_full_name(default_model)
    logger.info(f"✅ Base model resolved from env default: {full_name}")
    return full_name


def _clean_model_string(s: str) -> str:
    """Lowercase and strip separators (. _ - /) for normalized model-string comparison.
    
    Args:
        s: String to clean
        
    Returns:
        Cleaned string
    """
    return s.lower().replace('.', '').replace('_', '').replace('-', '').replace('/', '')


def extract_model_identifier(input_string: str, return_default: bool = True) -> str | None:
    """
    Extract the model identifier (gemma3/qwen3.5/smollm3) from various input formats.
    
    This unified function handles:
    - Full HuggingFace model names (e.g., "unsloth/gemma-3-270m-it")
    - Tokenized dataset names (e.g., "legal_corpus_tokenized-smollm3")
    - Model keywords (e.g., "gemma3")
    - Checkpoint paths

    Args:
        input_string: String containing model information
        return_default: If True, returns default model when not found; if False, returns None

    Returns:
        Model keyword (gemma3/qwen3.5/smollm3) or default_model/None if not found
    """
    if not input_string:
        return get_default_model() if return_default else None
    input_lower = input_string.lower()
    clean_input = _clean_model_string(input_lower)
    model_keywords = get_model_keywords()
    if MODEL_CONFIG_FILE.exists():
        try:
            config = load_config(MODEL_CONFIG_FILE)
            models = config.get("models", {})
            if input_lower in models:
                return input_lower
            for model_key, model_data in models.items():
                if model_data["name"].lower() == input_lower:
                    return model_key
            for model_key, model_data in sorted(
                    models.items(),
                    key=lambda kv: len(_clean_model_string(kv[0])),
                    reverse=True):
                clean_key  = _clean_model_string(model_key)
                clean_name = _clean_model_string(model_data["name"])
                if (clean_key and clean_key in clean_input) or \
                        (clean_name and clean_name in clean_input):
                    return model_key
        except Exception:
            pass
    for kw in sorted(model_keywords, key=lambda k: len(_clean_model_string(k)), reverse=True):
        if _clean_model_string(kw) in clean_input:
            return kw
    if "gemma" in input_lower:
        return "gemma3"
    elif "qwen" in input_lower:
        return "qwen3.5"
    elif "smollm" in input_lower:
        return "smollm3"
    return get_default_model() if return_default else None


def get_model_keywords() -> list:
    """
    Get list of model keywords from model_config.json.
    
    Args:
        None
    
    Returns:
        List of model keywords (e.g., ['gemma3', 'qwen3.5', 'smollm3'])
    """
    if MODEL_CONFIG_FILE.exists():
        try:
            config = load_config(MODEL_CONFIG_FILE)
            keywords = list(config.get("models", {}).keys())
            if keywords:
                return keywords
        except Exception:
            pass
    return ["gemma3", "qwen3.5", "smollm3"]


def get_model_full_name(model_input: str) -> str:
    """
    Resolve any model reference (keyword, HF id, dataset/checkpoint name) to the
    canonical Hugging Face model name from model_config.json.

    Args:
        model_input: Model reference string; falls back to the default model when no
            keyword can be extracted

    Returns:
        Full HuggingFace model name (e.g., 'unsloth/gemma-3-270m-it')

    Raises:
        ValueError: If the resolved keyword is not present in model_config.json
    """
    keyword = extract_model_identifier(model_input)
    return get_model_config(keyword)["name"]


def matches_current_model(filename: str, model_keyword: str | None, hidden_size: int) -> bool:
    """
    Check if the filename matches the current model and hidden size.
    
    Args:
        filename: Name of the datastore/index file
        model_keyword: Model keyword (gemma3/qwen3.5/smollm3)
        hidden_size: Hidden size of the model
    
    Returns:
        True if the filename matches the current model and hidden size, False otherwise
    """
    filename_lower = filename.lower()
    stem = os.path.splitext(filename_lower)[0]
    if not stem.endswith(f"_{hidden_size}"):
        return False
    if model_keyword:
        try:
            paths = get_knowledge_base_paths(model_keyword)
            if filename_lower in {os.path.basename(p).lower() for p in paths.values()}:
                return True
        except Exception:
            pass
        return model_keyword in filename_lower
    return True


def cleanup_qwen_config(config, model_name: str) -> None:
    """
    Remove vision_config and set text-only configuration for Qwen3.5 models.

    Args:
        config: AutoConfig object to modify
        model_name: Model name to check for Qwen3.5

    Returns:
        None
    """
    if "qwen" in model_name.lower() or (hasattr(config, 'model_type') and config.model_type == "qwen3_5_text"):
        if hasattr(config, "vision_config"):
            delattr(config, "vision_config")
        if hasattr(config, "text_config"):
            config.architectures = ["Qwen3_5ForCausalLM"]
            config.model_type = "qwen3_5_text"


def validate_checkpoint_model_type(checkpoint_path: str, base_model_path: str, base_model_config) -> None:
    """
    Validate that checkpoint model type matches base model type.

    Args:
        checkpoint_path: Path to the checkpoint
        base_model_path: Path or name of the base model
        base_model_config: AutoConfig object of the base model

    Returns:
        None

    Raises:
        ValueError: If model types don't match
    """
    from transformers import AutoConfig
    checkpoint_config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    checkpoint_model_type = getattr(checkpoint_config, 'model_type', 'unknown')
    base_model_type = getattr(base_model_config, 'model_type', 'unknown')
    checkpoint_model_norm = checkpoint_model_type.replace('_', '').replace('.', '').lower()
    base_model_norm = base_model_type.replace('_', '').replace('.', '').lower()
    if checkpoint_model_norm != base_model_norm and not (
        'qwen' in checkpoint_model_norm and 'qwen' in base_model_norm
    ):
        raise ValueError(
            f"Cannot load checkpoint: Checkpoint model type ({checkpoint_model_type}) does not match base model type ({base_model_type}). "
            f"Checkpoint: {checkpoint_path}, Base model: {base_model_path}. "
            f"Please use a checkpoint trained with the same model architecture."
        )


# --- SYSTEM ---
def setup_device(force_cpu: bool = False) -> torch.device:
    """
    Setup device for training.
    
    Args:
        force_cpu: If True, force CPU usage even if GPU is available
        
    Returns:
        torch.device: Device to use for training
    """
    if force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Using available device: {device}")
    if device_count > 0:
        print(f"Number of available GPUs: {device_count}")
        for i in range(device_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory/1024**3:.2f}GB")
    return device


def cleanup_temp_files(logger: logging.Logger | None = None) -> int:
    """
    Clean up Hugging Face temp dir and 'cache-*' files in dataset/ directories.
    
    Args:
        logger: Logger instance to use for logging
        
    Returns:
        int: Number of files cleaned
    """
    temp_dir = Path.home() / '.cache' / 'huggingface' / 'tmp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True)
        if logger:
            logger.info(f"Cleaned Hugging Face temp directory: {temp_dir}")

    def clean_cache_files(directory: Path) -> int:
        """Clean cache files from a directory."""
        if not directory.exists():
            return 0
        cache_files = list(directory.glob('cache-*'))
        if not cache_files:
            return 0
        count = 0
        for cache_file in cache_files:
            try:
                if cache_file.is_file():
                    cache_file.unlink()
                    count += 1
                elif cache_file.is_dir():
                    shutil.rmtree(cache_file, ignore_errors=True)
                    count += 1
            except Exception as e:
                if logger:
                    logger.warning(f"Could not delete {cache_file}: {e}")
        return count

    base_dir = Path('dataset')
    cleaned = 0
    if base_dir.exists():
        for dataset_dir in base_dir.iterdir():
            if dataset_dir.is_dir():
                cleaned += clean_cache_files(dataset_dir)
    if cleaned > 0 and logger:
        logger.info(f"Cleaned {cleaned} cache files from dataset directories")
    return cleaned


def project_rel(value: Optional[str]) -> Optional[str]:
    """Return an absolute path relative to PROJECT_ROOT for portable JSON output."""
    if not value:
        return value
    try:
        return str(Path(value).relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return value


# --- CLI ---
def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments (use default config values if not provided) for setting/getting the default LLM model.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    valid_models = get_model_keywords()
    parser = argparse.ArgumentParser(description="Set or get default LLM model for MemDec training")
    parser.add_argument(
        "model",
        nargs="?",
        choices=valid_models,
        help=f"Set default model ({'/'.join(valid_models)}). If not provided, shows current default."
    )
    parser.add_argument("--show", action="store_true", help="Show current default model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.show or not args.model:
        current_model = get_default_model()
        current_config = get_model_config(current_model)
        print(f"🤖 Current Base Model: {current_model}")
        print(f"📝 Description: {current_config['description']}")
        print(f"🔗 Hugging Face: {current_config['name']}")
        print(f"📁 Config file: {ENV_FILE}")
    else:
        set_default_model(args.model)