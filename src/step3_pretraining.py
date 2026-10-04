"""
MemDec Knowledge Base Creation
"""

import os, sys
os.environ['FORCE_TORCHAUDIO_AVAILABLE'] = '0'
sys.modules['torchaudio'] = None
import torchvision.io
class DummyVideoReader:
    def __init__(self, *args, **kwargs): pass
torchvision.io.VideoReader = DummyVideoReader
# ----------------------------------------------------------------------
import argparse, torch, pyarrow as pa
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from datasets import load_from_disk
from accelerate import Accelerator
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

from MemoryDecoder.knn_utils.saveEmbedMulti import KNNSaverMulti, KEY_TYPE, get_index_path


# --- LOGGING ---
from src.utils import get_logger, log_blank_line
logger = get_logger('3_pretraining')
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.add(
        lambda message: logger.info(str(message).rstrip("\n")),
        format="{message}",
        filter=lambda record: "Flushed buffer" not in record["message"],
    )
except Exception:
    pass


# --- CONFIG ---
from src.utils import cleanup_temp_files, get_pipeline_value, get_default_model, resolve_base_model, get_model_config, _resolve_named_path, setup_device, cleanup_qwen_config, get_dataset_tokenizer_model, normalize_model_name, matches_current_model, extract_model_identifier, PROJECT_ROOT, DATA_DIR, KNOWLEDGE_BASE_DIR, MODEL_CONFIG_FILE
CORPUS_NAME = get_pipeline_value("pipeline.corpus_name", "legal_corpus")
TOKENIZED_DATA_DIR = DATA_DIR / get_pipeline_value(
    "steps.step3_pretraining.tokenized_data", 
    f"{CORPUS_NAME}_tokenized-{extract_model_identifier(get_default_model())}"
)

@dataclass
class PretrainConfig:
    """Configuration class for pretraining parameters."""
    # Paths & LLM Model
    base_model: str = None
    tokenized_data_path: str = TOKENIZED_DATA_DIR
    knn_datastore_path: str = KNOWLEDGE_BASE_DIR
    # Training settings
    batch_size: int = 8
    max_length: int = 512
    seed: int = 42
    # KNN params
    code_size: int = 32
    probe: int = 8
    ncentroids: int = 2048     # 4096 for large datasets, 2048 for small datasets
    num_keys_to_add_at_a_time: int = 100_000   # 1_000_000 for large RAM, 100_000 for small RAM


# --- HELPERS ---
def build_datastore_labels(input_ids: list[int], attention_mask: list[int]) -> list[int]:
    """
    Build labels so that only REAL tokens become datastore entries.

    KNNSaverMulti stores (key at position j-1, label at position j) for every label != -100.
    Padding must therefore be -100. With left padding (Gemma tokenizers) the first real token
    must be -100 as well, because its key would come from a padding position.

    Args:
        input_ids: Token ids (already padded to sequence length)
        attention_mask: 1 = real token, 0 = padding (taken from step2, NOT rebuilt)

    Returns:
        Labels: token id where the token and its predecessor are real tokens, else -100
    """
    labels = []
    prev_real = 0
    for tok, m in zip(input_ids, attention_mask):
        labels.append(tok if (m == 1 and prev_real == 1) else -100)
        prev_real = m
    return labels


def _validate_arrow_file(file_path: str, dimension: int) -> int | None:
    """
    Validate a datastore Arrow IPC stream end-to-end (header, schema, EOS marker, all batches).

    Args:
        file_path: Path to the .arrow datastore file
        dimension: Expected fixed size of the 'keys' lists

    Returns:
        Row count if the file is complete and readable, else None
    """
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return None
        with open(file_path, 'rb') as f:
            f.seek(-8, os.SEEK_END)
            if f.read(8) != b'\xff\xff\xff\xff\x00\x00\x00\x00':
                logger.warning(f"⚠️ Arrow stream not properly terminated (missing EOS): {os.path.basename(file_path)}")
                return None
        arrow_file = pa.OSFile(file_path, 'rb')
        try:
            reader = pa.ipc.open_stream(arrow_file)
            keys_field = reader.schema.field('keys') if 'keys' in reader.schema.names else None
            if (keys_field is None or not pa.types.is_fixed_size_list(keys_field.type)
                    or keys_field.type.list_size != dimension):
                logger.warning(f"⚠️ Arrow schema mismatch (expected fixed-size keys[{dimension}]): {os.path.basename(file_path)}")
                return None
            n_rows = 0
            while True:
                try:
                    n_rows += reader.read_next_batch().num_rows
                except StopIteration:
                    break
            return n_rows if n_rows > 0 else None
        finally:
            arrow_file.close()
    except Exception:
        return None


def create_knn_datastore(config: PretrainConfig) -> None:
    """
    Create KNN datastore using the existing knn_utils (MemDec Repo).
    
    Args:
        config: PretrainConfig containing model and datastore parameters
        
    Returns:
        None
    """
    logger.info("🚀 Creating KNN datastore using optimized utilities...")
    os.makedirs(config.knn_datastore_path, exist_ok=True)
    logger.info("🔧 Setting up model...")
    device      = setup_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # autotune kernels for the fixed (B x 512) batch shape
    else:
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    device_map  = "auto" if device.type == "cuda" else None
    quantization_config = BitsAndBytesConfig(load_in_8bit=False) if device.type == "cuda" else None
    model_config = AutoConfig.from_pretrained(config.base_model)
    cleanup_qwen_config(model_config, config.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map=device_map,
        config=model_config,
        low_cpu_mem_usage=True
    )
    model.eval()
    if device.type != "cuda":
        model = model.to(device)
    max_sequence_length = getattr(model_config, 'max_position_embeddings', 2048)
    sliding_window = getattr(model_config, 'sliding_window', 512)
    model_capacity = min(sliding_window, max_sequence_length) if sliding_window else max_sequence_length
    effective_max_length = min(model_capacity, config.max_length)
    logger.info(
        f"Using sequence length: {effective_max_length} "
        f"(config.max_length: {config.max_length}, model capacity: {model_capacity} "
        f"[sliding_window: {sliding_window}, max_position_embeddings: {max_sequence_length}])"
    )
    hidden_size = getattr(model.config, 'hidden_size', None) or getattr(
        getattr(model.config, 'text_config', None), 'hidden_size', None)
    if hidden_size is None:
        raise ValueError(f"Could not determine hidden_size for model '{config.base_model}'")
    knn_saver = KNNSaverMulti(
        dstore_dir=config.knn_datastore_path,
        dimension=hidden_size,
        knn_keytype=KEY_TYPE.last_ffn_input,
        knn_gpu=(device.type == "cuda"),
        accelerator=Accelerator()
    )
    knn_saver.model = model

    model_keyword = extract_model_identifier(config.base_model, return_default=False)
    canonical_dstore = os.path.basename(knn_saver._get_arrow_file_path())
    canonical_index = os.path.basename(
        get_index_path(config.knn_datastore_path, model.config.model_type, None, hidden_size)
    )
    dstore_file = os.path.join(config.knn_datastore_path, canonical_dstore)
    index_file = os.path.join(config.knn_datastore_path, canonical_index)
    strays = [
        f for f in os.listdir(config.knn_datastore_path)
        if f.endswith(('.arrow', '.index', '.faiss'))
        and f not in (canonical_dstore, canonical_index)
        and matches_current_model(f, model_keyword, hidden_size)
    ]
    if strays:
        logger.warning(f"⚠️ Ignoring non-canonical datastore/index files: {strays}")
    logger.info(
        f"🔎 Looking for datastore '{canonical_dstore}' and index '{canonical_index}' "
        f"(model '{config.base_model}', hidden_size: {hidden_size})"
    )
    n_rows = _validate_arrow_file(dstore_file, hidden_size)
    if n_rows is None:
        if os.path.exists(dstore_file):
            logger.info(f"🗑️ Removing unusable datastore file: {canonical_dstore}")
            try:
                os.unlink(dstore_file)
            except OSError as e:
                logger.error(f"❌ Cannot remove datastore file - it's locked by another process: {e}")
                logger.info("💡 Please close any other Python processes and try again.")
                raise
        else:
            logger.info("🔧 No KNN datastore found for this model.")
        logger.info("Starting full datastore recreation...")
    else:
        logger.info(f"✅ Found valid datastore: {canonical_dstore} ({n_rows:,} rows, {os.path.getsize(dstore_file):,} bytes)")
        if not os.path.exists(index_file):
            logger.info("🔧 No existing FAISS index. Building it now...")
            try:
                knn_saver.build_index(
                    num_keys_to_add_at_a_time=config.num_keys_to_add_at_a_time,
                    ncentroids=config.ncentroids,
                    seed=config.seed,
                    code_size=config.code_size,
                    probe=config.probe
                )
                logger.info(f"✅ FAISS index created successfully!")
                return
            except Exception as e:
                logger.warning(f"⚠️ Failed to build index from existing datastore: {e}")
                logger.info("Will recreate datastore from scratch.")
                try:
                    os.unlink(dstore_file)
                except OSError as ue:
                    logger.error(f"❌ Cannot remove datastore file '{canonical_dstore}': {ue}")
                    raise
        else:
            logger.info(f"✅ Found existing FAISS index: {canonical_index}")
            logger.info("Skipping datastore creation as it already exists.")
            return
    if os.path.exists(index_file):
        logger.info(f"ℹ️ Existing FAISS index '{canonical_index}' will be rebuilt together with the datastore")
    try:
        logger.info("Loading tokenized dataset from Arrow format...")
        tokenized_path = Path(config.tokenized_data_path).resolve()
        logger.info(f"Looking for dataset in:")
        logger.info(f"1. {tokenized_path / 'arrow_data'}")
        logger.info(f"2. {tokenized_path}")
        
        arrow_path = tokenized_path / "arrow_data"
        if arrow_path.exists():
            try:
                logger.info(f"🔍 Loading dataset from arrow directory: {arrow_path}")
                dataset = load_from_disk(str(arrow_path))
                logger.info(f"✅ Successfully loaded dataset from {arrow_path} with {len(dataset)} examples")
            except Exception as e:
                logger.error(f"❌ Error loading dataset from {arrow_path}: {str(e)}")
                logger.exception("Detailed error:")
                return
        elif tokenized_path.exists():
            try:
                logger.info(f"🔍 Loading dataset from: {tokenized_path}")
                dataset = load_from_disk(str(tokenized_path))
                logger.info(f"✅ Successfully loaded dataset from {tokenized_path} with {len(dataset)} examples")
            except Exception as e:
                logger.error(f"❌ Error loading dataset from {tokenized_path}: {str(e)}")
                logger.exception("Detailed error:")
                return
        else:
            logger.error(f"❌ Tokenized dataset not found at {tokenized_path} (tried: {tokenized_path} and {arrow_path})")
            logger.info("Please make sure to run step2_tokenization.py first to create the dataset.")
            return
        logger.info("🔧 Initializing arrow writer for datastore creation...")
        knn_saver.break_into(model)  # creates the .arrow file and opens the writer
        total_positions, valid_positions = 0, 0
        with torch.inference_mode():
            for i in tqdm(range(0, len(dataset), config.batch_size), 
                 desc="Processing batches", 
                 unit="batch",
                 total=(len(dataset) + config.batch_size - 1) // config.batch_size):
                batch = dataset[i:i + config.batch_size]
                input_ids = []
                attention_masks = []
                label_rows = []
                for seq, seq_mask in zip(batch["input_ids"], batch["attention_mask"]):
                    if isinstance(seq, torch.Tensor):
                        seq = seq.tolist()
                    if isinstance(seq_mask, torch.Tensor):
                        seq_mask = seq_mask.tolist()
                    seq_len = min(len(seq), effective_max_length)
                    pad_len = effective_max_length - seq_len
                    padded = seq[:seq_len] + [0] * pad_len
                    mask = list(seq_mask[:seq_len]) + [0] * pad_len
                    row_labels = build_datastore_labels(padded, mask)
                    total_positions += len(row_labels)
                    valid_positions += sum(1 for l in row_labels if l != -100)
                    input_ids.append(padded)
                    attention_masks.append(mask)
                    label_rows.append(row_labels)
                input_ids = torch.tensor(input_ids, device=model.device, dtype=torch.long)
                attention_mask = torch.tensor(attention_masks, device=model.device, dtype=torch.long)
                labels = torch.tensor(label_rows, device=model.device, dtype=torch.long)
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                _ = model(**inputs)
        if total_positions:
            logger.info(
                f"Datastore entries: {valid_positions:,} of {total_positions:,} positions "
                f"({valid_positions / total_positions:.1%}) are real tokens (padding excluded)"
            )
        knn_saver.break_out()
        logger.info("Building KNN datastore (.arrow)...")
        knn_saver.build_index(
            num_keys_to_add_at_a_time=config.num_keys_to_add_at_a_time,
            ncentroids=config.ncentroids,
            seed=config.seed,
            code_size=config.code_size,
            probe=config.probe
        )
        logger.info("✅ KNN datastore created successfully!")

    except Exception as e:
        logger.error(f"❌ Error creating KNN datastore: {str(e)}")
        raise


# --- MAIN ---
def main(config: PretrainConfig) -> int:
    """
    Main function to create KNN datastore with proper error handling and cleanup.
    
    Args:
        config: PretrainConfig object with all configuration parameters
    
    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        resolved_path = _resolve_named_path(config.tokenized_data_path, DATA_DIR, "tokenized-data")
        if not resolved_path:
            logger.error(f"❌ Tokenized data path not found or invalid: {config.tokenized_data_path}")
            return 1
        tokenized_data_path = Path(resolved_path)
        knn_datastore_path = Path(config.knn_datastore_path)
        if not knn_datastore_path.is_absolute():
            knn_datastore_path = Path.cwd() / knn_datastore_path
        knn_datastore_path.mkdir(parents=True, exist_ok=True)
        config.tokenized_data_path = str(tokenized_data_path)
        config.knn_datastore_path = str(knn_datastore_path)
        log_blank_line(logger)
        logger.info("=" * 60)
        logger.info("🧠 STEP 3: MemDec Knowledge Base Creation")
        logger.info("=" * 60)
        logger.info(f"Base Model: {config.base_model}")
        logger.info(f"Tokenized Data Path: {config.tokenized_data_path}")
        logger.info(f"KNN Datastore Path: {config.knn_datastore_path}")
        logger.info(f"Batch Size: {config.batch_size}, Seed: {config.seed}")
        logger.info(f"Sequence Length: {config.max_length}")
        # Validate base model matches dataset tokenizer
        dataset_tokenizer = get_dataset_tokenizer_model(config.tokenized_data_path)
        if dataset_tokenizer:
            config_normalized = normalize_model_name(config.base_model)
            dataset_normalized = normalize_model_name(dataset_tokenizer)
            if config_normalized != dataset_normalized:
                logger.error(f"Model mismatch detected!")
                logger.error(f"  Dataset tokenizer model: {dataset_tokenizer}")
                logger.error(f"  Specified base model: {config.base_model}")
                logger.error(f"KNN datastore creation cannot proceed with mismatched models.")
                return 1
            else:
                logger.info(f"✅ Base model matches dataset tokenizer: {config.base_model}")
        
        arrow_data_path = tokenized_data_path / 'arrow_data'
        if not arrow_data_path.exists():
            logger.error(f"❌ Arrow data directory not found at: {arrow_data_path}")
            logger.info("Please make sure to run step2_tokenization.py first to create the dataset.")
            return 1
        try:
            contents = [f.name for f in tokenized_data_path.glob('*')]
            logger.info(f"Contents of tokenized data directory ({len(contents)} items): {contents}")
            if 'arrow_data' in contents:
                logger.info(f"Found arrow_data directory at: {arrow_data_path}")
        except Exception as e:
            logger.warning(f"Could not list contents of {tokenized_data_path}: {e}")
        create_knn_datastore(config)
        logger.info("="*60)
        dataset_name = Path(config.tokenized_data_path).name
        logger.info("ℹ️ Next step: MemDec Training")
        logger.info(f" Run: python -m src.step4_training {dataset_name}")
        logger.info("="*60)
        return 0
        
    except KeyboardInterrupt:
        logger.warning("🚫 Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Error during KNN datastore creation: {str(e)}", exc_info=True)
        return 1
    finally:
        cleanup_temp_files()


# --- CLI ---
def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments (use default config values if not provided) for pretraining.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Create KNN datastore for MemDec-Module')
    default_config = PretrainConfig()
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--model', type=str, default=None)
    base_parser.add_argument('tokenized_data', nargs='?', default=None)
    early_args = base_parser.parse_known_args()[0]
    # Determine tokenized_data: CLI arg > pipeline config
    if early_args.tokenized_data:
        config_tokenized_data = early_args.tokenized_data
    else:
        try:
            config_tokenized_data = get_pipeline_value("steps.step3_pretraining.tokenized_data", None)
        except Exception:
            config_tokenized_data = None
    # Determine base_model: CLI > step config > tokenized data > pipeline default > env
    config_base_model = resolve_base_model(
        cli_model=early_args.model,
        step_config_key="steps.step3_pretraining",
        tokenized_data=config_tokenized_data,
    )
    try:
        config_batch_size = get_pipeline_value("steps.step3_pretraining.batch_size", default_config.batch_size) or default_config.batch_size
    except Exception:
        config_batch_size = default_config.batch_size
    try:
        config_seed = get_pipeline_value("steps.step3_pretraining.seed", default_config.seed) or default_config.seed
    except Exception:
        config_seed = default_config.seed
    try:
        config_ncentroids = get_pipeline_value("steps.step3_pretraining.ncentroids", default_config.ncentroids) or default_config.ncentroids
    except Exception:
        config_ncentroids = default_config.ncentroids
    try:
        config_num_keys_to_add_at_a_time = get_pipeline_value("steps.step3_pretraining.num_keys_to_add_at_a_time", default_config.num_keys_to_add_at_a_time) or default_config.num_keys_to_add_at_a_time
    except Exception:
        config_num_keys_to_add_at_a_time = default_config.num_keys_to_add_at_a_time
    # Variable defaults via pipeline_config
    parser.add_argument('--model', default=config_base_model,
                       help=f'Base model choice (gemma3/qwen3.5/smollm3) [default: {config_base_model}]')
    parser.add_argument('tokenized_data', nargs='?', default=config_tokenized_data,
                       help=f'Path to the tokenized dataset directory [default: {config_tokenized_data}]')
    parser.add_argument('--batch-size', type=int, default=config_batch_size,
                       help=f'Batch size for processing [default: {config_batch_size}]')
    parser.add_argument('--seed', type=int, default=config_seed,
                       help=f'Random seed [default: {config_seed}]')
    parser.add_argument('--ncentroids', type=int, default=config_ncentroids,
                       help=f'Number of centroids for KNN [default: {config_ncentroids}]')
    parser.add_argument('--num-keys-to-add-at-a-time', type=int, default=config_num_keys_to_add_at_a_time,
                       help=f'Number of keys to add at a time [default: {config_num_keys_to_add_at_a_time}]')
    # Fixed defaults via PretrainConfig (cannot be changed via CLI)
    parser.add_argument('--knn-datastore-path', default=default_config.knn_datastore_path,
                       help=f'[FIXED] Path to KNN datastore [default: {default_config.knn_datastore_path}]')
    parser.add_argument('--max-length', type=int, default=default_config.max_length,
                       help=f'[FIXED] Maximum sequence length [default: {default_config.max_length}]')
    parser.add_argument('--code-size', type=int, default=default_config.code_size,
                       help=f'[FIXED] Code size for KNN [default: {default_config.code_size}]')
    parser.add_argument('--probe', type=int, default=default_config.probe,
                       help=f'[FIXED] Number of probes for KNN [default: {default_config.probe}]')
    args = parser.parse_args()
    if args.knn_datastore_path != default_config.knn_datastore_path:
        parser.error(f"--knn-datastore-path is fixed to '{default_config.knn_datastore_path}' and cannot be changed via CLI.")
    if args.max_length != default_config.max_length:
        parser.error(f"--max-length is fixed to {default_config.max_length} and cannot be changed via CLI.")
    if args.code_size != default_config.code_size:
        parser.error(f"--code-size is fixed to {default_config.code_size} and cannot be changed via CLI.")
    if args.probe != default_config.probe:
        parser.error(f"--probe is fixed to {default_config.probe} and cannot be changed via CLI.")
    return args


if __name__ == '__main__':
    args = parse_arguments()
    model_keyword = extract_model_identifier(args.model)
    model_info = get_model_config(model_keyword)
    config = PretrainConfig(
        tokenized_data_path=args.tokenized_data,
        base_model=model_info['name'],
        batch_size=args.batch_size,
        seed=args.seed,
        ncentroids=args.ncentroids,
        num_keys_to_add_at_a_time=args.num_keys_to_add_at_a_time
    )
    sys.exit(main(config))