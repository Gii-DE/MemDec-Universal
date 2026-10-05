"""
MemDec Training
"""

import os, sys, io, logging, contextlib
os.environ['FORCE_TORCHAUDIO_AVAILABLE'] = '0'
sys.modules['torchaudio'] = None
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
import torchvision.io
class DummyVideoReader:
    def __init__(self, *args, **kwargs): pass
torchvision.io.VideoReader = DummyVideoReader
# ----------------------------------------------------------------------
import os, sys, gc, math, time, argparse, torch
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_from_disk, load_dataset

from src.utils import get_pipeline_value
if str(get_pipeline_value("steps.step4_training.no_unsloth", False)).strip().lower() in ("1", "true", "yes"):
    os.environ["NO_UNSLOTH"] = "1"
with contextlib.redirect_stdout(io.StringIO()):
    from MemoryDecoder.train_memdec import parse_args, main as train_memdec_main


# --- LOGGING ---
from src.utils import get_logger, log_blank_line
logger = get_logger('4_training')


# --- CONFIG ---
from src.utils import resolve_base_model, get_model_config, get_knowledge_base_paths, get_dataset_tokenizer_model, normalize_model_name, extract_model_identifier, PROJECT_ROOT, DATA_DIR, KNOWLEDGE_BASE_DIR, OUTPUT_DIR

CORPUS_NAME = get_pipeline_value("pipeline.corpus_name")
TOKENIZED_DATA_NAME = get_pipeline_value(
    "steps.step3_pretraining.tokenized_data",
    f"{CORPUS_NAME}_tokenized" if CORPUS_NAME else None
)
TOKENIZED_DATA_DIR = DATA_DIR / TOKENIZED_DATA_NAME if TOKENIZED_DATA_NAME else None

@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    # Paths
    base_model: str = None
    tokenized_data_path: Optional[str] = str(TOKENIZED_DATA_DIR) if TOKENIZED_DATA_DIR else None
    knn_datastore_path: str = str(KNOWLEDGE_BASE_DIR)
    output_dir: str = str(OUTPUT_DIR)
    checkpoint_dir: Optional[str] = None
    # MemDec params
    block_size: int = 512   # context/sequence length
    k_neighbors: int = 4
    alpha: float = 0.5
    lmbda: float = 0.3      # affects tests & evaluation
    # Training params
    max_steps: int = 1500    # reduced from 5000 for testing
    num_train_epochs: int = 2   # auto-raised in train_memdec when too small for max_steps
    checkpointing_steps: int = 50   # saved checkpoints interval
    batch_size: int = 2    # GPU default; CPU fallback
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    no_unsloth: bool = False   # True disables Unsloth kernels (plain HF Transformers training)
    # Optimization params
    learning_rate: float = 1.5e-4
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    warmup_steps: int = 50
    seed: int = 42


# --- HELPERS ---
def setup_environment(config: TrainingConfig) -> None:
    """
    Set up environment variables and logging.
    
    Args:
        config: Training configuration containing all parameters

    Returns:
        None
    """
    log_blank_line(logger)
    logger.info("="*60)
    logger.info("⚡ STEP 4: MemDec Training")
    logger.info("="*60)
    logger.info("Starting training with configuration:")
    for k, v in config.__dict__.items():
        logger.info(f"  {k}: {v}")
    logger.info("Setting environment variables...")
    os.environ.update({
        "KNN_DATASTORE_PATH": str(config.knn_datastore_path),
        "K_NEIGHBORS": str(config.k_neighbors),
        "BLOCK_SIZE": str(config.block_size)
    })
    logger.info(f"Environment variables set: K_NEIGHBORS={os.environ.get('K_NEIGHBORS')}, BLOCK_SIZE={os.environ.get('BLOCK_SIZE')}")


def prepare_dataset(config: TrainingConfig) -> str:
    """
    Prepare and validate the training dataset.
    
    Args:
        config: Training configuration with dataset path
        
    Returns:
        Path to the training JSON file
        
    Raises:
        Exception: If dataset loading or validation fails
    """
    os.makedirs(config.tokenized_data_path, exist_ok=True)
    train_file = os.path.join(config.tokenized_data_path, "train.json")
    try:
        dataset = load_dataset('json', data_files={'train': train_file})['train']
        required_columns = {'input_ids', 'attention_mask'}
        missing_columns = required_columns - set(dataset.column_names)
        if missing_columns:
            raise ValueError(f"Missing required columns in dataset: {missing_columns}")
        dataset.save_to_disk(config.tokenized_data_path)
        return train_file
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def setup_training_args(config: TrainingConfig, train_file: str) -> List[str]:
    """
    Prepare command line arguments for training.

    Args:
        config: Training configuration
        train_file: Path to the training JSON file

    Returns:
        List of command line arguments for MemoryDecoder training
    """
    kb_paths = get_knowledge_base_paths(config.base_model, config.knn_datastore_path)
    logger.info(f"Knowledge base files:")
    logger.info(f"  dstore: {kb_paths['dstore']}")
    logger.info(f"  index: {kb_paths['index']}")
    args = [
        f"--model_name_or_path={config.base_model}",
        f"--train_file={train_file}",
        f"--output_dir={config.output_dir}",
        f"--per_device_train_batch_size={config.per_device_train_batch_size}",
        f"--learning_rate={config.learning_rate}",
        f"--weight_decay={config.weight_decay}",
        f"--num_train_epochs={config.num_train_epochs}",
        f"--max_train_steps={config.max_steps}",
        f"--gradient_accumulation_steps={config.gradient_accumulation_steps}",
        f"--lr_scheduler_type={config.lr_scheduler_type}",
        f"--num_warmup_steps={config.warmup_steps}",
        f"--seed={config.seed}",
        f"--checkpointing_steps={config.checkpointing_steps}",
        "--overwrite_cache",
        f"--knn_save_path={kb_paths['dstore']}",
        f"--lmbda={config.lmbda}",
        f"--block_size={config.block_size}",
        "--project_name=memory_decoder_training",
        "--group_name=memory_decoder",
    ]
    if config.no_unsloth:
        args.append("--no_unsloth")
    # Handle checkpoint resumption
    if not config.checkpoint_dir and os.path.exists(config.output_dir):
        checkpoints = [d for d in os.listdir(config.output_dir) 
                      if os.path.isdir(os.path.join(config.output_dir, d)) and ('checkpoint' in d or d.startswith('step_'))]
        if checkpoints:

            def _extract_number(checkpoint_name):
                for separator in ['-', '_']:
                    if separator in checkpoint_name:
                        parts = checkpoint_name.split(separator)
                        if parts[-1].isdigit():
                            return int(parts[-1])
                return 0
            
            latest_checkpoint = max(checkpoints, key=_extract_number)
            config.checkpoint_dir = os.path.join(config.output_dir, latest_checkpoint)
    elif config.checkpoint_dir and not os.path.isabs(config.checkpoint_dir):
        if not config.checkpoint_dir.startswith(str(config.output_dir)):
            config.checkpoint_dir = os.path.join(config.output_dir, config.checkpoint_dir)
    if config.checkpoint_dir:
        args.append(f"--resume_from_checkpoint={config.checkpoint_dir}")
        try:
            logger.info(f"Resuming training from checkpoint: {Path(config.checkpoint_dir).relative_to(PROJECT_ROOT)}")
        except ValueError:
            logger.info(f"Resuming training from checkpoint: {Path(config.checkpoint_dir)}")
    else:
        logger.info("Starting new training")
    return args


def validate_dataset(dataset_path: str) -> None:
    """
    Validate the dataset has all required columns and format.
    
    Args:
        dataset_path: Path to the dataset directory

    Returns:
        int: Number of examples in the dataset
    
    Raises:
        ValueError: If required columns are missing
    """
    logger.info(f"Validating dataset at {dataset_path}")
    dataset = load_from_disk(dataset_path)
    required_columns = {'input_ids', 'attention_mask', 'labels'}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")
    dataset.set_format(type='torch', columns=list(required_columns))
    logger.info(f"Dataset validated with {len(dataset)} examples")
    return len(dataset)


# --- MAIN ---
def main(config: TrainingConfig) -> None:
    """
    Main training function.
    
    Args:
        config: Complete training configuration
    
    Returns:
        None
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    setup_environment(config)
    os.makedirs(config.output_dir, exist_ok=True)
    dataset_tokenizer = get_dataset_tokenizer_model(config.tokenized_data_path)
    if dataset_tokenizer:
        config_normalized = normalize_model_name(config.base_model)
        dataset_normalized = normalize_model_name(dataset_tokenizer)
        if config_normalized != dataset_normalized:
            logger.error(f"Model mismatch detected!")
            logger.error(f"  Dataset tokenizer model: {dataset_tokenizer}")
            logger.error(f"  Specified base model: {config.base_model}")
            raise ValueError(
                f"Base model '{config.base_model}' does not match dataset tokenizer '{dataset_tokenizer}'."
            )
        else:
            logger.info(f"Base model matches dataset tokenizer: {config.base_model}")
    train_file = prepare_dataset(config)
    num_examples = validate_dataset(config.tokenized_data_path)
    steps_per_epoch = math.ceil(math.ceil(num_examples / config.per_device_train_batch_size) / config.gradient_accumulation_steps)
    min_epochs = math.ceil(config.max_steps / steps_per_epoch) + 1 if steps_per_epoch else 0
    if steps_per_epoch and config.num_train_epochs < min_epochs:
        logger.info(f"num_train_epochs={config.num_train_epochs} will be auto-raised to ~{min_epochs} in train_memdec "
                    f"({steps_per_epoch} steps/epoch needed for max_steps={config.max_steps})")
    args = setup_training_args(config, train_file)
    original_argv = sys.argv.copy()
    start_time = time.time()
    try:
        sys.argv = [original_argv[0]] + args
        args = parse_args()
        progress_bar = tqdm(
            desc="Training",
            total=config.max_steps,
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}',
            dynamic_ncols=True
        )
        try:
            train_memdec_main(progress_bar=progress_bar)
        except Exception as e:
            progress_bar.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.error(f"Training failed: {str(e)}")
            raise
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/3600:.2f} hours")
        # Find latest checkpoint for next steps
        output_path = Path(config.output_dir)
        if output_path.exists():
            checkpoints = [d for d in output_path.iterdir() 
                          if d.is_dir() and d.name.startswith("step_")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("_")[1]))
                checkpoint_name = latest_checkpoint.name
                logger.info("=" * 60)
                logger.info("ℹ️  Next step: Qualitative Evaluation")
                logger.info(f"   Run: python -m src.step5_testing --checkpoint {checkpoint_name}")
                logger.info("=" * 60)
        
    except Exception as e:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        if 'progress_bar' in locals():
            progress_bar.close()


# --- CLI ---
def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments (use default config values if not provided) for MemDec-Module training.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('tokenized_data', nargs='?', default=None,
                          help='Name of the tokenized dataset directory [default: %(default)s]')
    args = base_parser.parse_known_args()[0]
    default_config = TrainingConfig()
    if args.tokenized_data is None:
        args.tokenized_data = get_pipeline_value("steps.step4_training.tokenized_data", default_config.tokenized_data_path)
    # Determine base_model: CLI > step config > tokenized data > pipeline default > env
    base_parser_model = argparse.ArgumentParser(add_help=False)
    base_parser_model.add_argument('--model', type=str, default=None)
    cli_model = base_parser_model.parse_known_args()[0].model
    config_base_model = resolve_base_model(
        cli_model=cli_model,
        step_config_key="steps.step4_training",
        tokenized_data=args.tokenized_data,
    )
    config_max_steps = get_pipeline_value("steps.step4_training.max_steps", default_config.max_steps)
    config_checkpointing_steps = get_pipeline_value("steps.step4_training.checkpointing_steps", default_config.checkpointing_steps)
    default_per_device = 1 if not torch.cuda.is_available() else default_config.per_device_train_batch_size
    config_batch_size = get_pipeline_value("steps.step4_training.batch_size", default_per_device)
    config_per_device_train_batch_size = get_pipeline_value("steps.step4_training.per_device_train_batch_size", config_batch_size)
    config_gradient_accumulation_steps = get_pipeline_value("steps.step4_training.gradient_accumulation_steps", default_config.gradient_accumulation_steps)
    config_seed = get_pipeline_value("steps.step4_training.seed", default_config.seed)
    config_learning_rate = get_pipeline_value("steps.step4_training.learning_rate", default_config.learning_rate)
    config_k_neighbors = get_pipeline_value("steps.step4_training.k_neighbors", default_config.k_neighbors)
    config_alpha = get_pipeline_value("steps.step4_training.alpha", default_config.alpha)
    config_lmbda = get_pipeline_value("steps.step4_training.lmbda", default_config.lmbda)
    config_no_unsloth = get_pipeline_value("steps.step4_training.no_unsloth", default_config.no_unsloth)
    config_num_train_epochs = get_pipeline_value("steps.step4_training.num_train_epochs", default_config.num_train_epochs)
    parser = argparse.ArgumentParser(description='Train the MemDec-Module model')

    path_group = parser.add_argument_group('Path Arguments')
    # Variable defaults via pipeline_config
    path_group.add_argument('--model', type=str, default=config_base_model,
                         help=f'Base model choice (gemma3/qwen3.5/smollm3) [default: {config_base_model}]')
    path_group.add_argument('tokenized_data', nargs='?', default=args.tokenized_data,
                         help=f'Name of the tokenized dataset directory [default: {args.tokenized_data}]')
    path_group.add_argument('--checkpoint', type=str, default=default_config.checkpoint_dir,
                         help=f'Path to the checkpoint directory [default: {default_config.checkpoint_dir}]')
    train_group = parser.add_argument_group('Training Arguments')
    train_group.add_argument('--max-steps', type=int, default=config_max_steps,
                         help=f'Maximum number of training steps [default: {config_max_steps}]')
    train_group.add_argument('--checkpointing-steps', type=int, default=config_checkpointing_steps,
                         help=f'Save checkpoint every N steps [default: {config_checkpointing_steps}]')
    train_group.add_argument('--num-train-epochs', type=int, default=config_num_train_epochs,
                         help=f'Number of training epochs (upper loop bound; --max-steps caps updates) [default: {config_num_train_epochs}]')
    train_group.add_argument('--batch-size', type=int, default=config_batch_size,
                         help=f'Batch size for training [default: {config_batch_size}]')
    train_group.add_argument('--per-device-train-batch-size', type=int, default=config_per_device_train_batch_size,
                         help=f'Per-device batch size for training [default: {config_per_device_train_batch_size}]')
    train_group.add_argument('--gradient-accumulation-steps', type=int, default=config_gradient_accumulation_steps,
                         help=f'Number of gradient accumulation steps [default: {config_gradient_accumulation_steps}]')
    train_group.add_argument('--learning-rate', type=float, default=config_learning_rate,
                         help=f'Learning rate [default: {config_learning_rate}]')
    train_group.add_argument('--k-neighbors', type=int, default=config_k_neighbors,
                         help=f'Number of KNN neighbors [default: {config_k_neighbors}]')
    train_group.add_argument('--alpha', type=float, default=config_alpha,
                         help=f'Mixing weight for MemDec [default: {config_alpha}]')
    train_group.add_argument('--lmbda', type=float, default=config_lmbda,
                         help=f'Lambda for MemDec [default: {config_lmbda}]')
    train_group.add_argument('--seed', type=int, default=config_seed,
                         help=f'Random seed [default: {config_seed}]')
    train_group.add_argument('--no-unsloth', '--no_unsloth', dest='no_unsloth', action='store_true', default=config_no_unsloth,
                         help=f'Disable Unsloth kernels, train with plain HF Transformers [default: {config_no_unsloth}]')
    # Fixed defaults via PretrainConfig (cannot be changed via CLI)
    path_group.add_argument('--knn-datastore-path', type=str, default=default_config.knn_datastore_path,
                         help=f'[FIXED] Path to the KNN datastore [default: {default_config.knn_datastore_path}]')
    path_group.add_argument('--output-dir', type=str, default=default_config.output_dir,
                         help=f'[FIXED] Output directory for model checkpoints [default: {default_config.output_dir}]')
    train_group.add_argument('--block-size', type=int, default=default_config.block_size,
                         help=f'[FIXED] Context/sequence length for training [default: {default_config.block_size}]')
    train_group.add_argument('--weight-decay', type=float, default=default_config.weight_decay,
                         help=f'[FIXED] Weight decay [default: {default_config.weight_decay}]')
    train_group.add_argument('--warmup-steps', type=int, default=default_config.warmup_steps,
                         help=f'[FIXED] Number of warmup steps [default: {default_config.warmup_steps}]')
    train_group.add_argument('--lr-scheduler-type', type=str, default=default_config.lr_scheduler_type,
                         help=f'[FIXED] Learning rate scheduler type [default: {default_config.lr_scheduler_type}]')
    args = parser.parse_args()
    if not args.tokenized_data:
        parser.error("tokenized_data is required: pass it or set steps.step4_training.tokenized_data / pipeline.corpus_name in pipeline_config.yaml")
    if args.knn_datastore_path != default_config.knn_datastore_path:
        parser.error(f"--knn-datastore-path is fixed to '{default_config.knn_datastore_path}' and cannot be changed via CLI.")
    if args.output_dir != default_config.output_dir:
        parser.error(f"--output-dir is fixed to '{default_config.output_dir}' and cannot be changed via CLI.")
    if args.block_size != default_config.block_size:
        parser.error(f"--block-size is fixed to '{default_config.block_size}' and cannot be changed via CLI.")
    if args.weight_decay != default_config.weight_decay:
        parser.error(f"--weight-decay is fixed to '{default_config.weight_decay}' and cannot be changed via CLI.")
    if args.warmup_steps != default_config.warmup_steps:
        parser.error(f"--warmup-steps is fixed to '{default_config.warmup_steps}' and cannot be changed via CLI.")
    if args.lr_scheduler_type != default_config.lr_scheduler_type:
        parser.error(f"--lr-scheduler-type is fixed to '{default_config.lr_scheduler_type}' and cannot be changed via CLI.")
    return args


if __name__ == '__main__':
    logger = get_logger('4_training')
    args = parse_arguments()
    token_path = Path(args.tokenized_data)
    if not token_path.is_absolute():
        if not str(token_path).startswith("dataset"):
            tokenized_data_path = PROJECT_ROOT / 'dataset' / token_path
        else:
            tokenized_data_path = PROJECT_ROOT / token_path
    else:
        tokenized_data_path = token_path
    model_keyword = extract_model_identifier(args.model)
    model_info = get_model_config(model_keyword)
    config = TrainingConfig(
        base_model=model_info['name'],
        tokenized_data_path=str(tokenized_data_path),
        checkpoint_dir=args.checkpoint,
        batch_size=args.batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        k_neighbors=args.k_neighbors,
        alpha=args.alpha,
        lmbda=args.lmbda,
        seed=args.seed,
        checkpointing_steps=args.checkpointing_steps,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        no_unsloth=args.no_unsloth,
    )
    main(config)