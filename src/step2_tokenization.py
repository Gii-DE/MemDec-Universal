"""
Dataset Tokenization
"""

import argparse, time, json
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizer


# --- LOGGING ---
from src.utils import get_logger, log_blank_line, get_pipeline_value, extract_model_identifier, project_rel
logger = get_logger('2_tokenization')


# --- CONFIG ---
from src.utils import cleanup_temp_files, get_model_config, resolve_base_model, PROJECT_ROOT, DATA_DIR
CORPUS_NAME = get_pipeline_value("pipeline.corpus_name")
DATASET_CLEANED = get_pipeline_value("steps.step2_tokenization.dataset_cleaned", CORPUS_NAME)
CLEANED_DATA_DIR = DATA_DIR / DATASET_CLEANED if DATASET_CLEANED else None


# --- HELPERS ---
def initialize_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Initialize and return a tokenizer for the specified model.
    
    Args:
        model_name: Hugging Face model identifier
        
    Returns:
        Configured tokenizer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_batch(examples: Dict[str, Any], tokenizer: PreTrainedTokenizer, max_length: int = 512) -> Dict[str, Any]:
    """
    Tokenize a batch of text examples.

    Args:
        examples: Batch containing 'text' field
        tokenizer: Tokenizer instance to use
        max_length: Maximum sequence length for truncation

    Returns:
        Dictionary with tokenized inputs, attention masks, and labels
    """
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def load_cleaned_dataset(cleaned_path: str) -> Dataset:
    """
    Load the locally cleaned dataset from step1_cleaning.
    
    Args:
        cleaned_path: Path to the cleaned dataset directory
        
    Returns:
        Loaded dataset with text column
        
    Raises:
        ValueError: If dataset doesn't have required 'text' column
    """
    logger.info(f"🧹 Loading cleaned dataset: {cleaned_path}")
    try:
        dataset = load_from_disk(cleaned_path)
        logger.info(f"✅ Loaded {len(dataset)} cleaned examples")
    except Exception as e:
        logger.error(f"❌ Failed to load: {e}")
        logger.error("💡 Run: python -m src.step1_cleaning --hf-dataset <dataset_name> --hf-config <config_name>")
        raise

    if "text" not in dataset.column_names:
        logger.error(f"❌ Missing 'text' column. Found: {dataset.column_names}")
        raise ValueError("Dataset must have 'text' column from step1")
    logger.info(f"   Columns: {dataset.column_names}")
    return dataset


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    num_workers: Optional[int] = None,
    batch_size: int = 3000,
    output_path: Optional[Path] = None
) -> Dataset:
    """
    Tokenize the cleaned dataset and save as Arrow format.
    
    Args:
        dataset: Dataset to tokenize
        tokenizer: Tokenizer to use
        num_workers: Number of workers for tokenization
        batch_size: Batch size for tokenization
        output_path: Optional path to save Arrow dataset
        
    Returns:
        Tokenized dataset
    """
    num_workers = num_workers or 2
    logger.info(f"🔄 Tokenizing (batch_size={batch_size}, workers={num_workers})...")
    start_time = time.time()
    tokenized_dataset = dataset.map(
        lambda x: tokenize_batch(x, tokenizer),
        batched=True,
        batch_size=batch_size,
        num_proc=num_workers,
        remove_columns=dataset.column_names,
        load_from_cache_file=True,
        desc="Tokenizing dataset"
    )
    elapsed = time.time() - start_time
    logger.info(f"✅ Tokenization completed in {elapsed:.1f} seconds")
    if output_path:
        arrow_dir = output_path / "arrow_data"
        logger.info(f"📊 Saving as Arrow format ({arrow_dir})...")
        tokenized_dataset.save_to_disk(str(arrow_dir))
        try:
            arrow_display = arrow_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            arrow_display = arrow_dir
        logger.info(f"✅ Arrow dataset saved to {arrow_display}")
    return tokenized_dataset


def save_train_test_json(
    tokenized_dataset: Dataset,
    output_path: str,
    cleaned_path: str,
    tokenizer: PreTrainedTokenizer,
    train_test_split: float = 0.8
) -> Dict[str, Any]:
    """
    Split tokenized dataset and save train/test JSON files.
    
    Args:
        tokenized_dataset: Tokenized dataset to split
        output_path: Path to save the JSON files
        cleaned_path: Path to the cleaned dataset
        tokenizer: Tokenizer used for tokenization
        train_test_split: Fraction of data to use for training (default: 0.8)
    
    Returns:
        Dictionary containing metadata about the saved files
    """
    output_path = Path(output_path)
    train_file = output_path / "train.json"
    test_file = output_path / "test.json"
    state_file = output_path / "dataset_metadata.json"
    if all(f.exists() for f in [train_file, test_file, state_file]):
        logger.info("✅ train/test JSON files already exist!")
        try:
            with open(state_file, 'r') as f:
                metadata = json.load(f)
                logger.info(f" - Train: {train_file} ({metadata.get('train_samples', '?')} samples)")
                logger.info(f" - Test: {test_file} ({metadata.get('test_samples', '?')} samples)")
                return metadata
        except Exception as e:
            logger.warning(f"Could not read state.json: {e}")
    
    tokenizer_model = tokenizer.name_or_path
    tokenizer_family = extract_model_identifier(tokenizer_model)
    compatible_models = []
    try:
        config_file = PROJECT_ROOT / "config" / "model_config.json"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f) 
            models_dict = config_data.get("models", {})
            for key, model_info in models_dict.items():
                if tokenizer_family in key or tokenizer_family in model_info.get("name", ""):
                    compatible_models.append(model_info.get("name"))
    except Exception as e:
        logger.warning(f"Could not resolve compatible models from config: {e}")
    if not compatible_models:
        compatible_models = [tokenizer_model]

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Splitting dataset ({train_test_split*100:.0f}% train / {(1-train_test_split)*100:.0f}% test, seed=42)...")
    splits = tokenized_dataset.train_test_split(
        test_size=1-train_test_split,
        seed=42,
        shuffle=True
    )
    train_dataset = splits['train']
    test_dataset = splits['test']
    logger.info(f"✅ Split: {len(train_dataset)} train, {len(test_dataset)} test")
    
    def save_dataset(dataset: Dataset, output_file: Path, desc: str) -> int:
        """Save dataset to JSON file with dstore_range for KNN mapping."""
        total_samples = len(dataset)
        logger.info(f"{desc}: {total_samples:,} examples")
        all_examples = []
        cumulative_tokens = 0
        for i in tqdm(range(0, total_samples, 3000), desc=desc, unit="batches"):
            batch = dataset[i:i+3000]
            for j in range(len(batch['input_ids'])):
                seq_len = len(batch['input_ids'][j])
                dstore_range = [cumulative_tokens, cumulative_tokens + seq_len]
                cumulative_tokens += seq_len
                all_examples.append({
                    'input_ids': batch['input_ids'][j],
                    'attention_mask': batch['attention_mask'][j],
                    'labels': batch['input_ids'][j].copy(),
                    'dstore_range': dstore_range
                })
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_examples, f, ensure_ascii=False, indent=2)
        return total_samples
    
    train_samples = save_dataset(train_dataset, train_file, "Writing training data")
    test_samples = save_dataset(test_dataset, test_file, "Writing test data")
    metadata = {
        'train_file': project_rel(str(train_file.absolute())),
        'test_file': project_rel(str(test_file.absolute())),
        'train_samples': train_samples,
        'test_samples': test_samples,
        'total_samples': train_samples + test_samples,
        'cleaned_dataset': project_rel(cleaned_path),
        'tokenizer_model': tokenizer_model,
        'tokenizer_family': tokenizer_family,
        'compatible_models': compatible_models,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(state_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    tokenizer_dir = output_path / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))
    try:
        tokenizer_display = tokenizer_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        tokenizer_display = tokenizer_dir
    logger.info(f"✅ Tokenizer saved to {tokenizer_display}")
    return metadata


# --- MAIN ---
def main(
    dataset_cleaned: Optional[str], 
    base_model: str = None, 
    num_workers: Optional[int] = None,
    train_test_split: float = 0.8
) -> None:
    """
    Main function to tokenize the locally cleaned dataset and saves it as Arrow & JSON files.
    
    Args:
        dataset_cleaned: Name of the cleaned dataset to tokenize (required; falls back
            to steps.step2_tokenization.dataset_cleaned or pipeline.corpus_name via CLI)
        base_model: Model to use for tokenization (default: None)
        num_workers: Number of workers for tokenization (default: None)
        train_test_split: Fraction of data to use for training (default: 0.8)
    
    Returns:
        None
    
    Raises:
        SystemExit: If no dataset name is provided
    """
    if dataset_cleaned is None:
        logger.error("❌ No cleaned dataset specified: pass <dataset_cleaned> or set steps.step2_tokenization.dataset_cleaned / pipeline.corpus_name in pipeline_config.yaml")
        raise SystemExit(1)
    cleaned_path = DATA_DIR / dataset_cleaned
    model_keyword = extract_model_identifier(base_model)
    tokenized_path = DATA_DIR / f"{dataset_cleaned}_tokenized-{model_keyword}"
    log_blank_line(logger)
    logger.info("="*60)
    logger.info("🧾 STEP 2: Dataset Tokenization")
    logger.info("="*60)
    if tokenized_path.exists():
        logger.info(f"🔄 Found existing tokenized dataset: {tokenized_path}")
        try:
            metadata_file = tokenized_path / "dataset_metadata.json"
            if not metadata_file.exists():
                metadata_file = tokenized_path / "state.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                logger.info(f"✅ Loaded tokenized dataset:")
                logger.info(f"   Train: {metadata.get('train_samples', '?'):,} samples")
                logger.info(f"   test:   {metadata.get('test_samples', '?'):,} samples")
            logger.info("="*60)
            logger.info("ℹ️ Next step: Pretraining")
            logger.info(f" Run: python -m src.step3_pretraining {dataset_cleaned}_tokenized-{model_keyword}")
            logger.info("="*60)
            return
        except:
            logger.warning("⚠️ Existing tokenized dataset corrupted, reprocessing...")
    logger.info(f"🔤 Tokenizer: {base_model}")
    try:
        # 1. Load cleaned dataset
        dataset = load_cleaned_dataset(str(cleaned_path))
        # 2. Tokenize with Arrow saving
        tokenizer = initialize_tokenizer(base_model)
        tokenized_path.mkdir(parents=True, exist_ok=True)
        tokenized_dataset = tokenize_dataset(
            dataset,
            tokenizer,
            num_workers=num_workers,
            output_path=tokenized_path
        )
        # 3. Save Train-Test Split as JSON (for step3_pretraining.py & step5_evaluation.py)
        logger.info("📄 Creating Train-Test Split in JSON format...")
        metadata = save_train_test_json(
            tokenized_dataset,
            str(tokenized_path),
            str(cleaned_path),
            tokenizer,
            train_test_split=train_test_split
        )
        logger.info("="*60)
        logger.info(f"📁 Output directory: {tokenized_path}")
        logger.info("📊 Statistics:")
        logger.info(f" - Input examples: {len(dataset):,}")
        logger.info(f" - Kept entries: {metadata['train_samples'] + metadata['test_samples']:,}")
        logger.info(f" - Train: {metadata['train_samples']:,}")
        logger.info(f" - test: {metadata['test_samples']:,}")
        logger.info("📦 Saved formats:")
        logger.info(f"  Arrow (HF format): {Path(tokenized_path).relative_to(PROJECT_ROOT)}/arrow_data")
        logger.info(f"  JSON (Step3): {Path(tokenized_path).relative_to(PROJECT_ROOT)}/train.json + test.json")
        logger.info("="*60)
        logger.info("ℹ️ Next step: Pretraining")
        logger.info(f" Run: python -m src.step3_pretraining {dataset_cleaned}_tokenized-{model_keyword}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Error during tokenization: {e}")
        raise

    finally:
        try:
            cleanup_temp_files(logger)
        except Exception as e:
            if logger:
                logger.error(f"Error during cleanup: {e}")
            else:
                print(f"Error during cleanup: {e}")


# --- CLI ---
def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments (use default config values if not provided) for tokenization of cleaned dataset.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Tokenize cleaned dataset for training")
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--model', type=str, default=None)
    cli_model = base_parser.parse_known_args()[0].model
    # Determine base_model: .env > CLI > step config > pipeline default > system default
    config_base_model = resolve_base_model(
        cli_model=cli_model,
        step_config_key="steps.step2_tokenization",
    )
    config_dataset_cleaned = get_pipeline_value("steps.step2_tokenization.dataset_cleaned", CORPUS_NAME)
    config_train_test_split = get_pipeline_value("steps.step2_tokenization.train_test_split", 0.8)
    config_num_workers = get_pipeline_value("steps.step2_tokenization.num_workers", 2)
    parser.add_argument("--model", default=config_base_model,
                        help=f"Tokenizer model (gemma3/qwen3.5/smollm3) [default: {config_base_model}]")
    parser.add_argument("dataset_cleaned", nargs="?", default=config_dataset_cleaned,
                        help=f"Name from step1 (e.g., cases2022_1k) [default: {config_dataset_cleaned}]")
    parser.add_argument("--train-test-split", type=float, default=config_train_test_split,
                        help=f"Train/test split ratio [default: {config_train_test_split}]")
    parser.add_argument("--num-workers", type=int, default=config_num_workers, 
                        help=f"Number of worker processes [default: {config_num_workers}]")
    args = parser.parse_args()
    args.model = config_base_model
    if args.dataset_cleaned is None:
        parser.error("dataset_cleaned is required: pass it or set steps.step2_tokenization.dataset_cleaned / pipeline.corpus_name in pipeline_config.yaml")
    return args


if __name__ == '__main__':
    args = parse_arguments()
    model_keyword = extract_model_identifier(args.model)
    model_info = get_model_config(model_keyword)
    main(
        base_model=model_info['name'],
        dataset_cleaned=args.dataset_cleaned,
        train_test_split=args.train_test_split,
        num_workers=args.num_workers,
    )