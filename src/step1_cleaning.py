"""
Dataset Cleaning & Creation (Hugging Face Dataset)
"""

import argparse, os, re, shutil, unicodedata, hashlib
from pathlib import Path
from bs4 import BeautifulSoup
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets


# --- LOGGING ---
from src.utils import get_logger, log_blank_line, _load_usecase_config, DATASET_CONFIG_FILE, get_pipeline_value
logger = get_logger('1_preparation')


# --- CONFIG ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# --- HELPERS ---
def clean_html(text: str) -> str:
    """
    Remove HTML/Markup and extract clean text.
    
    Args:
        text: Text containing HTML markup
        
    Returns:
        Clean text with HTML removed
    """
    try:
        soup = BeautifulSoup(text, "html.parser")
        if "<invalid" in text:
            return text
        result = soup.get_text(" ")
        result = re.sub(r'\s+([.!?,;:])', r'\1', result)
        result = re.sub(r'\s+', ' ', result).strip()
        return result
    except Exception as e:
        logger.debug(f"Error cleaning HTML: {e}")
        return text


def remove_urls_emails(text: str) -> str:
    """
    Remove URLs and email addresses from text.
    
    Args:
        text (str): Input text potentially containing URLs/emails
        
    Returns:
        str: Text with URLs and emails removed
    """
    text = re.sub(r'\[[^\]]+\]\([^)]+\)', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+?(?=[,;!?\s]|$)', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return text
    

def fix_encoding_errors(text: str) -> str:
    """
    Fix common encoding errors and remove invalid Unicode characters.
    
    Args:
        text: Text with potential encoding issues
        
    Returns:
        Text with encoding errors fixed
    """
    text = text.replace('\ufffd', '')
    text = text.replace('\x00', '')
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    return text


def normalize_text(text: str) -> str:
    """
    Normalize text: fix spaces, unicode, whitespace, and punctuation.
    
    Args:
        text: Input text to be normalized
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace('\xa0', ' ').replace('\u200b', '')
    # Language-agnostic patterns using Unicode letter class \w
    text = re.sub(r'(\d+)\s+(\w)\s*\.\s*(\w)\s*\.\s*(\w)\s*\.\s*:\s*(\d+)', 
                  r'\1\2.\3.\4.:\5', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\w)\s*\.\s*(\w)\s*\.\s*(\w)\s*\.', 
                  r'\1.\2.\3.', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(\w)\s*\.\s*(\w)\s*\.(\w)\s*\.', 
                  r'\1.\2.\3.', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*:\s*', ':', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    text = re.sub(r'([.!?,;])\1{3,}', r'\1\1\1', text)
    return text


def is_valid_text(text: str, min_len: int = None, min_alpha: float = None) -> bool:
    """
    Validate text quality based on length, alphabet ratio, and repetition rate.

    Args:
        text: Input text to validate
        min_len: Minimum length threshold (from config if None)
        min_alpha: Minimum alphabet ratio (from config if None)
        
    Returns:
        True if text is valid, False otherwise
    """
    if min_len is None:
        cfg_val = get_pipeline_value("steps.step1_cleaning.min_text_length", 10)
        min_len = cfg_val if cfg_val is not None else 10
    if min_alpha is None:
        cfg_val = get_pipeline_value("steps.step1_cleaning.min_alpha_ratio", 0.3)
        min_alpha = cfg_val if cfg_val is not None else 0.3
    if not text or len(text) < min_len:
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    if alpha_ratio < min_alpha:
        return False
    if len(text) > 200:
        words = text.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return False  
    return True


def get_dataset_config(hf_config: str = None, hf_dataset: str = None) -> dict:
    """
    Get dataset configuration from dataset_config.yaml.
    
    Args:
        hf_config: Hugging Face dataset config name or output_name
        hf_dataset: Hugging Face dataset name (for disambiguation when multiple datasets use same config)
        
    Returns:
        Dictionary with dataset configuration (dataset_name, config, split, output_name, field_mapping)
    """
    try:
        config = _load_usecase_config("dataset_config.yaml", DATASET_CONFIG_FILE)
        hf_config_data = config.get("huggingface", {})
        datasets_list = hf_config_data.get("datasets", [])
        if datasets_list:
            if hf_dataset and hf_config:
                for ds in datasets_list:
                    if ds.get("dataset_name") == hf_dataset and (
                        ds.get("output_name") == hf_config or ds.get("config") == hf_config
                    ):
                        return ds
            if hf_config:
                matches = [ds for ds in datasets_list
                           if ds.get("output_name") == hf_config or ds.get("config") == hf_config]
                if len(matches) > 1:
                    names = [f"{ds.get('dataset_name')} (output_name: {ds.get('output_name')})" for ds in matches]
                    logger.error(
                        f"❌ Ambiguous --hf-config '{hf_config}': matches {len(matches)} datasets: {names}. "
                        "Disambiguate via --hf-dataset or a unique output_name"
                    )
                    raise SystemExit(1)
                if matches:
                    return matches[0]
            if hf_dataset:
                for ds in datasets_list:
                    if ds.get("dataset_name") == hf_dataset:
                        return ds
    except (FileNotFoundError, ImportError, KeyError):
        pass
    return None


def get_field_mapping(hf_config: str = None, hf_dataset: str = None, dataset_name: str = None) -> dict:
    """
    Get field mapping configuration for different datasets.
    
    Args:
        hf_config: Hugging Face dataset config name or output_name
        hf_dataset: Hugging Face dataset name (for disambiguation)
        dataset_name: Dataset name for local_file lookup
        
    Returns:
        Dictionary mapping generic field names to dataset-specific field names
    """
    ds_config = get_dataset_config(hf_config, hf_dataset)
    if ds_config:
        return ds_config.get("field_mapping", {})
    if dataset_name:
        try:
            config = _load_usecase_config("dataset_config.yaml", DATASET_CONFIG_FILE)
            for section in ("local_files", "datasets"):
                entries = config.get(section, {}) or {}
                if dataset_name in entries:
                    return entries[dataset_name].get("field_mapping", {})
        except Exception:
            pass
    return {
        "text": ["markdown_content", "content", "text"],
        "meta_category": ["category", "court", "type"],
        "meta_date": ["date"],
        "id": ["id"]
    }


def extract_field(example: dict, field_mapping: dict, generic_name: str, default=None):
    """
    Extracts and concatenates all fields mapped to generic_name (e.g. user_question + response).
    
    Args:
        example: Dataset example dictionary
        field_mapping: Field mapping configuration
        generic_name: Generic field name to look up
        default: Default value if field not found
        
    Returns:
        Extracted field value or default
    """
    possible_fields = field_mapping.get(generic_name, [generic_name])
    clean_example_keys = {k.strip(): k for k in example.keys()}
    if generic_name in ["meta_category", "meta_date", "id"]:
        for field in possible_fields:
            field_clean = field.strip()
            if field_clean in clean_example_keys:
                actual_key = clean_example_keys[field_clean]
                val = example.get(actual_key)
                if val is not None:
                    val_str = str(val).strip()
                    if val_str and val_str.lower() != "none":
                        return val
        return default
    extracted_parts = []
    for field in possible_fields:
        field_clean = field.strip()
        if field_clean in clean_example_keys:
            actual_key = clean_example_keys[field_clean]
            value = example.get(actual_key)
            if value is not None:
                val_str = str(value).strip()
                if val_str and val_str.lower() != "none":
                    extracted_parts.append(val_str)     
    if extracted_parts:
        return "\n\n".join(extracted_parts)
    return default


def cleaning_process(ds, dataset_name: str, field_mapping: dict = None) -> Dataset:
    """
    Process dataset by cleaning text and filtering invalid examples.
    
    Args:
        ds: Input dataset
        dataset_name: Name of the dataset
        field_mapping: Optional field mapping for different datasets
        
    Returns:
        Processed dataset
    """
    if field_mapping is None:
        field_mapping = get_field_mapping(dataset_name=dataset_name)
    DATA_BASE = DATA_DIR / dataset_name
    PROCESSED_DATA = DATA_BASE
    PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
    if PROCESSED_DATA.exists():
        logger.info(f"🔄 Found existing processed dataset at {PROCESSED_DATA}")
        try:
            existing_ds = load_from_disk(PROCESSED_DATA)
            logger.info(f"✅ Loaded existing dataset with {len(existing_ds)} examples")
            logger.info("="*60)
            logger.info("ℹ️ Next step: Tokenization")
            logger.info(f" Run: python -m src.step2_tokenization {dataset_name}")
            logger.info("="*60)
            return existing_ds
        except Exception:
            logger.warning(f"⚠️ Existing dataset not valid yet, reprocessing...")
            
    logger.info("🔄 Processing dataset...")
    if len(ds) > 0:
        logger.info(f"📋 First example keys: {list(ds[0].keys())}")
        logger.info(f"📋 Field mapping: {field_mapping}")
    data_entries = []
    chunk_size = get_pipeline_value("steps.step1_cleaning.chunk_size", 5000)
    chunk_count = 0
    seen_hashes = set()
    check_duplicates = get_pipeline_value("steps.step1_cleaning.check_duplicates", True)
    dbg_kept = 0
    dbg_too_short = 0
    dbg_duplicates = 0
    dbg_parse_errors = 0
    DBG_PREVIEW_LIMIT = 3
    dbg_preview_samples = 0
    for i, example in enumerate(ds):
        try:
            raw_text = extract_field(example, field_mapping, "text", "")
            text = clean_html(raw_text)
            text = remove_urls_emails(text)
            text = fix_encoding_errors(text)
            text = normalize_text(text)
            if not is_valid_text(text):
                dbg_too_short += 1
                continue
            if check_duplicates:
                h = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
                if h in seen_hashes:
                    dbg_duplicates += 1
                    continue
                seen_hashes.add(h)
            data_entries.append({
                "text": text,
                "meta_category": str(extract_field(example, field_mapping, "meta_category", "")),
                "meta_date": str(extract_field(example, field_mapping, "meta_date", "")),
                "id": str(extract_field(example, field_mapping, "id", i))
            })
            dbg_kept += 1
            if dbg_preview_samples < DBG_PREVIEW_LIMIT:
                logger.info(f"Preview[{dbg_preview_samples}] len={len(text)} text={text[:200]!r}")
                dbg_preview_samples += 1
                
        except Exception as e:
            dbg_parse_errors += 1
            logger.warning(f"Parse error for example {i}: {e}")
            logger.debug(f"Example data: {example}")
            continue
        
        if (i + 1) % 10000 == 0:
            logger.info(
                f"Progress {i+1:,}/{len(ds)}: kept={dbg_kept:,}, too_short={dbg_too_short:,}, "
                f"dups={dbg_duplicates:,}, parse_err={dbg_parse_errors:,}"
            )
        if len(data_entries) >= chunk_size:
            chunk_path = PROCESSED_DATA.parent / f"{PROCESSED_DATA.name}_chunk_{chunk_count}"
            Dataset.from_list(data_entries).save_to_disk(chunk_path)
            logger.info(
                f"💾 Chunk saved: {chunk_path} | entries={len(data_entries)} | "
                f"totals: kept={dbg_kept:,}, too_short={dbg_too_short:,}, dups={dbg_duplicates:,}"
            )
            chunk_count += 1
            data_entries = []
            dbg_preview_samples = 0
    
    if data_entries:
        chunk_path = PROCESSED_DATA.parent / f"{PROCESSED_DATA.name}_chunk_{chunk_count}"
        Dataset.from_list(data_entries).save_to_disk(chunk_path)
        logger.info(
            f"💾 Final chunk saved: {chunk_path} | entries={len(data_entries)} | "
            f"totals: kept={dbg_kept:,}, too_short={dbg_too_short:,}, dups={dbg_duplicates:,}"
        )
    chunks = sorted(PROCESSED_DATA.parent.glob(f"{PROCESSED_DATA.name}_chunk_*"))
    logger.info(f"🔗 Found {len(chunks)} chunk directories")
    if not chunks:
        logger.error(f"❌ No valid entries were created after processing!")
        logger.error(f"📊 Statistics:")
        logger.error(f" - Input examples: {len(ds):,}")
        logger.error(f" - Kept entries: {dbg_kept:,}")
        logger.error(f" - Too short: {dbg_too_short:,}")
        logger.error(f" - Duplicates: {dbg_duplicates:,}")
        logger.error(f" - Parse errors: {dbg_parse_errors:,}")
        logger.error(f" - Field mapping: {field_mapping}")
        raise RuntimeError("No valid entries were created after processing!")
    try:
        dataset = load_from_disk(chunks[0])
        if len(chunks) > 1:
            for chunk in chunks[1:]:
                next_ds = load_from_disk(chunk)
                dataset = concatenate_datasets([dataset, next_ds])
        dataset.save_to_disk(PROCESSED_DATA)
        try:
            processed_display = PROCESSED_DATA.relative_to(PROJECT_ROOT)
        except ValueError:
            processed_display = PROCESSED_DATA
        logger.info(f"✅ All {len(dataset)} entries successfully saved to {processed_display}")
        for chunk_dir in chunks:
            shutil.rmtree(chunk_dir)
            logger.debug(f"🗑️ Deleted temporary chunk: {chunk_dir}")
            
    except Exception as e:
        logger.error(f"❌ Error processing chunks: {e}")
        raise
    logger.info("="*60)
    logger.info(f"📁 Output directory: {PROCESSED_DATA}")
    logger.info("📊 Statistics:")
    logger.info(f" - Input examples: {len(ds):,}")
    logger.info(f" - Kept entries: {dbg_kept:,}")
    logger.info(f" - Too short: {dbg_too_short:,}")
    logger.info(f" - Duplicates: {dbg_duplicates:,}")
    logger.info(f" - Errors: {dbg_parse_errors:,}")
    return dataset


def get_dataset_name_from_config(hf_config: str, hf_dataset: str = None) -> str:
    """
    Generate dataset name based on HF config.

    Args:
        hf_config: Hugging Face dataset config name or dataset name
        hf_dataset: Hugging Face dataset name (for disambiguation)

    Returns:
        Generated dataset name
    """
    try:
        config = _load_usecase_config("dataset_config.yaml", DATASET_CONFIG_FILE)
        hf_config_data = config.get("huggingface", {})
        config_mappings = hf_config_data.get("config_mappings", {})
        if hf_config in config_mappings:
            return config_mappings[hf_config].get("output_name", "unknown")
        datasets = hf_config_data.get("datasets", [])
        for ds in datasets:
            ds_name = ds.get("dataset_name")
            ds_config = ds.get("config")
            ds_output = ds.get("output_name")
            if hf_config == ds_name or hf_config == ds_config:
                if ds_output:
                    return ds_output
            if hf_dataset and hf_dataset == ds_name:
                if ds_output:
                    return ds_output
    except (FileNotFoundError, ImportError, KeyError):
        logger.warning("Could not load dataset config")
    return "unknown"


# --- MAIN ---
def main(hf_config: str = None, hf_dataset: str = None) -> None:
    """
    Main function to clean and prepare HF-dataset for training.

    Args:
        hf_config: Hugging Face dataset config name or output_name (e.g., "dump-20260520-1k" or "cases2026_1k")
                   If None, loads default from config
        hf_dataset: Hugging Face dataset name (e.g., "openlegaldata/court-decisions-germany")
                    If None, loads from config based on hf_config
    
    Returns:
        None
    """
    try:
        config = _load_usecase_config("dataset_config.yaml", DATASET_CONFIG_FILE)
        hf_config_data = config.get("huggingface", {})
        ds_config = get_dataset_config(hf_config, hf_dataset)
        if ds_config:
            hf_dataset_name = ds_config.get("dataset_name")
            hf_split = ds_config.get("split", "train")
            hf_config_value = ds_config.get("config")
            dataset_name = ds_config.get("output_name", hf_config)
        else:
            hf_dataset_name = hf_dataset or hf_config_data.get("dataset_name", "openlegaldata/court-decisions-germany")
            hf_split = hf_config_data.get("split", "train")
            hf_config_value = hf_config
            dataset_name = get_dataset_name_from_config(hf_config, hf_dataset)
            if dataset_name == "unknown" and hf_dataset:
                dataset_name = hf_dataset.replace("/", "_").replace("-", "_")
    except (FileNotFoundError, ImportError, KeyError):
        hf_dataset_name = hf_dataset or "openlegaldata/court-decisions-germany"
        hf_split = "train"
        hf_config_value = hf_config
        dataset_name = get_dataset_name_from_config(hf_config, hf_dataset)
        if dataset_name == "unknown" and hf_dataset:
            dataset_name = hf_dataset.replace("/", "_").replace("-", "_")
    
    field_mapping = get_field_mapping(hf_config or dataset_name, hf_dataset)
    log_blank_line(logger)
    logger.info("="*60)
    logger.info("💾 STEP 1: Dataset Cleaning (HF-Dataset creation included)")
    logger.info("="*60)
    logger.info(f"Loading Hugging Face dataset: {hf_dataset_name}")
    hf_token = os.getenv("HF_TOKEN")
    try:
        if hf_config_value:
            ds = load_dataset(hf_dataset_name, hf_config_value, split=hf_split, token=hf_token)
            logger.info(f"✅ Loaded {len(ds)} examples from config '{hf_config_value}'")
        else:
            ds = load_dataset(hf_dataset_name, split=hf_split, token=hf_token)
            logger.info(f"✅ Loaded {len(ds)} examples from default split '{hf_split}'")
    except Exception as e:
        logger.error(f"❌ Failed to load HF dataset: {e}")
        logger.error("💡 Make sure to: huggingface-cli login  (if dataset is gated)")
        raise
    cleaning_process(ds, dataset_name, field_mapping)
    logger.info("="*60)
    logger.info("ℹ️ Next step: Tokenization")
    logger.info(f" Run: python -m src.step2_tokenization {dataset_name}")
    logger.info("="*60)


# --- CLI ---
def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments (use default config values if not provided) for fetching & cleaning datasets via HuggingFace.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Download & clean Hugging Face datasets")
    parser.add_argument("--hf-dataset", required=False, default=None, 
        help="HF dataset name (e.g., 'openlegaldata/court-decisions-germany')")
    parser.add_argument("--hf-config", required=False, default=None,
        help="HF dataset config or output_name (e.g., 'dump-20260520-10k' or 'cases2026_10k')")
    args = parser.parse_args()
    if args.hf_config is None and args.hf_dataset is None:
        args.hf_config = get_pipeline_value("steps.step1_cleaning.hf_config", None)
        args.hf_dataset = get_pipeline_value("steps.step1_cleaning.hf_dataset", None)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(hf_config=args.hf_config, hf_dataset=args.hf_dataset)