"""
Dataset Manager (Corpus Dataset)
"""

import os, argparse, json, shutil, tempfile
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_from_disk, concatenate_datasets


# --- LOGGING ---
from src.utils import get_logger, log_blank_line, get_pipeline_value
logger = get_logger('0_data_management')


# --- CONFIG ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CORPUS_NAME = get_pipeline_value("pipeline.corpus_name")


# --- HELPERS ---
def load_index(index_file: str) -> Dict[str, Any]:
    """
    Load the dataset index file.
    
    Args:
        index_file: Path to the index file
        
    Returns:
        dict: Loaded index data or default empty index
    """
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        logger.warning(f"Could not load index file {index_file}. Creating a new one.")
        return {"datasets": []}


def save_index(index: Dict[str, Any], index_file: str) -> None:
    """
    Save the dataset index to a file.
    
    Args:
        index: Index data to save
        index_file: Path where to save the index
    
    Returns:
        None
    """
    os.makedirs(os.path.dirname(index_file), exist_ok=True)
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    logger.debug(f"Index file saved to {index_file}")


def process_datasets(dataset_paths: List[str], output_path: str, index_file: str) -> Any:
    """
    Process and combine multiple datasets into a single dataset.
    
    Args:
        dataset_paths: List of dataset paths to process
        output_path: Path to save combined dataset
        index_file: Path to index file for tracking processed datasets
        
    Returns:
        The combined dataset
        
    Raises:
        FileNotFoundError: If no valid datasets could be loaded
        RuntimeError: If the existing combined dataset cannot be loaded but the
            index already references previously merged datasets
    """
    current_index = load_index(index_file)
    indexed_sources = current_index.get("datasets", [])
    new_datasets = []
    new_sources = []
    total_samples = 0
    output_path_resolved = str(Path(output_path).resolve())
    for dataset_path in dataset_paths:
        if dataset_path in indexed_sources:
            logger.info(f"✅ Dataset already in index: {dataset_path}")
            continue
        abs_path = PROJECT_ROOT / dataset_path
        if str(abs_path.resolve()) == output_path_resolved:
            logger.info(f"ℹ️ Skipping {dataset_path}: this is the output corpus itself, not a source dataset")
            continue
        if not abs_path.exists():
            logger.warning(f"  ❌ Dataset not found: {dataset_path}")
            continue   
        try:
            ds = load_from_disk(str(abs_path))
            new_datasets.append(ds)
            new_sources.append(dataset_path)
            total_samples += len(ds)
            logger.info(f"  ✅ Found new dataset: {dataset_path} ({len(ds):,} samples)")
        except Exception as e:
            logger.warning(f"  ❌ Could not load dataset {dataset_path}: {e}")
            continue
    if not new_datasets:
        if os.path.exists(output_path):
            try:
                combined = load_from_disk(output_path)
                logger.info(f"✅ No new datasets. Loaded existing combined dataset with {len(combined):,} samples")
                return combined
            except Exception as e:
                logger.warning(f"Failed to load existing dataset: {e}. Rebuilding...")
        else:
            logger.info("ℹ️ No datasets found and no existing combined dataset.")
            return None
    
    combined = None
    existing_dataset_loaded = False
    if os.path.exists(output_path):
        try:
            combined = load_from_disk(output_path)
            existing_dataset_loaded = True
            logger.info(f"🔄 Loaded existing combined dataset with {len(combined):,} samples")
        except Exception as e:
            if indexed_sources:
                logger.error(
                    f"❌ Existing combined dataset at {output_path} could not be loaded, but the index "
                    f"already lists {len(indexed_sources)} previously merged dataset(s): {e}"
                )
                raise RuntimeError(
                    f"Cannot load existing combined dataset at '{output_path}', but the index file "
                    f"'{index_file}' already references {len(indexed_sources)} previously merged "
                    f"dataset(s). Refusing to continue to avoid silently discarding that data. Please "
                    f"repair or delete '{output_path}' and its index file manually before re-running."
                ) from e
            logger.warning(f"Failed to load existing dataset: {e}. Creating new...")
    successfully_added = []
    for i, new_ds in enumerate(new_datasets):
        if combined is None:
            combined = new_ds
            successfully_added.append(new_sources[i])
        else:
            try:
                combined = concatenate_datasets([combined, new_ds])
                logger.info(f"  ✅ Added {len(new_ds):,} samples from {new_sources[i]}")
                successfully_added.append(new_sources[i])
            except Exception as e:
                logger.warning(f"  ❌ Could not combine dataset {new_sources[i]}: {e}")
                continue
    if combined is None:
        raise FileNotFoundError("No valid datasets could be loaded")
    if successfully_added:
        temp_dir = tempfile.mkdtemp(prefix="memdec_corpus_")
        temp_path = Path(temp_dir) / "combined_dataset"
        combined.save_to_disk(temp_path)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        shutil.move(str(temp_path), output_path)
        shutil.rmtree(temp_dir)
        updated_sources = indexed_sources + successfully_added
        save_index({"datasets": updated_sources}, index_file)
        logger.info("=" * 60)
        logger.info(f"✅ Successfully updated combined dataset with {len(combined):,} samples")
        logger.info(f"   - Added {len(successfully_added)} new datasets")
        logger.info(f"   - Total sources: {len(updated_sources)} datasets")
        try:
            output_display = Path(output_path).relative_to(PROJECT_ROOT)
        except ValueError:
            output_display = Path(output_path)
        logger.info(f"   - Saved to: {output_display}")
    else:
        if existing_dataset_loaded:
            logger.info("=" * 60)
            logger.info(f"ℹ️ No new datasets could be added. Combined dataset remains unchanged with {len(combined):,} samples")
        else:
            logger.info("=" * 60)
            logger.info(f"ℹ️ Using existing dataset with {len(combined):,} samples")
    return combined


# --- MAIN ---
def main(datasets: List[str], output_dir: str = None, corpus_name: str = None) -> None:
    """
    Main function to process and combine multiple datasets.
    
    Args:
        datasets: List of dataset names or paths to process
        output_dir: Directory to save the combined dataset (if None, loads from config)
        corpus_name: Name of the corpus used for the output dir default and the
            index file (if None, falls back to pipeline.corpus_name from config;
            required if neither is set)
    
    Returns:
        None
    
    Raises:
        SystemExit: If an error occurs during processing or no corpus name is configured
    """
    corpus_name = corpus_name or CORPUS_NAME
    if corpus_name is None:
        logger.error("❌ No corpus name configured: set pipeline.corpus_name in pipeline_config.yaml or pass --corpus-name")
        raise SystemExit(1)
    if output_dir is None:
        output_dir = get_pipeline_value("steps.step0_data_management.default_output_dir", f"dataset/{corpus_name}")
    log_blank_line(logger)
    logger.info("=" * 60)
    logger.info("🗃️ STEP 0: Dataset Corpus Creation")
    logger.info("=" * 60)
    try:
        output_dir = str(Path(output_dir).resolve())
        os.makedirs(output_dir, exist_ok=True)
        full_dataset_paths = []
        for ds in datasets:
            for base in [PROJECT_ROOT, DATA_DIR]:
                path = base / ds
                if path.exists():
                    try:
                        rel_path = str(path.resolve().relative_to(PROJECT_ROOT)).replace('\\', '/')
                    except ValueError:
                        rel_path = str(path.resolve())
                    full_dataset_paths.append(rel_path)
                    break
            else:
                path = Path(ds)
                if path.exists():
                    try:
                        rel_path = str(path.resolve().relative_to(PROJECT_ROOT)).replace('\\', '/')
                    except ValueError:
                        rel_path = str(path)
                    full_dataset_paths.append(rel_path)
                else:
                    logger.warning(f"Dataset not found: {ds}")
    
        if not full_dataset_paths:
            raise FileNotFoundError("No valid datasets found. Please check your dataset paths.")
        index_file = str(DATA_DIR / f"{corpus_name}_index.json")
        logger.info(f"Processing {len(full_dataset_paths)} datasets:")
        for ds in full_dataset_paths:
            logger.info(f"  • {ds}")
        combined = process_datasets(full_dataset_paths, output_dir, index_file)
        logger.info("=" * 60)
        logger.info(f"📊 Combined dataset: {len(combined):,} total samples")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info("=" * 60)
        logger.info("ℹ️  Next step: Tokenization")
        logger.info(f"   Run: python -m src.step2_tokenization {Path(output_dir).name}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise SystemExit(1) from e


# --- CLI ---
def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments (use default config values if not provided) for dataset handling (corpus creation).
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Combine multiple datasets into a single dataset for processing"
    )
    config_datasets = get_pipeline_value("steps.step0_data_management.datasets", [])
    parser.add_argument("--corpus-name", type=str, default=None,
                        help="Corpus name for the default output dir and <corpus>_index.json "
                             f"[default: pipeline.corpus_name = {CORPUS_NAME or 'unset — flag required'}]")
    parser.add_argument("datasets", nargs="*", default=config_datasets,
                        help=f"Names or paths of the datasets to combine [default: {config_datasets}]")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for the combined dataset [default: dataset/<corpus_name>]")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(datasets=args.datasets, output_dir=args.output_dir, corpus_name=args.corpus_name)