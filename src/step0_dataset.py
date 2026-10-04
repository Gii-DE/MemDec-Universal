"""
Dataset Preparation (from URL or local file) & Cleaning
"""

import argparse, os, csv, gzip, json, requests, shutil, tarfile
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple
from datasets import load_dataset, Dataset


# --- LOGGING ---
from src.utils import get_logger, log_blank_line, _load_usecase_config, DATASET_CONFIG_FILE, get_pipeline_value
logger = get_logger('0_dataset')


# --- CONFIG ---
from src.step1_cleaning import cleaning_process
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# --- DATASET OPTIONS ---
def get_dataset_urls() -> dict:
    """
    Load dataset URLs from configuration file.
    
    Returns:
        Dictionary mapping dataset names to their URLs or paths
    """
    try:
        config = _load_usecase_config("dataset_config.yaml", DATASET_CONFIG_FILE)
        urls = {}
        datasets = config.get("datasets", {}) or {}
        for key, value in datasets.items():
            if isinstance(value, dict):
                urls[key] = value.get("url", "")
            else:
                urls[key] = value
        local_file = config.get("local_files") or {}
        for key, value in local_file.items():
            urls[key] = value.get("path", "")
        return urls
    except (FileNotFoundError, ImportError):
        raise ImportError("Could not load dataset config")

DATASET_URLS = get_dataset_urls()


# --- HELPERS ---
def _resolve_dataset_url(choice: str) -> str:
    """
    Resolve the full URL or path for the selected dataset.
    
    Args:
        choice: Dataset selection (either a key from DATASET_URLS, direct URL, or local file path)
        
    Returns:
        str: Full URL or path of the selected dataset
        
    Raises:
        ValueError: If the selected dataset is not found in available options
    """
    if choice.startswith("http"):
        return choice
    if choice.endswith((".json", ".jsonl", ".tsv", ".csv")) or Path(choice).exists():
        return choice
    if choice not in DATASET_URLS:
        available = ', '.join(DATASET_URLS.keys())
        raise ValueError(f"Invalid selection: {choice}. Available: {available}")
    return DATASET_URLS[choice]


def _download_with_progress(url: str, target_path: Path, chunk_size: int = 8192) -> None:
    """
    Download a file from URL with progress bar and resume support.
    
    Args:
        url: Source URL of the file
        target_path: Destination path for the downloaded file
        chunk_size: Size of chunks to download at once
    
    Returns:
        None
    
    Raises:
        requests.HTTPError: If the download fails
    """
    temp_path = target_path.with_suffix('.part')
    headers = {}
    start_byte = 0
    if temp_path.exists():
        start_byte = os.path.getsize(temp_path)
        headers = {'Range': f'bytes={start_byte}-'}
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0)) + start_byte
    if total_size == start_byte and temp_path.exists():
        logger.warning("File already fully downloaded")
        return
    progress = tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        desc=f"Downloading {target_path.name}",
        initial=start_byte
    )
    mode = 'ab' if start_byte > 0 else 'wb'
    with open(temp_path, mode) as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                progress.update(len(chunk))
                f.write(chunk)
    progress.close()
    if start_byte == 0 or not target_path.exists():
        temp_path.rename(target_path)
    else:
        with open(target_path, 'ab') as f_out, open(temp_path, 'rb') as f_in:
            shutil.copyfileobj(f_in, f_out)
        temp_path.unlink()


def _extract_gz(gz_path: Path, target_path: Path) -> None:
    """
    Extract a .gz file to the target path.
    
    Args:
        gz_path: Path to the .gz file
        target_path: Destination path for the extracted file
    
    Returns:
        None
    """
    with gzip.open(gz_path, 'rb') as f_in, open(target_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


def _extract_tar(tar_path: Path, target_dir: Path) -> Path:
    """
    Extract a .tar.gz / .tgz archive into the target directory.

    Args:
        tar_path: Path to the tar archive
        target_dir: Destination directory for the extracted members

    Returns:
        Path to the first extracted file member
    """
    with tarfile.open(tar_path, 'r:*') as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        if not members:
            raise RuntimeError(f"Archive contains no files: {tar_path.name}")
        if not all((target_dir / m.name).exists() for m in members):
            try:
                tar.extractall(target_dir, filter="data")
            except TypeError:
                tar.extractall(target_dir)
    if len(members) > 1:
        logger.warning(f"{tar_path.name} contains {len(members)} files — using {members[0].name}")
    return target_dir / members[0].name


def download_and_extract(url: str, target_dir: Path, preferred_stem: Optional[str] = None) -> Tuple[Path, str]:
    """
    Download and extract a dataset to the target directory.
    If url is a local file path, returns it directly without downloading.
    
    Args:
        url: URL or local path of the dataset
        target_dir: Target directory for the downloaded dataset
        preferred_stem: Optional preferred filename stem (without extension)
        
    Returns:
        Tuple containing:
            - Path to the extracted file
            - Status message
    """
    if not url.startswith("http"):
        local_path = Path(url)
        if not local_path.exists():
            local_path = PROJECT_ROOT / url
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {url}")
        if not local_path.name.lower().endswith((".gz", ".tgz")):
            return local_path, f"Using local file: {local_path.name}"
        target_dir.mkdir(parents=True, exist_ok=True)
        if local_path.name.lower().endswith((".tar.gz", ".tgz")):
            try:
                extracted = _extract_tar(local_path, target_dir)
            except Exception as e:
                raise RuntimeError(f"Failed to extract {local_path}: {str(e)}")
            return extracted, f"Extracted local archive: {extracted.name}"
        inner_name = local_path.with_suffix('').name  # strip .gz
        output_filename = f"{preferred_stem}{Path(inner_name).suffix}" if preferred_stem else inner_name
        output_path = target_dir / output_filename
        if output_path.exists():
            return output_path, "Already exists"
        logger.info(f"Extracting {local_path.name} to {output_filename}...")
        try:
            _extract_gz(local_path, output_path)
        except Exception as e:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to extract {local_path}: {str(e)}")
        return output_path, f"Successfully extracted: {output_filename}"
    # Handle remote URLs
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = os.path.basename(url)
    download_name = f"{preferred_stem}{''.join(Path(filename).suffixes)}" if preferred_stem else filename
    download_path = target_dir / download_name
    output_filename = download_name[:-3] if download_name.lower().endswith(".gz") else download_name
    output_path = target_dir / output_filename
    if not download_path.exists() and not output_path.exists():
        _download_with_progress(url, download_path)
    if download_path != output_path and download_path.exists() and not output_path.exists():
        logger.info(f"Extracting {download_path.name} to {output_filename}...")
        try:
            _extract_gz(download_path, output_path)
            download_path.unlink(missing_ok=True)
            return output_path, f"Successfully extracted: {output_filename}"
        except Exception as e:
            if download_path.exists():
                download_path.unlink()
            raise RuntimeError(f"Failed to extract {download_path}: {str(e)}")
    if output_path.exists():
        return output_path, "Already exists"
    return download_path, "Downloaded"


# --- MAIN ---
def main(dataset_choice: str = None) -> None:
    """
    Main function to download and process datasets.
    
    Args:
        dataset_choice: Dataset selection: key in DATASET_URLS, a direct http(s) URL, or a local file path
    
    Returns:
        None
    
    Raises:
        SystemExit: If an error occurs during processing
    """
    log_blank_line(logger)
    logger.info("=" * 60)
    logger.info("💾 STEP 0: Dataset Download & Cleaning (step1_cleaning.py included)")
    logger.info("=" * 60)
    try:
        dataset_url = _resolve_dataset_url(dataset_choice)
        extracted_file, status = download_and_extract(
            dataset_url,
            DATA_DIR,
            preferred_stem=dataset_choice if dataset_choice in DATASET_URLS else None
        )
        logger.info(f"✅ {status}")
        logger.info(f"📁 Path: {extracted_file.absolute()}")
        logger.info("🧹 Cleaning downloaded dataset...")
        if extracted_file.suffix == ".json":
            raw_ds = load_dataset('json', data_files={'train': str(extracted_file)})['train']
        elif extracted_file.suffix == ".jsonl":
            data = []
            with open(extracted_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            raw_ds = Dataset.from_list(data)
        elif extracted_file.suffix in (".tsv", ".csv"):
            data = []
            csv.field_size_limit(2**31 - 1)
            delimiter = '\t' if extracted_file.suffix == ".tsv" else ','
            prefix = extracted_file.name.split('.')[0]
            ext = extracted_file.suffix
            tsv_files = sorted(extracted_file.parent.glob(f"{prefix}.*{ext}")) or [extracted_file]
            train_files = [f for f in tsv_files if f.stem.endswith(".train")]
            if train_files:
                tsv_files = train_files  # benchmark splits (dev/test exist) → use train only
            for tsv_file in tsv_files:
                with open(tsv_file, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.reader(f, delimiter=delimiter)
                    first = next(reader, [])
                    header = [h.strip().lower() for h in first]
                    has_header = any(h in ("id", "text", "title", "query") for h in header)
                    fields = header if has_header else ["id", "text"]
                    if not has_header and first:
                        data.append(dict(zip(fields, first)))
                    for row in reader:
                        if row:
                            data.append(dict(zip(fields, row)))
            if len(tsv_files) > 1:
                logger.info(f"Merged {len(tsv_files)} TSV splits: {', '.join(f.name for f in tsv_files)}")
            raw_ds = Dataset.from_list(data)
        else:
            raise ValueError(f"Unsupported file format: {extracted_file.suffix} (supported: .json, .jsonl, .tsv, .csv)")
        logger.info(f"📥 Loaded {len(raw_ds)} examples from {extracted_file}")
        cleaning_process(raw_ds, dataset_choice)
        logger.info("=" * 60)
        logger.info("ℹ️  Next step: Dataset Combination")
        logger.info(f"   Run: python -m src.step0_data_management {dataset_choice} <dataset2> <datasetN>")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        raise SystemExit(1) from e


# --- CLI ---
def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments (use default config values if not provided) for dataset download.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Download & clean JSON/JSONL datasets")
    config_dataset = get_pipeline_value("steps.step0_dataset.default_dataset", None)
    local_file = get_pipeline_value("steps.step0_dataset.local_file", None)
    if config_dataset is None and local_file:
        config_dataset = local_file
    available_datasets = list(DATASET_URLS.keys())
    if not available_datasets:
        parser.error("No datasets available in configuration. Please check dataset_config.yaml")
    if config_dataset not in available_datasets:
        config_dataset = available_datasets[0]
    parser.add_argument("dataset", nargs="?", default=config_dataset, choices=available_datasets,
                        help=f"Name of the dataset to download [default: {config_dataset}]")
    parser.add_argument("--local-file", dest="local_file_override", 
                        help="Override local_file from config")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    dataset_to_process = args.local_file_override if args.local_file_override else args.dataset
    main(dataset_to_process)