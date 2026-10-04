"""
Quantitative Analysis via PPL Score
"""

import os, sys
os.environ['FORCE_TORCHAUDIO_AVAILABLE'] = '0'
sys.modules['torchaudio'] = None
# ----------------------------------------------------------------------
import os, gc, json, math, time, argparse, torch
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from transformers import AutoModelForCausalLM, AutoConfig, default_data_collator


# --- LOGGING ---
from src.utils import get_logger, log_blank_line
logger = get_logger('5b_evaluation')
logger.setLevel('DEBUG')


# --- CONFIG ---
from src.utils import project_rel, setup_device, get_pipeline_value, resolve_base_model, resolve_checkpoint_path, _resolve_named_path, get_model_config, cleanup_qwen_config, validate_checkpoint_model_type, extract_model_identifier, PROJECT_ROOT, DATA_DIR, OUTPUT_DIR
from src.step2_tokenization import initialize_tokenizer
RESULTS_DIR = OUTPUT_DIR / "test_results"

@dataclass
class EvalConfig:
    """Configuration class for evaluation parameters."""
    # Paths
    base_model: Optional[str] = None
    tokenized_data_path: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    output_dir: str = str(OUTPUT_DIR)
    results_dir: str = str(RESULTS_DIR)
    # Data
    split: str = "test"
    max_examples: Optional[int] = None
    # Model params
    lmbda: float = 0.3      # use same as in step4
    knn_temp: float = 1.0   # MemDec's default
    batch_size: int = 2
    seed: int = 42
    # Evaluation settings
    save_results: bool = True
    compare_with_base: bool = True


# --- HELPERS ---
def load_eval_dataset(config: EvalConfig) -> dict:
    """
    Load evaluation dataset from JSON files and dynamically mask prompts and padding.

    Args:
        config: Evaluation configuration containing dataset path, split, and base model

    Returns:
        HFDataset: Loaded and processed dataset with dynamic prompt and padding masking

    Raises:
        FileNotFoundError: If the tokenized data path or the split JSON file does not exist
    """
    path = Path(config.tokenized_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenized dataset not found: {path}")
    logger.info(f"📂 Loading dataset from: {path}  (split: {config.split})")
    logger.info(f"📄 Loading from JSON files")
    json_file = path / f"{config.split}.json"
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tokenizer = initialize_tokenizer(config.base_model)
    separators = [
        "ANTWORT (mit Gesetz + Begründung):\n",
        "ZUSAMMENFASSUNG:\n",
        "WIDERRUFSSCHREIBEN:\n",
        "Klage-Vorlage:\n",
        "RECHTSBEHELFSBELEHRUNG:\n",
        "SCHREIBEN:\n"
    ]
    final_input_ids = []
    final_attention_masks = []
    final_labels = []
    for item in data:
        input_ids = item["input_ids"]
        attention_mask = item["attention_mask"]
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        found_separator = None
        for sep in separators:
            if sep in decoded_text:
                found_separator = sep
                break
        if found_separator:
            parts = decoded_text.split(found_separator)
            prompt_text = parts[0] + found_separator
            first_real = attention_mask.index(1) if 1 in attention_mask else 0
            prompt_token_len = first_real + len(tokenizer.encode(prompt_text, add_special_tokens=True))
            label = [-100] * prompt_token_len + input_ids[prompt_token_len:]
        else:
            label = input_ids.copy()
        label = [tok if tok != tokenizer.pad_token_id else -100 for tok in label]
        final_input_ids.append(input_ids)
        final_attention_masks.append(attention_mask)
        final_labels.append(label)
    dataset = HFDataset.from_dict({
        "input_ids": final_input_ids,
        "attention_mask": final_attention_masks,
        "labels": final_labels
    })
    logger.info(f"✅ Loaded {len(dataset):,} examples, dynamically masked prompts for accurate PPL.")
    if config.max_examples is not None and len(dataset) > config.max_examples:
        logger.info(f"📊 Limiting evaluation to first {config.max_examples:,} examples")
        dataset = dataset.select(range(config.max_examples))
        logger.info(f"✅ Using {len(dataset):,} examples for evaluation")
    return dataset


def joint_evaluate(lm_logits, knn_logits, batch, lmbda):
    """
    Joint NLL evaluation computed only at label positions

    Args:
        lm_logits: Base model logits (B, T, V)
        knn_logits: KNN model logits (B, T, V)
        batch: Batch dict whose 'labels' use -100 for masked positions
        lmbda: Interpolation weight λ

    Returns:
        Tuple of (joint_nll_sum, lm_nll_sum, token_count). NLL sums are
        float32 scalars on the logits device. Batches whose shifted labels
        contain zero unmasked tokens return (0.0, 0.0, 0).
    """
    shift_lm  = lm_logits[:, :-1]
    shift_knn = knn_logits[:, :-1]
    labels    = batch['labels'][:, 1:]
    mask      = labels != -100
    total_tokens = int(mask.sum())
    if total_tokens == 0:
        zero = torch.zeros((), device=lm_logits.device, dtype=torch.float32)
        return zero, zero, 0
    safe_labels = labels.clamp(min=0).unsqueeze(-1)
    lm_logp  = (shift_lm.gather(-1,  safe_labels).squeeze(-1).float()
                - shift_lm.logsumexp(-1).float())
    knn_logp = (shift_knn.gather(-1, safe_labels).squeeze(-1).float()
                - shift_knn.logsumexp(-1).float())
    joint_logp = torch.logaddexp(
        lm_logp  + math.log(1 - lmbda),
        knn_logp + math.log(lmbda),
    )
    lm_nll    = -lm_logp[mask].sum()
    joint_nll = -joint_logp[mask].sum()
    return joint_nll, lm_nll, total_tokens


def load_models(config: EvalConfig, tokenizer) -> tuple:
    """
    Load and initialize base and KNN models for evaluation.
    
    Args:
        config: Evaluation configuration with model parameters
        tokenizer: Tokenizer instance for model compatibility
        
    Returns:
        Tuple containing (base_lm, knn_generator)
        
    Raises:
        Exception: If model loading fails critically
    """
    device      = setup_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    device_map  = "auto"         if device.type == "cuda" else None
    logger.info(f"Loading base model: {config.base_model}")
    base_model_config = AutoConfig.from_pretrained(config.base_model)
    cleanup_qwen_config(base_model_config, config.base_model)
    
    base_lm = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        config=base_model_config,
    )
    knn_path = config.checkpoint_dir or config.base_model
    logger.info(f"Loading knn_generator from: {knn_path}")
    if config.checkpoint_dir:
        validate_checkpoint_model_type(knn_path, config.base_model, base_model_config)
    try:
        knn_generator = AutoModelForCausalLM.from_pretrained(
            knn_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        src = "trained checkpoint" if config.checkpoint_dir else "base model (no checkpoint)"
        logger.info(f"✅ knn_generator loaded from {src}")
    except Exception as e:
        if config.checkpoint_dir:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            raise
        logger.warning("⚠️ Falling back to base model as knn_generator.")
        knn_generator = base_lm

    vocab_size = len(tokenizer)
    base_lm.resize_token_embeddings(vocab_size)
    knn_generator.resize_token_embeddings(vocab_size)
    base_lm.eval()
    knn_generator.eval()
    if device.type != "cuda":
        base_lm       = base_lm.to(device)
        knn_generator = knn_generator.to(device)
    logger.info(f"✅ Models ready  λ={config.lmbda}  knn_temp={config.knn_temp}")
    return base_lm, knn_generator


# --- MAIN ---
def main(config: EvalConfig) -> dict:
    """
    Main evaluation function.
    
    Args:
        config: Complete evaluation configuration
    
    Returns:
        dict: Evaluation results
    """
    log_blank_line(logger)
    logger.info("=" * 60)
    logger.info("📊 STEP 5: Quantitative PPL Evaluation")
    logger.info("=" * 60)
    for k, v in config.__dict__.items():
        logger.info(f"  {k}: {v}")
    dataset    = load_eval_dataset(config)
    dataloader = DataLoader(
        dataset,
        collate_fn  = default_data_collator,
        batch_size  = config.batch_size,
        shuffle     = False,
        num_workers = 0,
    )
    logger.info(f"📦 DataLoader: {len(dataloader)} batches × batch_size={config.batch_size}")
    tokenizer             = initialize_tokenizer(config.base_model)
    base_lm, knn_generator = load_models(config, tokenizer)
    device = next(base_lm.parameters()).device
    total_joint_nll = 0.0
    total_base_nll  = 0.0
    total_tokens    = 0
    logger.info("🔄 Starting evaluation loop...")
    t_start = time.time()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        batch_device   = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        with torch.inference_mode():
            lm_out  = base_lm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            knn_out = knn_generator(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            joint_nll, lm_nll, cnt = joint_evaluate(
                lm_out.logits, knn_out.logits, batch_device, config.lmbda
            )
        total_joint_nll += joint_nll.item()
        total_base_nll  += lm_nll.item()
        total_tokens    += cnt
        if (batch_idx + 1) % 50 == 0:
            logger.debug(
                f"  [{batch_idx+1}/{len(dataloader)}] "
                f"running base_ppl={math.exp(total_base_nll / total_tokens):.2f}  "
                f"joint_ppl={math.exp(total_joint_nll / total_tokens):.2f}  "
                f"tokens={total_tokens:,}"
            )

    elapsed = time.time() - t_start
    if total_tokens == 0:
        logger.error("❌ No tokens were evaluated. Check that 'labels' column is set correctly.")
        return {}
    base_ppl  = math.exp(total_base_nll  / total_tokens)
    joint_ppl = math.exp(total_joint_nll / total_tokens)
    ppl_delta = base_ppl - joint_ppl
    ppl_improvement_pct = (ppl_delta / base_ppl) * 100 if base_ppl > 0 else 0.0
    logger.info("=" * 60)
    logger.info(f"📊 Tokens evaluated : {total_tokens:,}")
    logger.info(f"⏱️ Elapsed time     : {elapsed:.1f}s")
    logger.info(f"🔧 λ (lmbda)        : {config.lmbda}")
    logger.info("  " + "-" * 40)
    logger.info(f"📈 Base LM PPL      : {base_ppl:.4f}")
    logger.info(f"📈 MemDec PPL       : {joint_ppl:.4f}")
    logger.info(f"📉 Δ PPL            : {ppl_delta:+.4f}  ({ppl_improvement_pct:+.2f}%)")
    if ppl_delta > 0:
        logger.info(f"  ✅ MemDec IMPROVES perplexity by {ppl_improvement_pct:+.2f}%")
    elif ppl_delta < 0:
        logger.warning(f"  ⚠️ MemDec is WORSE by {abs(ppl_improvement_pct):.2f}% (check λ or training)")
    else:
        logger.info("  ℹ️  No difference (knn_generator = base model?)")
    logger.info("=" * 60)
    model_keyword = extract_model_identifier(config.base_model) or "model"
    results = {
        "base_ppl":            round(base_ppl,  6),
        "joint_ppl":           round(joint_ppl, 6),
        "ppl_delta":           round(ppl_delta, 6),
        "ppl_improvement_pct": round(ppl_improvement_pct, 4),
        "n_tokens":            total_tokens,
        "elapsed_seconds":     round(elapsed, 2),
        "lmbda":               config.lmbda,
        "knn_temp":            config.knn_temp,
        "split":               config.split,
        "tokenized_data_path": project_rel(config.tokenized_data_path),
        "checkpoint_dir":      project_rel(config.checkpoint_dir),
        "base_model":          config.base_model,
        "model_keyword":       model_keyword,
    }
    if config.save_results:
        os.makedirs(config.results_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path  = Path(config.results_dir) / f"ppl_results_{model_keyword}_{timestamp}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        try:
            display_path = out_path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = out_path
        logger.info(f"✅ Results saved to {display_path}")
    return results


# --- CLI ---
def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments (use default config values if not provided) for evaluation.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate Memory Decoder with quantitative metrics")
    default_config = EvalConfig()
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--model", type=str, default=None)
    base_parser.add_argument("--checkpoint", type=str, default=None)
    base_parser.add_argument("tokenized_data", nargs="?", type=str, default=None)
    early_args = base_parser.parse_known_args()[0]
    # Determine tokenized_data: CLI > pipeline config
    if early_args.tokenized_data:
        config_tokenized_data = early_args.tokenized_data
    else:
        config_tokenized_data = get_pipeline_value("steps.step5_evaluation.tokenized_data", None)
        if not config_tokenized_data:
            config_tokenized_data = default_config.tokenized_data_path
    config_checkpoint = get_pipeline_value("steps.step5_evaluation.checkpoint", default_config.checkpoint_dir)
    # Determine base_model: CLI > step config > tokenized data > checkpoint (CLI or pipeline config) > pipeline default > env
    config_base_model = resolve_base_model(
        cli_model=early_args.model,
        step_config_key="steps.step5_evaluation",
        tokenized_data=config_tokenized_data,
        checkpoint=early_args.checkpoint or config_checkpoint,
        output_dir=str(default_config.output_dir),
    )
    config_split = get_pipeline_value("steps.step5_evaluation.split", default_config.split)
    config_max_examples = get_pipeline_value("steps.step5_evaluation.max_examples", default_config.max_examples)
    config_lmbda = get_pipeline_value("steps.step5_evaluation.lmbda", default_config.lmbda)
    config_knn_temp = get_pipeline_value("steps.step5_evaluation.knn_temp", default_config.knn_temp)
    config_batch_size = get_pipeline_value("steps.step5_evaluation.batch_size", default_config.batch_size)
    config_seed = get_pipeline_value("steps.step5_evaluation.seed", default_config.seed)
    config_save_results = get_pipeline_value("steps.step5_evaluation.save_results", default_config.save_results)
    config_compare_base = get_pipeline_value("steps.step5_evaluation.compare_base", default_config.compare_with_base)
    action_save = "store_false" if config_save_results else "store_true"
    action_compare = "store_false" if config_compare_base else "store_true"
    # Variable defaults via pipeline_config
    parser.add_argument("--model", type=str, default=config_base_model,
                        help=f"Base model preset (gemma3/qwen3.5/smollm3) [default: {config_base_model}]")
    parser.add_argument("tokenized_data", nargs="?", type=str, default=config_tokenized_data,
                        help=f"Path to tokenized dataset directory [default: {config_tokenized_data}]")
    parser.add_argument("--checkpoint", type=str, default=config_checkpoint,
                        help="Trained checkpoint to load as knn_generator. Accepts: <step_5000>, <outputs/step_5000>, absolute path, or <latest>")
    parser.add_argument("--split", type=str, default=config_split, choices=["train", "validation", "test"],
                        help=f"Dataset split to evaluate [default: {config_split}]")
    parser.add_argument("--max-examples", type=int, default=config_max_examples,
                        help=f"Limit evaluation to first N examples (useful for testing with limited memory) [default: {config_max_examples}]")
    parser.add_argument("--lmbda", type=float, default=config_lmbda,
                        help=f"Interpolation weight λ [default: {config_lmbda}]")
    parser.add_argument("--knn-temp", type=float, default=config_knn_temp,
                        help=f"knn_generator logit temperature [default: {config_knn_temp}]")
    parser.add_argument("--batch-size", type=int, default=config_batch_size,
                        help=f"Batch size [default: {config_batch_size}]")
    parser.add_argument("--seed", type=int, default=config_seed,
                        help=f"Random seed [default: {config_seed}]")
    parser.add_argument("--save-results", action=action_save,
                        help=f"Skip writing result JSON files [default: {config_save_results}]")
    parser.add_argument("--compare-base", action=action_compare,
                        help=f"Skip base model comparison [default: {config_compare_base}]")
    # Fixed defaults via PretrainConfig (cannot be changed via CLI)
    parser.add_argument("--output-dir", type=str, default=default_config.output_dir,
                        help=f"[FIXED] Output directory for checkpoints [default: {default_config.output_dir}]")
    parser.add_argument("--results-dir", type=str, default=default_config.results_dir,
                        help=f"[FIXED] Where to write result JSON files [default: {default_config.results_dir}]")
    args = parser.parse_args()
    if args.output_dir != default_config.output_dir:
        parser.error(f"--output-dir is fixed to '{default_config.output_dir}' and cannot be changed via CLI.")
    if args.results_dir != default_config.results_dir:
        parser.error(f"--results-dir is fixed to '{default_config.results_dir}' and cannot be changed via CLI.")
    return args


if __name__ == "__main__":
    args = parse_arguments()
    model_keyword = extract_model_identifier(args.model)
    model_info = get_model_config(model_keyword)
    resolved_checkpoint = resolve_checkpoint_path(args.checkpoint, args.output_dir)
    if args.checkpoint and not resolved_checkpoint:
        logger.error(
            f"❌ --checkpoint '{args.checkpoint}' could not be resolved. "
            "Check that the directory exists inside outputs/ "
            "(or pass an absolute path or 'latest')."
        )
        raise SystemExit(1)
    resolved_tokenized = _resolve_named_path(args.tokenized_data, DATA_DIR, "tokenized-data")
    if not resolved_tokenized:
        logger.error(
            f"❌ tokenized_data '{args.tokenized_data}' could not be resolved. "
            "Check that the folder exists inside dataset/."
        )
        raise SystemExit(1)
    config = EvalConfig(
        base_model          = model_info['name'],
        tokenized_data_path = resolved_tokenized,
        checkpoint_dir      = resolved_checkpoint,
        split               = args.split,
        max_examples        = args.max_examples,
        lmbda               = args.lmbda,
        knn_temp            = args.knn_temp,
        batch_size          = args.batch_size,
        seed                = args.seed,
        save_results        = args.save_results,
        compare_with_base   = args.compare_base,
    )
    try:
        main(config)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()