"""
Qualitative Analysis via Human Review & Source-Fidelity Check
"""

import os, sys, warnings, logging
os.environ['FORCE_TORCHAUDIO_AVAILABLE'] = '0'
sys.modules['torchaudio'] = None
warnings.filterwarnings("ignore", message=".*`torch_dtype` is deprecated.*")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
# ----------------------------------------------------------------------
import os, gc, argparse, torch, json, time, re, difflib
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoConfig, GenerationConfig

from MemoryDecoder.demo.memDec import MemoryDecoder


# --- LOGGING ---
from src.utils import get_logger, log_blank_line
logger = get_logger('5a_testing')
logger.setLevel('DEBUG')


# --- CONFIG ---
from src.utils import setup_device, get_pipeline_value, _load_usecase_config, _resolve_checkpoint_or_exit, resolve_base_model, get_model_config, cleanup_qwen_config, validate_checkpoint_model_type, extract_model_identifier, PROJECT_ROOT, DATA_DIR, KNOWLEDGE_BASE_DIR, OUTPUT_DIR
from src.step2_tokenization import initialize_tokenizer
RESULTS_DIR = OUTPUT_DIR / "test_results"

@dataclass
class TestingConfig:
    """Configuration dataclass for all testing parameters."""
    # Paths
    base_model: Optional[str] = None
    knowledge_base_path: str = str(KNOWLEDGE_BASE_DIR)
    checkpoint_dir: Optional[str] = None
    output_dir: str = str(OUTPUT_DIR)
    results_dir: str = str(RESULTS_DIR)
    # MemDec params
    lmbda: float = 0.3      # use same as in step4
    knn_temp: float = 1.0   # MemDec's default
    # Generation settings
    max_new_tokens: int = 300
    repetition_penalty: float = 1.25
    do_sample: bool = False     # original demo default
    # False: always pick the most likely token (greedy)
    # True:  sample tokens randomly; experimental — produced broken output in tests
    # --- Only when "do_sample=True" ---
    temperature: float = 1.0    # sampling sharpness
    top_p: float = 1.0          # nucleus threshold
    top_k: int = 0              # top-k limit; 0 = off
    # ----------------------------------
    batch_size: int = 2
    scenarios: Optional[List[str]] = None
    tasks: Optional[List[str]] = None
    save_results: bool = True
    compare_with_base: bool = True


# --- HELPERS ---
def load_test_config() -> Dict[str, Any]:
    """
    Load test cases from use-case specific config file.
    
    Returns:
        Dictionary with test cases in the format expected by the testing pipeline
    """
    try:
        raw_config = _load_usecase_config("test_cases.yaml")
    except FileNotFoundError:
        raw_config = {}
    if not raw_config:
        logger.warning("⚠️ No test cases config found. Using empty test set.")
        return {}
    test_cases = raw_config.get("test_cases", {})
    converted = {}
    for scenario_name, scenario_data in test_cases.items():
        if not isinstance(scenario_data, dict):
            continue
        converted[scenario_name] = {
            "title": scenario_data.get("title", scenario_name),
            "task": scenario_data.get("task", "mixed"),
            "prompts": []
        }
        for prompt_item in scenario_data.get("prompts", []):
            if isinstance(prompt_item, dict):
                converted[scenario_name]["prompts"].append(prompt_item.get("text", ""))
            elif isinstance(prompt_item, str):
                converted[scenario_name]["prompts"].append(prompt_item)
    logger.info(f"✅ Loaded {len(converted)} test scenarios from config")
    return converted


def _load_lm(path: str, config: TestingConfig, torch_dtype, device_map, model_config=None):
    """
    Load a causal LM via plain transformers.

    Args:
        path: HF model id or local checkpoint directory (full weights, not LoRA)
        config: TestingConfig (currently unused; kept for signature symmetry)
        torch_dtype: dtype for weights
        device_map: device_map for the model
        model_config: optional AutoConfig for the model

    Returns:
        Loaded model
    """
    kwargs = dict(
        dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if model_config is not None:
        kwargs["config"] = model_config
    return AutoModelForCausalLM.from_pretrained(path, **kwargs)


def load_models(config: TestingConfig) -> tuple:
    """
    Load and initialise all models required for testing
    
    Args:
        config: TestingConfig carrying model paths, dtype, and MemDec params
        
    Returns:
        Tuple of (tokenizer, base_model, knn_generator, memory_decoder)
        
    Raises:
        Exception: Re-raised after logging when a critical loading error occurs
    """
    logger.info("Loading models...")
    try:
        device      = setup_device()
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        device_map  = "auto" if device.type == "cuda" else None
        tokenizer = initialize_tokenizer(config.base_model)
        base_model_config = AutoConfig.from_pretrained(config.base_model)
        cleanup_qwen_config(base_model_config, config.base_model)
        base_lm = _load_lm(config.base_model, config, torch_dtype, device_map, model_config=base_model_config)
        knn_generator_path = config.checkpoint_dir if config.checkpoint_dir else config.base_model
        if config.checkpoint_dir:
            validate_checkpoint_model_type(knn_generator_path, config.base_model, base_model_config)
        try:
            knn_generator = _load_lm(knn_generator_path, config, torch_dtype, device_map)
            if config.checkpoint_dir:
                logger.info(f"✅ KNN generator loaded from checkpoint: {knn_generator_path}")
            else:
                logger.info("✅ KNN generator = base model (no checkpoint supplied)")
        except Exception as knn_error:
            if config.checkpoint_dir:
                logger.error(f"❌ Failed to load KNN generator from checkpoint: {str(knn_error)}")
                raise
            logger.warning(
                "⚠️ Failed to load KNN generator from base model, "
                "falling back to base_lm."
            )
            knn_generator = base_lm
        base_lm.resize_token_embeddings(len(tokenizer))
        knn_generator.resize_token_embeddings(len(tokenizer))
        base_lm.eval()
        knn_generator.eval()
        joint = MemoryDecoder(base_lm, knn_generator, lmbda=config.lmbda, knn_temp=config.knn_temp)
        if device.type != "cuda":
            joint = joint.to(device)
        memory_decoder = joint
        logger.info(f"✅ MemDec-Module created  λ={config.lmbda}, knn_temp={config.knn_temp}")
        logger.info("ℹ️  MemDec-Module uses trained model weights at inference")
        return tokenizer, base_lm, knn_generator, memory_decoder

    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        raise


# --- QUALITATIVE TESTING ---
def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    config: TestingConfig,
    model_name: str = "Unknown"
) -> List[str]:
    """
    Generate responses for a batch of legal-scenario prompts
    
    Args:
        model: Language model instance (MemoryDecoder or AutoModelForCausalLM)
        tokenizer: Tokenizer matching the model vocabulary
        prompts: Legal scenario prompts to complete (left-padded to equal length)
        config: TestingConfig supplying max_new_tokens, repetition_penalty, etc
        model_name: Display name used in log messages
        
    Returns:
        List of decoded and deduplicated response strings, aligned with `prompts`;
        individual rows that finished early are trimmed at their first EOS.
        Per-row errors are logged and fall back to an empty string.
    """
    device = next(model.parameters()).device
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(list(prompts), return_tensors="pt", padding=True).to(device)
    finally:
        tokenizer.padding_side = prev_padding_side
    for p in prompts:
        logger.info(f"📝 Prompt: '{p[:100]}{'...' if len(p) > 100 else ''}'")
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
    with torch.inference_mode():
        gen_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "pad_token_id": eos_id,
            "eos_token_id": eos_id,
            "repetition_penalty": config.repetition_penalty,
        }
        if config.do_sample:
            gen_kwargs.update({
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
            })
        if isinstance(model, MemoryDecoder):
            out_ids = model.generate(**inputs, generation_config=GenerationConfig(**gen_kwargs))
        else:
            out_ids = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    responses = []
    for prompt, row in zip(prompts, out_ids):
        new_tokens = row[prompt_len:]
        stop_pos = (new_tokens == eos_id).nonzero()
        if len(stop_pos):
            new_tokens = new_tokens[:stop_pos[0].item() + 1]
        if len(new_tokens) < config.max_new_tokens:
            logger.debug(
                f"⚠️ {model_name} generated {len(new_tokens)} tokens "
                f"(limit {config.max_new_tokens}) – stopped early (EOS or natural completion)"
            )
        elif len(new_tokens) > config.max_new_tokens:
            logger.debug(
                f"⚠️ {model_name} generated {len(new_tokens)} tokens "
                f"(expected {config.max_new_tokens})."
            )
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = remove_repetitions(response, prompt)
        # Fallback
        if not response.strip():
            full_output = tokenizer.decode(row, skip_special_tokens=True)
            if full_output.startswith(prompt):
                response = full_output[len(prompt):].strip()
        logger.info(f"🤖 {model_name} - Response: '{response[:150]}{'...' if len(response) > 150 else ''}'")
        logger.info(f"📊 {model_name} - Generated {len(new_tokens)} tokens")
        responses.append(response)
    return responses


def remove_repetitions(text: str, prompt: str) -> str:
    """
    Remove repetitive segments from generated text
    
    Args:
        text: Generated model output that may contain repetitive segments
        prompt: Original prompt string; its prefix filters echoed content
        
    Returns:
        Deduplicated text with a trailing period appended when missing
    """
    segments        = re.split(r'(?<=[.!?])\s+|\n+', text)
    unique_segments = []
    seen            = set()
    for seg in segments:
        seg = seg.strip()
        if seg and seg not in seen and not seg.startswith(prompt[:50]):
            unique_segments.append(seg)
            seen.add(seg)
    result = " ".join(unique_segments)
    if result and not re.search(r'[.!?]$', result):
        result += "."
    return result


def filter_prompts_by_tasks(prompts: List[str], tasks: Optional[List[str]]) -> List[str]:
    """
    Filter a scenario's prompts to the requested task types
    
    Args:
        prompts: Ordered list of all prompts for a scenario (length 4)
        tasks: Task-type keys to include; None returns the full list
        
    Returns:
        Filtered list of prompts in their original order
    """
    if not tasks:
        return prompts
    task_mapping = {
        "question_answering": [0],
        "completion": [1],
        "summarization": [2],
        "content_creation": [3]
    }
    filtered_indices = []
    for task in tasks:
        if task in task_mapping:
            filtered_indices.extend(task_mapping[task])
        else:
            logger.warning(f"⚠️ Unknown task '{task}'. Available: {list(task_mapping.keys())}")
    filtered_indices = sorted(set(filtered_indices))
    filtered_prompts = []
    for i, prompt in enumerate(prompts):
        if i in filtered_indices:
            filtered_prompts.append(prompt)
    return filtered_prompts

# --- SOURCE-CHECK ---
# Matches: legal §-refs (§355, §§312g), short abbreviations (BGB, KSchG), and normal words (2+ chars)
_TOKEN_RE = re.compile(r"§§?[0-9]+[a-z]*|[a-zA-ZäöüßÄÖÜ]{2,}[0-9]*|[0-9]+[a-zA-ZäöüßÄÖÜ]{2,}")

@lru_cache(maxsize=1)
def _load_sources() -> Dict:
    """
    Load and cache the sources.json reference file
    
    Returns:
        Parsed JSON dict {"sources": {scenario: {key: {"content": str}}}};
        empty dict on any load failure (missing file, parse error)
    """
    try:
        from src.utils import get_pipeline_value
        sources_path = get_pipeline_value("pipeline.sources_path", None)
        if sources_path:
            sources_path = Path(sources_path)
        else:
            sources_path = PROJECT_ROOT / "config" / "sources.json"
    except (ImportError, AttributeError):
        sources_path = PROJECT_ROOT / "config" / "sources.json"
    
    try:
        with open(sources_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"⚠️ sources.json not found at {sources_path}. Source-check disabled.")
        return {}
    except Exception as e:
        logger.warning(f"⚠️ Failed to load sources.json ({sources_path}): {e}. Source-check disabled.")
        return {}


def _get_scenario_source_texts(scenario_name: str) -> List[str]:
    """
    Extract all reference content strings for a given scenario
    
    Args:
        scenario_name: Key matching a top-level entry in sources
            (e.g. "withdrawals", "employment")
        
    Returns:
        List of raw content strings; empty list if scenario not found
        or sources.json could not be loaded
    """
    data = _load_sources()
    sources = data.get("sources", {}) if isinstance(data, dict) else {}
    scenario_sources = sources.get(scenario_name, {}) if isinstance(sources, dict) else {}
    texts: List[str] = []
    if isinstance(scenario_sources, dict):
        for v in scenario_sources.values():
            if isinstance(v, dict):
                c = v.get("content")
                if isinstance(c, str) and c.strip():
                    texts.append(c)
    return texts


def _normalize_for_match(text: str) -> str:
    """
    Normalise text for fuzzy matching and token extraction
    
    Args:
        text: Raw string to normalise
        
    Returns:
        Lowercased, whitespace-collapsed string
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _simple_source_check(response: str, scenario_name: str) -> Dict:
    """
    Compute lightweight source-fidelity scores for a model response
    
    Args:
        response: Generated model output to evaluate
        scenario_name: Scenario key used to look up reference source texts
        
    Returns:
        Dict with "enabled", "max_fuzzy" and "max_overlap" when scoring
        succeeds; {"enabled": False, "reason": str} when no sources exist;
        {"enabled": True, "max_fuzzy": 0.0, "max_overlap": 0.0} for empty
        or invalid responses
    """
    if not isinstance(response, str) or not response.strip():
        return {"enabled": True, "max_fuzzy": 0.0, "max_overlap": 0.0}
    sources = _get_scenario_source_texts(scenario_name)
    if not sources:
        return {"enabled": False, "reason": "no_sources_for_scenario"}
    resp_norm   = _normalize_for_match(response)
    resp_tokens = set(_TOKEN_RE.findall(resp_norm))
    max_fuzzy = 0.0
    max_overlap = 0.0
    for s in sources:
        s_norm = _normalize_for_match(s)
        fuzzy = difflib.SequenceMatcher(None, resp_norm, s_norm).ratio()
        if fuzzy > max_fuzzy:
            max_fuzzy = fuzzy
        s_tokens = set(_TOKEN_RE.findall(s_norm))
        if resp_tokens and s_tokens:
            # Jaccard similarity coefficient
            overlap = len(resp_tokens & s_tokens) / float(len(resp_tokens | s_tokens))
            if overlap > max_overlap:
                max_overlap = overlap
    return {
        "enabled": True,
        "max_fuzzy": round(max_fuzzy, 3),
        "max_overlap": round(max_overlap, 3),
    }


def test_legal_scenario(
    scenario_name: str,
    scenario_data: Dict,
    tokenizer,
    base_model,
    memory_decoder,
    config: TestingConfig
) -> Dict:
    """
    Run all prompts of a single legal scenario through both models
    
    Args:
        scenario_name: Key identifying the scenario in loaded test config
        scenario_data: Dict with 'title', 'task', and 'prompts' keys
        tokenizer: Tokenizer matching both models' vocabulary
        base_model: Baseline AutoModelForCausalLM for comparison
        memory_decoder: MemoryDecoder wrapping base_lm and knn_generator
        config: TestingConfig controlling generation and prompt filtering
        
    Returns:
        Dict with scenario_name, title, tasks, prompts, base_model_responses,
        memory_decoder_responses, and source_checks (each per-prompt check
        entry also carries a rounded per-model 'gen_time_s' key)
        
    Raises:
        None: Individual prompt errors are logged; processing continues
    """
    logger.info(f"Testing scenario: {scenario_data['title']} (task: {scenario_data.get('task', 'mixed')})")
    filtered_prompts = filter_prompts_by_tasks(scenario_data["prompts"], config.tasks)
    if not filtered_prompts:
        logger.warning(f"⚠️ No prompts found for specified tasks. Using all prompts.")
        filtered_prompts = scenario_data["prompts"]
    actual_task = "mixed"
    if config.tasks and len(config.tasks) == 1:
        actual_task = config.tasks[0]
    elif config.tasks and len(config.tasks) > 1:
        actual_task = f"mixed_{'_'.join(config.tasks)}"
    logger.info(f"📝 Testing {len(filtered_prompts)} prompts ({len(scenario_data['prompts'])} total) - Task: {actual_task}")
    results = {
        "scenario_name": scenario_name,
        "title": scenario_data["title"],
        "tasks": config.tasks,
        "prompts": [],
        "base_model_responses": [],
        "memory_decoder_responses": [],
        "source_checks": []
    }
    gen_batch = max(1, config.batch_size)
    for start in range(0, len(filtered_prompts), gen_batch):
        chunk = filtered_prompts[start:start + gen_batch]
        logger.info(f"  Prompts {start + 1}-{start + len(chunk)}/{len(filtered_prompts)}")
        results["prompts"].extend(chunk)
        t0          = time.time()
        memdec_resps = generate_responses(memory_decoder, tokenizer, chunk, config, "MemoryDecoder")
        memdec_time = round((time.time() - t0) / len(chunk), 3)
        results["memory_decoder_responses"].extend(memdec_resps)
        base_resps = None
        base_time = None
        if config.compare_with_base:
            t0         = time.time()
            base_resps = generate_responses(base_model, tokenizer, chunk, config, "Base Model")
            base_time  = round((time.time() - t0) / len(chunk), 3)
            results["base_model_responses"].extend(base_resps)
        for i in range(len(chunk)):
            check_entry = {
                "memory_decoder": {"gen_time_s": memdec_time,
                                   **_simple_source_check(memdec_resps[i], scenario_name)}
            }
            if config.compare_with_base:
                check_entry["base_model"] = {"gen_time_s": base_time,
                                             **_simple_source_check(base_resps[i], scenario_name)}
            results["source_checks"].append(check_entry)
    return results


def summarize_source_checks(all_results: List[Dict], compare_with_base: bool) -> Dict:
    """
    Aggregate per-prompt source-check scores into overall verdicts

    Args:
        all_results: Scenario result dicts, each carrying a 'source_checks'
            list of per-prompt {model: {gen_time_s, enabled, max_fuzzy,
            max_overlap}} dicts
        compare_with_base: Whether 'base_model' scores exist for pairwise
            win-count comparison

    Returns:
        Dict with 'overall' and 'by_scenario' aggregates. Each aggregate holds
        per-model 'n', 'mean_max_fuzzy', 'mean_max_overlap', combined 'score'
        (mean of both means), pairwise 'fuzzy_wins'/'overlap_wins', and
        'winner' (higher score; None when no pairwise comparison possible).
        Scenarios without enabled checks yield zeroed aggregates.
    """
    keys = ["memory_decoder", "base_model"] if compare_with_base else ["memory_decoder"]

    def _aggregate(checks: List[Dict]) -> Dict:
        totals = {k: {"n": 0, "fuzzy": 0.0, "overlap": 0.0} for k in keys}
        wins   = {k: {"fuzzy_wins": 0, "overlap_wins": 0} for k in keys}
        pairs  = 0
        for c in checks:
            scored = [k for k in keys if isinstance(c.get(k), dict) and c[k].get("enabled")]
            for k in scored:
                totals[k]["n"]       += 1
                totals[k]["fuzzy"]   += c[k].get("max_fuzzy", 0.0)
                totals[k]["overlap"] += c[k].get("max_overlap", 0.0)
            if len(scored) == 2:
                pairs += 1
                m, b = c["memory_decoder"], c["base_model"]
                if m["max_fuzzy"]   > b["max_fuzzy"]:   wins["memory_decoder"]["fuzzy_wins"]   += 1
                elif b["max_fuzzy"] > m["max_fuzzy"]:   wins["base_model"]["fuzzy_wins"]     += 1
                if m["max_overlap"] > b["max_overlap"]: wins["memory_decoder"]["overlap_wins"] += 1
                elif b["max_overlap"] > m["max_overlap"]: wins["base_model"]["overlap_wins"]   += 1
        agg: Dict[str, Any] = {"evaluated_pairs": pairs}
        best_key, best_score = None, -1.0
        for k in keys:
            n      = totals[k]["n"]
            mean_f = totals[k]["fuzzy"]   / n if n else 0.0
            mean_o = totals[k]["overlap"] / n if n else 0.0
            score  = (mean_f + mean_o) / 2
            agg[k] = {
                "n":                 n,
                "mean_max_fuzzy":    round(mean_f, 3),
                "mean_max_overlap":  round(mean_o, 3),
                "score":             round(score, 3),
                "fuzzy_wins":        wins[k]["fuzzy_wins"],
                "overlap_wins":      wins[k]["overlap_wins"],
            }
            if n and score > best_score:
                best_key, best_score = k, score
        agg["winner"] = best_key if (compare_with_base and pairs) else None
        return agg

    all_checks = [c for r in all_results for c in r.get("source_checks", [])]
    return {
        "overall":     _aggregate(all_checks),
        "by_scenario": {r["scenario_name"]: _aggregate(r.get("source_checks", [])) for r in all_results},
    }


def save_results(all_results: List[Dict], config: TestingConfig, summary: Optional[Dict] = None) -> None:
    """
    Persist test results, run configuration and aggregate summary to a
    timestamped JSON file

    Args:
        all_results: List of per-scenario result dicts from test_legal_scenario
        config: TestingConfig for this run; stored verbatim under 'config' key
        summary: Optional aggregate dict from summarize_source_checks; stored
            under the 'summary' key

    Returns:
        None
    """
    timestamp   = time.strftime("%Y%m%d_%H%M%S")
    config_dict = config.__dict__.copy()
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value.relative_to(PROJECT_ROOT)).replace("\\", "/")
        elif isinstance(value, str) and str(PROJECT_ROOT) in value:
            config_dict[key] = str(Path(value).relative_to(PROJECT_ROOT)).replace("\\", "/")
    model_keyword = extract_model_identifier(config.base_model) or "model"
    out_path = Path(config.results_dir) / f"test_results_{model_keyword}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": config_dict, "summary": summary, "results": all_results},
                  f, ensure_ascii=False, indent=2)
    try:
        display_path = out_path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = out_path
    logger.info(f"✅ Results saved to {display_path}")


# --- MAIN ---
def main(config: TestingConfig) -> None:
    """
    Orchestrate the full qualitative testing pipeline
    
    Args:
        config: Fully populated TestingConfig for this run
    
    Returns:
        None
    """
    log_blank_line(logger)
    logger.info("=" * 60)
    logger.info("🧪 STEP 5: Qualitative Source-Fidelity Testing")
    logger.info("=" * 60)
    for k, v in config.__dict__.items():
        logger.info(f"  {k}: {v}")
    try:
        os.makedirs(config.results_dir, exist_ok=True)
        os.makedirs(config.knowledge_base_path, exist_ok=True)
        tokenizer, base_model, knn_generator, memory_decoder = load_models(config)
        LEGAL_TESTS = load_test_config()
        if not LEGAL_TESTS:
            logger.error("❌ No test cases found in config. Exiting.")
            return
        scenarios_to_test = LEGAL_TESTS
        if config.scenarios:
            scenarios_to_test = {}
            for name in config.scenarios:
                if name in LEGAL_TESTS:
                    scenarios_to_test[name] = LEGAL_TESTS[name]
                else:
                    logger.warning(f"⚠️ Unknown scenario '{name}'. Available: {list(LEGAL_TESTS.keys())}")
            if not scenarios_to_test:
                logger.error("❌ No valid scenarios. Exiting.")
                return
        all_results = []
        for scenario_name, scenario_data in scenarios_to_test.items():
            try:
                result = test_legal_scenario(
                    scenario_name, scenario_data,
                    tokenizer, base_model, memory_decoder, config
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"❌ Scenario {scenario_name} failed: {e}")
                continue

        logger.info("=" * 60)
        total_scenarios = len(all_results)
        total_prompts = sum(len(r["prompts"]) for r in all_results)
        logger.info(f"📊 Total scenarios tested: {total_scenarios}")
        logger.info(f"📊 Total prompts processed: {total_prompts}")
        summary = summarize_source_checks(all_results, config.compare_with_base)
        overall = summary["overall"]
        for model_key in ("memory_decoder", "base_model"):
            if model_key in overall and isinstance(overall[model_key], dict):
                s = overall[model_key]
                logger.info(
                    f"🔎 Source-check {model_key}: mean_fuzzy={s['mean_max_fuzzy']:.3f}  "
                    f"mean_overlap={s['mean_max_overlap']:.3f}  score={s['score']:.3f}  "
                    f"(fuzzy_wins={s['fuzzy_wins']}, overlap_wins={s['overlap_wins']})"
                )
        if overall.get("winner"):
            logger.info(f"🏆 Source-check winner: {overall['winner']} "
                        f"(over {overall['evaluated_pairs']} compared prompts)")
        if config.save_results:
            save_results(all_results, config, summary)
        logger.info("=" * 60)
        logger.info("ℹ️  Next step (optional): Quantitative Evaluation")
        if config.checkpoint_dir:
            checkpoint_name = Path(config.checkpoint_dir).name
            logger.info(f" Run: python -m src.step5_evaluation <tokenized_data name> --checkpoint {checkpoint_name}")
        else:
            logger.info("ℹ️  No checkpoint available for quantitative evaluation")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Error during testing: {str(e)}")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --- CLI ---
def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments (use default config values if not provided) for qualitative testing
    
    Returns:
        argparse.Namespace with all arguments resolved to their final values
    """
    parser = argparse.ArgumentParser(description="Test MemDec-Module with legal scenarios")
    default_config = TestingConfig()
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--model", type=str, default=None)
    base_parser.add_argument("--checkpoint", type=str, default=None)
    early_args = base_parser.parse_known_args()[0]
    config_checkpoint = get_pipeline_value("steps.step5_testing.checkpoint", default_config.checkpoint_dir)
    # Determine base_model: .env > CLI > step config > pipeline default > checkpoint > system default
    config_base_model = resolve_base_model(
        cli_model=early_args.model,
        step_config_key="steps.step5_testing",
        checkpoint=early_args.checkpoint or config_checkpoint,
        output_dir=str(default_config.output_dir),
    )
    config_lmbda = get_pipeline_value("steps.step5_testing.lmbda", default_config.lmbda)
    config_knn_temp = get_pipeline_value("steps.step5_testing.knn_temp", default_config.knn_temp)
    config_max_new_tokens = get_pipeline_value("steps.step5_testing.max_new_tokens", default_config.max_new_tokens)
    config_repetition_penalty = get_pipeline_value("steps.step5_testing.repetition_penalty", default_config.repetition_penalty)
    config_do_sample = get_pipeline_value("steps.step5_testing.do_sample", default_config.do_sample)
    config_temperature = get_pipeline_value("steps.step5_testing.temperature", default_config.temperature)
    config_top_p = get_pipeline_value("steps.step5_testing.top_p", default_config.top_p)
    config_top_k = get_pipeline_value("steps.step5_testing.top_k", default_config.top_k)
    config_scenarios = get_pipeline_value("steps.step5_testing.scenarios", default_config.scenarios)
    config_tasks = get_pipeline_value("steps.step5_testing.tasks", default_config.tasks)
    config_save_results = get_pipeline_value("steps.step5_testing.save_results", default_config.save_results)
    config_compare_base = get_pipeline_value("steps.step5_testing.compare_base", default_config.compare_with_base)
    action_save = "store_false" if config_save_results else "store_true"
    action_compare = "store_false" if config_compare_base else "store_true"
    action_sample = "store_false" if config_do_sample else "store_true"
    # Variable defaults via pipeline_config
    parser.add_argument("--model", type=str, default=config_base_model,
                        help=f"Base model preset (gemma3/qwen3.5/smollm3) [default: auto-derived from checkpoint or default]")
    parser.add_argument("--checkpoint", type=str, default=config_checkpoint,
                        help="Trained checkpoint to load as knn_generator. Accepts: <step_5000>, <outputs/step_5000>, absolute path, or <latest> (default).")
    parser.add_argument("--max-new-tokens", type=int, default=config_max_new_tokens,
                        help=f"Max tokens to generate [default: {config_max_new_tokens}]")
    parser.add_argument("--repetition-penalty", type=float, default=config_repetition_penalty,
                        help=f"Repetition penalty for generation (1.0 = off) [default: {config_repetition_penalty}]")
    parser.add_argument("--do-sample", action=action_sample,
                        help=f"Enable multinomial sampling instead of greedy decoding [default: {config_do_sample}]")
    parser.add_argument("--temperature", type=float, default=config_temperature,
                        help=f"Sampling temperature (only with --do-sample) [default: {config_temperature}]")
    parser.add_argument("--top-p", type=float, default=config_top_p,
                        help=f"Nucleus sampling threshold (only with --do-sample) [default: {config_top_p}]")
    parser.add_argument("--top-k", type=int, default=config_top_k,
                        help=f"Top-k sampling filter, 0 = off (only with --do-sample) [default: {config_top_k}]")
    parser.add_argument("--lmbda", type=float, default=config_lmbda,
                        help=f"Interpolation weight λ [default: {config_lmbda}]")
    parser.add_argument("--knn-temp", type=float, default=config_knn_temp,
                        help=f"knn_generator logit temperature [default: {config_knn_temp}]")
    parser.add_argument("--scenarios", type=str, nargs="+", default=config_scenarios,
                        help=f"Scenarios to run (e.g. --scenarios withdrawals employment) [default: {config_scenarios}]")
    parser.add_argument("--tasks", type=str, nargs="+", default=config_tasks,
                        help=f"Task types to run (e.g. --tasks question_answering completion) [default: {config_tasks}]")
    parser.add_argument("--save-results", action=action_save,
                        help=f"Skip writing result JSON files [default: {config_save_results}]")
    parser.add_argument("--compare-base", action=action_compare,
                        help=f"Skip base model comparison [default: {config_compare_base}]")
    # Fixed defaults via PretrainConfig (cannot be changed via CLI)
    parser.add_argument("--knowledge-base-path", type=str, default=default_config.knowledge_base_path,
                        help=f"[FIXED] KNN datastore directory [default: {default_config.knowledge_base_path}]")
    parser.add_argument("--output-dir", type=str, default=default_config.output_dir,
                        help=f"[FIXED] Output / checkpoint directory [default: {default_config.output_dir}]")
    parser.add_argument("--results-dir", type=str, default=default_config.results_dir,
                        help=f"[FIXED] Where to write result JSON files [default: {default_config.results_dir}]")
    parser.add_argument("--batch-size", type=int, default=default_config.batch_size,
                        help=f"[FIXED] Batch size [default: {default_config.batch_size}]")
    args = parser.parse_args()
    args.model = config_base_model
    if args.knowledge_base_path != default_config.knowledge_base_path:
        parser.error(f"--knowledge-base-path is fixed to '{default_config.knowledge_base_path}' and cannot be changed via CLI.")
    if args.output_dir != default_config.output_dir:
        parser.error(f"--output-dir is fixed to '{default_config.output_dir}' and cannot be changed via CLI.")
    if args.results_dir != default_config.results_dir:
        parser.error(f"--results-dir is fixed to '{default_config.results_dir}' and cannot be changed via CLI.")
    if args.batch_size != default_config.batch_size:
        parser.error(f"--batch-size is fixed to '{default_config.batch_size}' and cannot be changed via CLI.")
    return args


if __name__ == "__main__":
    args = parse_arguments()
    model_keyword = extract_model_identifier(args.model)
    model_info = get_model_config(model_keyword)
    resolved_checkpoint = _resolve_checkpoint_or_exit(args.checkpoint, args.output_dir)
    config = TestingConfig(
        base_model=model_info['name'],
        checkpoint_dir=resolved_checkpoint,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        lmbda=args.lmbda,
        knn_temp=args.knn_temp,
        batch_size=args.batch_size,
        scenarios=args.scenarios,
        tasks=args.tasks,
        save_results=args.save_results,
        compare_with_base=args.compare_base,
    )
    main(config)