#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ----------------------------------------------------------------------
# MODIFICATION NOTICE
# ----------------------------------------------------------------------
# Modified 2026 by Gii-DE for the MemDec-Universal project.
# Changes include:
# - Code Refactoring: Removed unused imports and refined overall code layout/formatting
# - Performance & Memory Optimization: Added mixed precision detection (bf16/fp16) and CPU-only Accelerator configuration
# - Performance & Memory Optimization: Added gradient checkpointing for CPU and Tensor Core embedding padding (multiple of 8)
# - Performance & Memory Optimization: Added memory-efficient batch iteration with psutil RAM monitoring and OOM handling
# - Performance & Memory Optimization: Rewrote knn_collate_fn() with direct Arrow->NumPy caching (shared label/token_id buffer), one-hot fast path, and multi-token fallback
# - Performance & Memory Optimization: KNN datastore memory-mapped via Dataset.from_file() with zero-copy keys/vals conversion (metadata-only column ops: drop the multi-GB keys column, rename vals->label, share the vals buffer for token_id; no table rebuild or fingerprint hashing)
# - Performance & Memory Optimization: Switched Unsloth full fine-tuning to bfloat16 (UNSLOTH_DTYPE), halving memory footprint and enabling double-buffered gradient offloading on low-VRAM GPUs
# - Performance & Memory Optimization: Added bitsandbytes AdamW8bit optimizer on CUDA (~75% smaller optimizer states) with torch.optim.AdamW fallback and optimizer-class logging
# - Model Architecture Support: Added Qwen3.5 vision_config cleanup and guarded Unsloth fast-finetuning path
# - Model Architecture Support: Added multi-format KNN datastore loading (Arrow/Parquet/JSON/JSONL)
# - CLI & UX: Added --no_unsloth/--no-unsloth flag (or NO_UNSLOTH env var) to skip the unsloth import entirely; filtered unsloth deprecation warnings
# - Robustness: Added cleanup_old_checkpoints(), robust checkpoint resumption, has_knn warning fallback, and tensor-guarded batch device moves (knn_probs=None fast path)
# - Robustness: Added optimizer-state compatibility validation on checkpoint resume (torch 'exp_avg' vs bitsandbytes 'state1') with automatic momentum reset to prevent KeyError at first optimizer.step()
#
# The original code is available at: 
# https://github.com/LUMIA-Group/MemoryDecoder
# ----------------------------------------------------------------------

import os, sys, time, warnings, random
warnings.filterwarnings("ignore", message=".*is deprecated.*torch.backends.*", category=UserWarning)
UNSLOTH_AVAILABLE = False
FastLanguageModel = None
FastModel = None
_NO_UNSLOTH = os.environ.get("NO_UNSLOTH", "").strip().lower() in ("1", "true", "yes")
if not _NO_UNSLOTH and not any(a in sys.argv for a in ("--no_unsloth", "--no-unsloth")):
    try:
        from unsloth import FastLanguageModel
        UNSLOTH_AVAILABLE = True
        try:
            from unsloth import FastModel
        except ImportError:
            FastModel = None
    except Exception:
        UNSLOTH_AVAILABLE = False
try:
    from bitsandbytes.optim import AdamW8bit
except Exception:
    AdamW8bit = None

import inspect
import argparse
import logging
import math
import numpy as np
import pyarrow as pa
import psutil, shutil, gc
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset, load_from_disk, Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    DataCollatorForLanguageModeling,
)
from loguru import logger

from MemoryDecoder.utils.cal_loss import kl_loss_token, kl_loss_evaluate


MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--dataset_split_name", type=str, default="test")
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--validation_split_percentage", default=5)
    parser.add_argument("--model_name_or_path", type=str, required=False)
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model_type", type=str, default=None, choices=MODEL_TYPES)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--no_keep_linebreaks", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str)
    parser.add_argument("--hub_token", type=str)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--checkpointing_steps", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--report_to", type=str, default="all")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    # KNN args
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--init_lm_head", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--group_name", type=str)
    parser.add_argument("--lmbda", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--knn_save_path", type=str, default=None)
    parser.add_argument("--no_unsloth", "--no-unsloth", dest="no_unsloth", action="store_true",
                        help="Disable Unsloth fast full-finetuning kernels.")
    
    args = parser.parse_args()
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    return args


def knn_collate_fn(batch, knn_dstore=None, vocab_size=None, base_collator=default_data_collator):
    """Fast collator for KNN datastores. Avoids building dense (B*T, vocab_size) tensors
    for one-hot KNN distributions by passing knn_label only."""
    collated_batch = base_collator(batch)
    if (
        all("dstore_range" in item and item["dstore_range"] is not None for item in batch)
        and knn_dstore is not None
        and vocab_size is not None
    ):
        try:
            if not hasattr(knn_dstore, '_cached_numpy'):
                t0 = time.time()
                def _col_to_numpy(name):
                    col = knn_dstore.data.column(name)
                    if isinstance(col, pa.ChunkedArray):
                        col = col.combine_chunks()
                    return col.to_numpy(zero_copy_only=False)
                labels_arr = _col_to_numpy('label')
                token_ids_arr = _col_to_numpy('token_id')
                if np.array_equal(labels_arr, token_ids_arr):
                    token_ids_arr = labels_arr
                knn_dstore._cached_numpy = {
                    'label': labels_arr,
                    'token_id': token_ids_arr,
                    'prob': _col_to_numpy('prob'),
                }
                logger.info(f"kNN collator cache built in {time.time() - t0:.1f}s")
            cache = knn_dstore._cached_numpy
            starts = [int(item["dstore_range"][0]) for item in batch]
            ends = [int(item["dstore_range"][1]) for item in batch]
            labels = np.concatenate([cache['label'][s:e] for s, e in zip(starts, ends)])
            token_ids = np.concatenate([cache['token_id'][s:e] for s, e in zip(starts, ends)])
            probs = np.concatenate([cache['prob'][s:e] for s, e in zip(starts, ends)])
            knn_label = torch.from_numpy(labels).long()
            token_ids_t = torch.from_numpy(token_ids).long()
            probs_t = torch.from_numpy(probs).float()
            total_len = token_ids_t.shape[0]
            is_one_hot = (
                token_ids_t.dim() == 1 and probs_t.dim() == 1
                and torch.allclose(probs_t, torch.ones_like(probs_t))
            )
            if is_one_hot:
                collated_batch["knn_label"] = knn_label
                collated_batch["knn_probs"] = None
            else:
                knn_probs = torch.zeros(total_len, vocab_size)
                for i in range(total_len):
                    tid = token_ids_t[i]
                    p = probs_t[i]
                    if isinstance(tid, (list, np.ndarray)) or (torch.is_tensor(tid) and tid.numel() > 1):
                        if isinstance(p, (list, np.ndarray)) or (torch.is_tensor(p) and p.numel() > 1):
                            for t, prob_val in zip(tid, p):
                                knn_probs[i, t] = prob_val
                        else:
                            for t in tid:
                                knn_probs[i, t] = p
                    else:
                        knn_probs[i, tid.item()] = p
                collated_batch["knn_label"] = knn_label
                collated_batch["knn_probs"] = knn_probs
        except Exception as e:
            logger.warning(f"Error processing kNN data in collator: {str(e)}")

    return collated_batch


def cleanup_old_checkpoints(output_dir: str, current_checkpoint: str) -> None:
    if not os.path.exists(output_dir):
        return
    logger.info(f"Cleaning up old checkpoints, keeping: {current_checkpoint}")
    checkpoints = [
        item for item in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, item)) and ('checkpoint' in item or item.startswith('step_') or item.startswith('epoch_'))
    ]
    for checkpoint in checkpoints:
        if checkpoint != current_checkpoint:
            checkpoint_path = os.path.join(output_dir, checkpoint)
            try:
                shutil.rmtree(checkpoint_path)
                logger.info(f"🗑️ Removed old checkpoint: {checkpoint}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to remove checkpoint {checkpoint}: {str(e)}")


UNSLOTH_DTYPE = torch.bfloat16

def load_model_unsloth(model_path, args):
    kwargs = dict(
        model_name=model_path,
        max_seq_length=args.block_size,
        dtype=UNSLOTH_DTYPE,
        load_in_4bit=False,
        full_finetuning=True,
        trust_remote_code=args.trust_remote_code,
    )
    for name, loader in (("FastLanguageModel", FastLanguageModel), ("FastModel", FastModel)):
        if loader is None:
            continue
        try:
            model, _ = loader.from_pretrained(**kwargs)
            logger.info(f"⚡ Loaded {model_path} via Unsloth {name} (full fine-tuning)")
            return model
        except Exception as e:
            logger.warning(f"⚠️ Unsloth {name} failed for {model_path}: {e}")
    return None


def main(progress_bar=None):
    args = parse_args()
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    mixed_precision = "no"
    if torch.cuda.is_available():
        mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        cpu=not torch.cuda.is_available(),
        **accelerator_log_kwargs
    )
    
    if args.report_to == "wandb" and args.with_tracking:
        accelerator.init_trackers(
            project_name=args.project_name, 
            config=args,
            init_kwargs={
                "wandb": {
                    "name": args.run_name,
                    "group": args.group_name,
                    "save_code": True,
                },
            }
        )

    log_level = logging.INFO if accelerator.is_local_main_process else logging.ERROR
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <blue>{process.name}</blue> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )
    
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            frame, depth = inspect.currentframe(), 0
            while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
                frame = frame.f_back
                depth += 1
            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=log_level, force=True)
    transformers.utils.logging.disable_default_handler()
    transformers.utils.logging.add_handler(InterceptHandler())
    
    if args.seed is not None:
        set_seed(args.seed)

    # ------------------ Load Config, Tokenizer & Model ------------------
    is_checkpoint = args.resume_from_checkpoint and os.path.exists(os.path.join(args.resume_from_checkpoint, "config.json"))
    if is_checkpoint:
        config = AutoConfig.from_pretrained(args.resume_from_checkpoint, trust_remote_code=args.trust_remote_code)
        model_path = args.resume_from_checkpoint
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
        model_path = args.model_name_or_path
    else:
        config = CONFIG_MAPPING[args.model_type]()
        model_path = None

    if model_path and not args.from_scratch:
        if not is_checkpoint and ("qwen" in model_path.lower() or getattr(config, 'model_type', '') == "qwen3_5_text"):
            if hasattr(config, "vision_config"):
                delattr(config, "vision_config")
            if hasattr(config, "text_config"):
                config.architectures = ["Qwen3_5ForCausalLM"]
                config.model_type = "qwen3_5_text"

        model = None
        use_unsloth = (
            UNSLOTH_AVAILABLE
            and not args.no_unsloth
            and torch.cuda.is_available()
            and accelerator.num_processes == 1
            and "qwen" not in model_path.lower()
            and "qwen" not in str(getattr(config, "model_type", "")).lower()
        )
        if use_unsloth:
            model = load_model_unsloth(model_path, args)
            
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                from_tf=bool(".ckpt" in model_path),
                config=config,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                trust_remote_code=args.trust_remote_code,
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    if not torch.cuda.is_available():
        model.gradient_checkpointing_enable()

    tokenizer_path = args.resume_from_checkpoint if is_checkpoint else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.eos_token_id
        if hasattr(model, "config"):
            model.config.pad_token_id = tokenizer.eos_token_id

    # ------------------ Preprocess Dataset ------------------
    if args.train_file:
        if args.train_file.endswith('.json'):
            lm_datasets = load_dataset('json', data_files=args.train_file)['train']
        else:
            lm_datasets = load_from_disk(args.train_file)
    else:
        raise ValueError("No train_file provided.")

    vocab_size = len(tokenizer)
    if model.get_input_embeddings().weight.shape[0] != vocab_size:
        if torch.cuda.is_available():
            model.resize_token_embeddings(vocab_size, pad_to_multiple_of=8)
        else:
            model.resize_token_embeddings(vocab_size)

    # ------------------ Load KNN Datastore ------------------
    knn_dstore = None
    if args.knn_save_path:
        if os.path.isdir(args.knn_save_path):
            data_files = [f for f in os.listdir(args.knn_save_path) if f.endswith(('.arrow', '.parquet', '.json', '.jsonl'))]
            if not data_files:
                raise ValueError(f"No valid data files found in {args.knn_save_path}")
            data_file = os.path.join(args.knn_save_path, data_files[0])
        else:
            data_file = args.knn_save_path
        try:
            knn_dstore = Dataset.from_file(data_file)
        except Exception:
            knn_dstore = load_dataset('arrow', data_files=data_file)['train']

        available_columns = set(knn_dstore.column_names)
        if available_columns == {'keys', 'vals'}:
            logger.info("Converting keys/vals format to id_cnt/token_id/prob/label format...")
            t0 = time.time()
            num_entries = len(knn_dstore)
            vals_np = knn_dstore.data.column('vals').combine_chunks().to_numpy(zero_copy_only=False)
            knn_dstore = knn_dstore.remove_columns('keys')
            knn_dstore = knn_dstore.rename_column('vals', 'label')
            knn_dstore = knn_dstore.add_column('token_id', vals_np)
            knn_dstore = knn_dstore.add_column('id_cnt', np.arange(num_entries, dtype=np.int64))
            knn_dstore = knn_dstore.add_column('prob', np.ones(num_entries, dtype=np.float32))
            logger.info(f"Converted {num_entries} entries in {time.time() - t0:.1f}s")

        knn_dstore.set_format(type='torch', columns=['id_cnt', 'token_id', 'prob', 'label'])

    # ------------------ Data Collator & Loaders ------------------
    smart_base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    collate_fn = partial(
        knn_collate_fn,
        knn_dstore=knn_dstore,
        vocab_size=vocab_size,
        base_collator=smart_base_collator
    ) if knn_dstore is not None else smart_base_collator

    train_dataloader = DataLoader(
        lm_datasets,
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # ------------------ Optimizer & Scheduler ------------------
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer_cls = AdamW8bit if (AdamW8bit is not None and torch.cuda.is_available()) else torch.optim.AdamW
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.learning_rate)
    logger.info(f"Optimizer: {optimizer_cls.__module__}.{optimizer_cls.__name__}")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # ------------------ Resume Logic ------------------
    completed_steps = 0
    starting_epoch = 0
    resume_step = None

    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        try:
            accelerator.load_state(checkpoint_path, strict=False)
        except Exception as e:
            logger.warning(
                "accelerator.load_state rejected checkpoint keys "
                "(canonical 'model.language_model.*' names on disk vs 'model.*' in memory); "
                "weights already loaded via from_pretrained — restoring optimizer/scheduler/RNG directly"
            )
            logger.debug(f"load_state error: {e}")
            try:
                opt_file = os.path.join(checkpoint_path, "optimizer.bin")
                if os.path.exists(opt_file):
                    optimizer.load_state_dict(torch.load(opt_file, map_location="cpu", weights_only=False))
                sched_file = os.path.join(checkpoint_path, "scheduler.bin")
                if os.path.exists(sched_file):
                    lr_scheduler.load_state_dict(torch.load(sched_file, weights_only=False))
                rng_file = os.path.join(checkpoint_path, f"random_states_{accelerator.process_index}.pkl")
                if os.path.exists(rng_file):
                    states = torch.load(rng_file, map_location="cpu", weights_only=False)
                    if "random_state" in states:
                        random.setstate(states["random_state"])
                    if "numpy_random_seed" in states:
                        np.random.set_state(states["numpy_random_seed"])
                    if "torch_manual_seed" in states:
                        torch.set_rng_state(states["torch_manual_seed"].cpu())
                    if torch.cuda.is_available() and "torch_cuda_manual_seed_all" in states:
                        torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed_all"])
            except Exception as e2:
                logger.warning(f"Could not restore trainer state, continuing with model weights only: {e2}")

        expected_key = "state1" if optimizer_cls is AdamW8bit else "exp_avg"
        inner_opt = getattr(optimizer, "optimizer", optimizer)
        loaded_states = list(inner_opt.state.values())
        if loaded_states and any(expected_key not in s for s in loaded_states):
            logger.warning(
                f"Checkpoint optimizer state incompatible with {optimizer_cls.__name__} "
                f"(missing '{expected_key}') — resetting optimizer momentum."
            )
            inner_opt.state.clear()
        path = os.path.basename(checkpoint_path)
        training_difference = os.path.splitext(path)[0]
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # ------------------ Training Loop ------------------
    if progress_bar is None:
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.update(completed_steps)
    
    logging_interval_loss = 0.0
    logging_interval_kl_loss = 0.0
    logging_interval_lm_loss = 0.0
    warned_no_knn = False

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps when resuming
            if epoch == starting_epoch and resume_step is not None and step < resume_step:
                continue
            try:
                # OOM Check
                if psutil.virtual_memory().available < 512 * 1024**2:
                    logger.warning("Low RAM detected, triggering GC...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                batch = {k: v.to(accelerator.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                has_knn = "knn_label" in batch and "knn_probs" in batch
                if not has_knn and not warned_no_knn:
                    logger.warning("No kNN targets found in batch. Training with plain standard CE loss!")
                    warned_no_knn = True

                with accelerator.accumulate(model):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=None if has_knn else batch.get("labels", None)
                    )

                    if has_knn:
                        loss, kl_loss, lm_loss = kl_loss_token(
                            outputs.logits if hasattr(outputs, 'logits') else outputs[0],
                            batch,
                            tokenizer,
                            args,
                            batch["knn_label"],
                            batch.get("knn_probs", None),
                            alpha=args.alpha
                        )
                    else:
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                        kl_loss = torch.tensor(0.0, device=accelerator.device)
                        lm_loss = loss.detach().clone()

                    accelerator.backward(loss)
                    logging_interval_loss += loss.detach().float()
                    logging_interval_kl_loss += kl_loss.detach().float()
                    logging_interval_lm_loss += lm_loss.detach().float()
                    current_loss = loss.item()

                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                        completed_steps += 1
                        progress_bar.update(1)

                        if accelerator.is_local_main_process:
                            current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler.get_last_lr() else 0.0
                            progress_bar.set_postfix({
                                'loss': f"{current_loss:.4f}",
                                'lr': f"{current_lr:.2e}",
                                'epoch': f"{epoch+1}/{args.num_train_epochs}",
                                'mem': f"{psutil.Process().memory_info().rss / 1024**2:.1f}MB"
                            })

                        # Logging
                        if args.logging_steps and completed_steps % args.logging_steps == 0:
                            actual_steps = args.gradient_accumulation_steps
                            avg_loss = accelerator.gather(logging_interval_loss).mean().item() / actual_steps / args.logging_steps
                            avg_kl_loss = accelerator.gather(logging_interval_kl_loss).mean().item() / actual_steps / args.logging_steps
                            avg_lm_loss = accelerator.gather(logging_interval_lm_loss).mean().item() / actual_steps / args.logging_steps
                            
                            accelerator.log({
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                                "kl_loss": avg_kl_loss,
                                "lm_loss": avg_lm_loss,
                                "grad_norm": grad_norm if isinstance(grad_norm, (int, float)) else grad_norm.item(),
                                "epoch": epoch,
                            }, step=completed_steps)

                            logging_interval_loss = 0.0
                            logging_interval_kl_loss = 0.0
                            logging_interval_lm_loss = 0.0

                        # Checkpointing
                        if args.checkpointing_steps and args.checkpointing_steps.isdigit():
                            ch_steps = int(args.checkpointing_steps)
                            if completed_steps % ch_steps == 0:
                                output_dir = os.path.join(args.output_dir or "", f"step_{completed_steps}")
                                accelerator.save_state(output_dir)
                                if accelerator.is_main_process:
                                    unwrapped_model = accelerator.unwrap_model(model)
                                    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
                                    tokenizer.save_pretrained(output_dir)
                                    cleanup_old_checkpoints(args.output_dir, os.path.basename(output_dir))

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    logger.warning(f"CUDA OOM at step {completed_steps}, skipping batch.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            output_dir = os.path.join(args.output_dir or "", f"epoch_{epoch}")
            accelerator.save_state(output_dir)
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
                tokenizer.save_pretrained(output_dir)
                cleanup_old_checkpoints(args.output_dir, os.path.basename(output_dir))

        if completed_steps >= args.max_train_steps:
            break


if __name__ == "__main__":
    main()