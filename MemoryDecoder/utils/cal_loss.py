# ----------------------------------------------------------------------
# MODIFICATION NOTICE
# ----------------------------------------------------------------------
# Modified 2026 by Gii-DE for the MemDec-Universal project.
# Changes include:
# - Code Refactoring: Removed unused imports and refined overall code layout/formatting
# - Performance & Memory Optimization: Added KL_LOSS_CHUNK_SIZE and chunked KL processing with conditional CPU offloading
# - Performance & Memory Optimization: Added try-except with gc.collect() and torch.cuda.empty_cache() on VRAM errors
# - Numerical Stability: Added epsilon smoothing (1e-10 / 1e-8) for zero probabilities
# - Robustness: Fallback to input_ids when labels are missing in kl_loss_token()
# - Correctness: Trim/pad one-hot knn_label to logit sequence length and aligned 2D/3D knn_prob shapes with shifted logits
#
# The original code is available at: 
# https://github.com/LUMIA-Group/MemoryDecoder
# ----------------------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
import gc

EMBED_PAD_NUM = -10000
KL_LOSS_CHUNK_SIZE = 64

def interpolate(knn_log_probs, lm_log_probs, lmbda=0.25):
    interpolated = torch.logaddexp(
        lm_log_probs + np.log(1 - lmbda), 
        knn_log_probs + np.log(lmbda))
    return interpolated

def kl_loss_evaluate(logits, batch, tokenizer, args, knn_label, knn_prob):
    label_probs = knn_prob
    shift_logits = logits[:, :-1].contiguous() # (batch, seq_len-1, vocab_size)
    shift_labels = batch['labels'][:, 1:].contiguous() # (batch, seq_len-1)
    nonpad_mask = shift_labels != -100
    shift_logits = shift_logits[nonpad_mask] # (nonpad b*t, vocab_size)
    shift_labels = shift_labels[nonpad_mask] # (nonpad b*t)
    label_probs = label_probs / (label_probs.sum(dim=-1, keepdim=True) + 1e-8) # Normalize label_probs
    # Ensure that the dimensions match
    assert shift_logits.shape == label_probs.shape, f"shift_logits.shape = {shift_logits.shape}, label_probs.shape = {label_probs.shape}"
    assert torch.all(shift_labels == knn_label), f"shift_labels and knn_label are not the same"
    assert torch.allclose(label_probs.sum(dim=-1), torch.ones_like(label_probs.sum(dim=-1))), f"label_probs does not sum to 1"
    # Compute the label_probs
    shift_probs = F.softmax(shift_logits, dim=-1)
    # Calculate PPL
    label_log_probs = label_probs.log()
    label_log_probs = torch.nan_to_num(label_log_probs, nan=None, neginf=-10000.0)
    lm_log_probs = F.log_softmax(shift_logits, dim=-1)
    interpolate_log_probs = interpolate(label_log_probs, lm_log_probs, lmbda=args.lmbda)
    nll_loss = F.nll_loss(interpolate_log_probs, shift_labels, reduction='sum')
    lm_loss = F.nll_loss(lm_log_probs, shift_labels, reduction='sum')
    token_num = shift_labels.shape[0]
    return nll_loss, lm_loss, token_num

def kl_loss_token(logits, batch, tokenizer, args, knn_label, knn_prob=None, alpha=0.5):
    """MemDec training loss. Fast path uses knn_label directly for one-hot KNN distributions;
    dense knn_prob path is kept for multi-token KNN distributions."""
    try:
        labels = batch.get('labels', batch['input_ids'])
        shift_logits = logits[..., :-1, :].contiguous()  # [batch, seq_len-1, vocab_size]
        shift_labels = labels[..., 1:].contiguous()  # [batch, seq_len-1]
        nonpad_mask = shift_labels != -100
        shift_logits_flat = shift_logits[nonpad_mask]
        shift_labels_flat = shift_labels[nonpad_mask]
        log_probs = F.log_softmax(shift_logits_flat, dim=-1)
        if knn_prob is None:
            # Fast path: one-hot KNN distribution -> NLL at knn_label tokens
            if knn_label is not None:
                if knn_label.dim() > 1:
                    knn_label = knn_label[nonpad_mask]
                # Trim to matching length in case dstore_range is slightly off
                n = log_probs.size(0)
                if knn_label.size(0) > n:
                    knn_label = knn_label[:n]
                elif knn_label.size(0) < n:
                    # Pad with -100 to be ignored if KNN labels are short
                    pad = torch.full((n - knn_label.size(0),), -100, dtype=knn_label.dtype, device=knn_label.device)
                    knn_label = torch.cat([knn_label, pad])
                kl_loss = F.nll_loss(log_probs, knn_label, reduction='sum', ignore_index=-100)
                kl_loss = kl_loss / max(log_probs.size(0), 1)
            else:
                kl_loss = torch.tensor(0.0, device=log_probs.device)
        else:
            # Dense path for multi-token KNN distributions
            if len(knn_prob.shape) == 3:
                # (B, T, V) -> align with shift_logits (B, T-1, V) by dropping the last row
                knn_prob = knn_prob[..., :-1, :]
                knn_prob = knn_prob[nonpad_mask]
            elif len(knn_prob.shape) == 2:
                # (N, V) flattened; trim/pad to match log_probs length
                n = log_probs.size(0)
                if knn_prob.size(0) > n:
                    knn_prob = knn_prob[:n]
                elif knn_prob.size(0) < n:
                    pad = torch.zeros(n - knn_prob.size(0), knn_prob.size(-1), device=knn_prob.device, dtype=knn_prob.dtype)
                    knn_prob = torch.cat([knn_prob, pad], dim=0)
            knn_prob = knn_prob + 1e-10
            knn_prob = knn_prob / (knn_prob.sum(dim=-1, keepdim=True) + 1e-8)
            kl_loss = torch.tensor(0.0, device=shift_logits.device)
            for i in range(0, log_probs.size(0), KL_LOSS_CHUNK_SIZE):
                chunk_log_probs = log_probs[i:i+KL_LOSS_CHUNK_SIZE]
                chunk_knn_prob = knn_prob[i:i+KL_LOSS_CHUNK_SIZE]
                if chunk_log_probs.size(0) > 8:
                    chunk_log_probs = chunk_log_probs.cpu()
                    chunk_knn_prob = chunk_knn_prob.cpu()
                chunk_kl = F.kl_div(
                    chunk_log_probs,
                    chunk_knn_prob,
                    reduction='sum',
                    log_target=False
                )
                if chunk_kl.device != kl_loss.device:
                    chunk_kl = chunk_kl.to(kl_loss.device)
                kl_loss += chunk_kl
                del chunk_log_probs, chunk_knn_prob, chunk_kl
            kl_loss = kl_loss / max(log_probs.size(0), 1)

        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        total_loss = alpha * kl_loss + (1-alpha) * lm_loss
        return total_loss, kl_loss.detach(), lm_loss.detach()

    except Exception as e:
        logger.error(f"Error in kl_loss_token: {str(e)}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise