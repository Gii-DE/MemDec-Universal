# ----------------------------------------------------------------------
# MODIFICATION NOTICE
# ----------------------------------------------------------------------
# Modified 2026 by Gii-DE for the MemDec-Universal project.
# Changes include:
# - Code Refactoring: Removed unused imports and refined overall code layout/formatting
# - Performance & Memory Optimization: Cached log-lambda scalars as buffers to avoid per-forward tensor creation
# - Model Architecture Support / API Compatibility: Updated generate() to accept GenerationConfig and kwargs
# - Generation Behavior: Added dynamic extraction of eos_token_id, max_new_tokens, repetition_penalty
# - Generation Behavior: Replaced StoppingCriteriaList with EOS token detection for early stopping
# - Generation Behavior: Integrated repetition penalty before and during the generation loop
# - Generation Behavior: Added _apply_repetition_penalty() static helper
# - API Compatibility: __init__ deep-copies base_lm.config and pins the wrapper's attn implementation to 'eager' so transformers' SDPA dispatch check no longer rejects the wrapper when submodels are loaded with attn_implementation='sdpa'
#
# The original code is available at: 
# https://github.com/LUMIA-Group/MemoryDecoder
# ----------------------------------------------------------------------

from typing import Optional, Tuple
import copy
import torch, math
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import (
    GenerationMixin,
    PreTrainedModel,
    GenerationConfig,
)
from transformers.utils import ModelOutput


@dataclass
class MemoryDecoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    knn_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class MemoryDecoder(PreTrainedModel, GenerationMixin):
    """
    A light wrapper around **two** causal‑LMs that fuses their logits:

        logits_joint = logaddexp(logits_base + log(1‑λ),
                                 logits_knn  + log(λ))

    Greedy decoding chooses argmax over `logits_joint`.
    """
    def __init__(
        self,
        base_lm,
        knn_generator,
        lmbda: float = 0.25,
        knn_temp: float = 1.0,
    ):
        config = copy.deepcopy(base_lm.config)
        config._attn_implementation = "eager"
        super().__init__(config)
        self.base_lm = base_lm
        self.knn_generator = knn_generator
        self.lmbda = float(lmbda)
        self.knn_temp = float(knn_temp)
        self.register_buffer('_log_lambda', torch.log(torch.tensor(self.lmbda)))
        self.register_buffer('_log_one_minus_lambda', torch.log(torch.tensor(1.0 - self.lmbda)))
        
    # ------------------------------------------------------------------ #
    #                       1. forward()
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        knn_past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        """
        Forward pass that returns **fused log‑probs** as logits.
        We keep separate caches for each sub‑model.
        """
        base_outputs = self.base_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        knn_outputs = self.knn_generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=knn_past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        # Temperature on k-NN logits only
        logits_base = base_outputs.logits      # (B, T, V)
        logits_knn  = knn_outputs.logits
        if self.knn_temp != 1.0:
            logits_knn = logits_knn / self.knn_temp
        # Convert to log-probabilities first (numerically stable when fusing)
        logp_base = F.log_softmax(logits_base, dim=-1)
        logp_knn  = F.log_softmax(logits_knn,  dim=-1)
        logp_joint = torch.logaddexp(
            logp_base + self._log_one_minus_lambda.to(logp_base.device),
            logp_knn  + self._log_lambda.to(logp_base.device),
        )
        return MemoryDecoderOutput(
            logits=logp_joint,
            past_key_values=base_outputs.past_key_values,
            knn_past_key_values=knn_outputs.past_key_values,
            hidden_states=None,
            attentions=None
        )
        
    # ------------------------------------------------------------------ #
    #                       2. generate()
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(  # type: ignore[override]
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        do_sample = getattr(generation_config, 'do_sample', kwargs.get('do_sample', False)) if generation_config is not None else kwargs.get('do_sample', False)
        if do_sample:
            raise ValueError("MemoryDecoder.generate only supports greedy decoding (do_sample=False).")
        _gc = generation_config or GenerationConfig()
        eos_token_id = (
            kwargs.get("eos_token_id")
            or getattr(_gc, "eos_token_id", None)
            or self.base_lm.config.eos_token_id
        )
        max_new_tokens = (
            kwargs.get("max_new_tokens")
            or getattr(_gc, "max_new_tokens", None)
            or 100
        )
        repetition_penalty = (
            kwargs.get("repetition_penalty")
            or getattr(_gc, "repetition_penalty", None)
            or 1.0
        )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # Initialise caches with a single forward.
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True
        )
        next_token_logits = outputs["logits"][:, -1, :]   # (B, V)
        base_past = outputs["past_key_values"]
        knn_past  = outputs["knn_past_key_values"]
        if repetition_penalty != 1.0:
            next_token_logits = self._apply_repetition_penalty(
                next_token_logits, input_ids, repetition_penalty
            )
        # Greedy select
        next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # (B, 1)
        generated   = torch.cat([input_ids, next_tokens], dim=-1)            # (B, T+1)
        current_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)], dim=-1)
        
        # --- main loop -------------------------------------------------- #
        num_new_tokens = 1
        while num_new_tokens < max_new_tokens:
            finished_seq = (generated == eos_token_id).any(dim=-1)
            if finished_seq.all():
                break
            outputs = self.forward(
                input_ids=next_tokens,
                attention_mask=current_mask,
                past_key_values=base_past,
                knn_past_key_values=knn_past,
                use_cache=True
            )
            next_token_logits = outputs["logits"][:, -1, :]
            base_past = outputs["past_key_values"]
            knn_past  = outputs["knn_past_key_values"]
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated, repetition_penalty
                )
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated   = torch.cat([generated, next_tokens], dim=-1)
            current_mask = torch.cat([current_mask, torch.ones((current_mask.shape[0], 1), device=current_mask.device)], dim=-1)
            num_new_tokens += 1
        return generated

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.FloatTensor,
        input_ids: torch.LongTensor,
        penalty: float,
    ) -> torch.FloatTensor:
        if penalty == 1.0:
            return logits
        log_penalty = math.log(penalty)
        score = logits.clone()
        for b in range(input_ids.shape[0]):
            unique_tokens = input_ids[b].unique()
            score[b, unique_tokens] -= log_penalty
        return score