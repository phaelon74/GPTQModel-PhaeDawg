"""
Runtime hot-patches that fix shape mismatches in HuggingFace Cohere2 models.

Two independent fixes are applied on import:
1. Safe `apply_rotary_pos_emb` (Solution A) – slices the pre-computed
   `cos` / `sin` tables so they always match the true `head_dim` coming from
   the query / key tensors.
2. Correct `_split_heads` reshape (Solution B) – restores the expected
   `[B, T, n_heads, head_dim]` layout required by RoPE.

The file is imported from `gptqmodel.__init__`, therefore the patches are in
place before any Cohere2 model is instantiated by the library or by user
code.
"""

from __future__ import annotations

import logging
from types import MethodType

log = logging.getLogger(__name__)

try:
    # Transformers ≥ 4.52 ships the Cohere2 implementation under this path.
    from transformers.models.cohere2 import modeling_cohere2 as _cohere2
except (ImportError, ModuleNotFoundError):
    _cohere2 = None  # pragma: no cover

# ---------------------------------------------------------------------------
# Solution A – robust `apply_rotary_pos_emb`
# ---------------------------------------------------------------------------
if _cohere2 is not None and hasattr(_cohere2, "apply_rotary_pos_emb"):
    if not getattr(_cohere2.apply_rotary_pos_emb, "_gptqmodel_patched", False):
        _orig_apply_rope = _cohere2.apply_rotary_pos_emb  # keep reference

        def _safe_apply_rope(q, k, cos, sin):  # type: ignore[override]
            """Apply RoPE regardless of `(seq_len, head_dim)` axis order.

            Cohere-2 swaps the two trailing axes compared to the reference
            implementation.  We detect the layout at run-time and, if
            necessary, transpose `[B, head_dim, seq_len] -> [B, seq_len, head_dim]`
            before applying the usual formula, then transpose the result back.
            """

            need_transpose = q.shape[-2] >  q.shape[-1]  # 128 vs 83 → True

            if need_transpose:
                q, k = q.transpose(-1, -2), k.transpose(-1, -2)

            seq_len, head_dim = q.shape[-2], q.shape[-1]

            cos = cos[:, :seq_len, :head_dim]
            sin = sin[:, :seq_len, :head_dim]

            q_out = (q * cos) + (_cohere2.rotate_half(q) * sin)
            k_out = (k * cos) + (_cohere2.rotate_half(k) * sin)

            if need_transpose:
                q_out, k_out = q_out.transpose(-1, -2), k_out.transpose(-1, -2)

            return q_out, k_out

        _safe_apply_rope._gptqmodel_patched = True  # type: ignore[attr-defined]
        _cohere2.apply_rotary_pos_emb = _safe_apply_rope  # type: ignore[assignment]
        log.debug("[GPTQModel] Patched Cohere2.apply_rotary_pos_emb (Solution A)")

# ---------------------------------------------------------------------------
# Solution B – fix `_split_heads` dimension order
# ---------------------------------------------------------------------------
if _cohere2 is not None and hasattr(_cohere2, "Cohere2Attention"):
    _attn_cls = _cohere2.Cohere2Attention
    if not hasattr(_attn_cls, "_gptqmodel_split_heads_patched"):

        def _split_heads_fixed(self, x, num_heads, head_dim):  # noqa: N802
            """Reshape to [B, T, num_heads, head_dim] (expected by RoPE)."""
            return x.view(x.size(0), x.size(1), num_heads, head_dim)

        # Bind method to class (no instance yet)
        _attn_cls._split_heads = _split_heads_fixed  # type: ignore[assignment]
        _attn_cls._gptqmodel_split_heads_patched = True  # type: ignore[attr-defined]
        log.debug("[GPTQModel] Patched Cohere2Attention._split_heads (Solution B)")

else:
    if _cohere2 is None:
        log.debug("[GPTQModel] transformers Cohere2 implementation not found – no patches applied") 