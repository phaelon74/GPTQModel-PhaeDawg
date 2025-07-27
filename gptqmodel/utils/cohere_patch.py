"""
Runtime hot-patch for HuggingFace Cohere-2 models.

• Leaves Cohere-2′s original multi-query-attention reshape untouched.
• Replaces `apply_rotary_pos_emb` with a layout-agnostic, size-safe version
  that works for both `[B⋅H, dim, seq]` and `[B, seq, H, dim]` layouts.

The patch is idempotent and applied on import from `gptqmodel.__init__`.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    from transformers.models.cohere2 import modeling_cohere2 as _c2
except (ImportError, ModuleNotFoundError):
    _c2 = None  # pragma: no cover

if _c2 is not None and not getattr(_c2.apply_rotary_pos_emb, "_gptqmodel_safe", False):

    def _apply_rope_safe(q, k, cos, sin):  # type: ignore[override]
        """Safe RoPE that auto-detects axis order and trims tables.

        1. Detect Cohere-2 layout (`…, dim, seq`) vs ref layout (`…, seq, dim`).
        2. Transpose to `(…, seq, dim)` if needed.
        3. Trim `cos`/`sin` and `q`/`k` to common `(seq_len, head_dim)`
           (rotate_half requires even head_dim → round down if odd).
        4. Apply standard RoPE, transpose back if we transposed in step-1.
        """

        transposed = False
        if q.shape[-2] > q.shape[-1]:  # dim > seq  → Cohere-2 “swapped” layout
            q, k = q.transpose(-1, -2), k.transpose(-1, -2)
            transposed = True

        seq_len, head_dim = q.shape[-2], q.shape[-1]
        seq_len = min(seq_len, cos.shape[-2])
        head_dim = min(head_dim, cos.shape[-1]) & ~1  # make even for rotate_half

        # slice everything to the agreed size
        cos = cos[:, :seq_len, :head_dim]
        sin = sin[:, :seq_len, :head_dim]
        q = q[:, :seq_len, :head_dim]
        k = k[:, :seq_len, :head_dim]

        q_out = (q * cos) + (_c2.rotate_half(q) * sin)
        k_out = (k * cos) + (_c2.rotate_half(k) * sin)

        if transposed:
            q_out, k_out = q_out.transpose(-1, -2), k_out.transpose(-1, -2)
        return q_out, k_out

    _apply_rope_safe._gptqmodel_safe = True  # type: ignore[attr-defined]
    _c2.apply_rotary_pos_emb = _apply_rope_safe  # type: ignore[assignment]
    log.debug("[GPTQModel] Replaced Cohere2.apply_rotary_pos_emb with safe version") 