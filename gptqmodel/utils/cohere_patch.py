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
        """Layout-agnostic and broadcast-safe RoPE.

        1. Detect which axis in `q`/`k` matches the RoPE table's head_dim
           (=`cos.shape[-1]`).
        2. If that axis is -2 (Cohere-2 layout `…, dim, seq`) transpose the
           *positional tables* instead of the model tensors – keeps MQA shape.
        """

        import torch

        head_dim_rope = cos.shape[-1]

        if q.shape[-1] == head_dim_rope:  # normal layout  (…, seq, dim)
            seq_axis, head_axis = -2, -1
            cos_s, sin_s = cos, sin
        elif q.shape[-2] == head_dim_rope:  # Cohere-2 layout (…, dim, seq)
            seq_axis, head_axis = -1, -2
            cos_s, sin_s = cos.permute(0, 2, 1), sin.permute(0, 2, 1)
        else:  # fallback – assume normal
            seq_axis, head_axis = -2, -1
            cos_s, sin_s = cos, sin

        seq_len = min(q.shape[seq_axis], cos_s.shape[-2])
        head_dim = head_dim_rope & ~1  # even

        # slice tensors
        sl = [slice(None)] * q.dim()
        sl[seq_axis] = slice(0, seq_len)
        sl[head_axis] = slice(0, head_dim)
        q, k = q[tuple(sl)], k[tuple(sl)]
        cos_s, sin_s = cos_s[:, :seq_len, :head_dim], sin_s[:, :seq_len, :head_dim]

        # rotate half along head_axis
        def rhalf(x):
            x1, x2 = x.split(head_dim // 2, dim=head_axis)
            return torch.cat((-x2, x1), dim=head_axis)

        return (q * cos_s) + (rhalf(q) * sin_s), (k * cos_s) + (rhalf(k) * sin_s)

    _apply_rope_safe._gptqmodel_safe = True  # type: ignore[attr-defined]
    _c2.apply_rotary_pos_emb = _apply_rope_safe  # type: ignore[assignment]
    log.debug("[GPTQModel] Replaced Cohere2.apply_rotary_pos_emb with safe version") 