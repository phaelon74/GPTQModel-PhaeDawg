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
        """RoPE that works no matter which axis is sequence vs head_dim.

        We *do not* change the tensor layout, preserving Cohere-2’s MQA logic.
        Instead we adapt the cosine / sine tables and apply rotation along the
        correct axis.
        """

        # identify axes
        if q.shape[-1] % 2 == 0 and q.shape[-1] == cos.shape[-1]:
            head_axis, seq_axis = -1, -2  # standard layout (…, seq, dim)
        else:
            head_axis, seq_axis = -2, -1  # Cohere-2 layout  (…, dim, seq)

        seq_len = min(q.shape[seq_axis], cos.shape[-2])
        head_dim = min(q.shape[head_axis], cos.shape[-1]) & ~1  # even

        # slice q,k
        slicer_qk = [slice(None)] * q.dim()
        slicer_qk[seq_axis] = slice(0, seq_len)
        slicer_qk[head_axis] = slice(0, head_dim)
        q = q[tuple(slicer_qk)]
        k = k[tuple(slicer_qk)]

        # prepare cos/sin to broadcast with q layout
        cos = cos[:, :seq_len, :head_dim]
        sin = sin[:, :seq_len, :head_dim]
        if head_axis == -2:  # need head_dim before seq for broadcasting
            cos = cos.permute(0, 2, 1)
            sin = sin.permute(0, 2, 1)

        # rotate_half along head_axis
        def rotate_half_axis(x, axis):
            dim = x.shape[axis]
            x1, x2 = x.split(dim // 2, dim=axis)
            return torch.cat((-x2, x1), dim=axis)

        import torch

        q_out = (q * cos) + (rotate_half_axis(q, head_axis) * sin)
        k_out = (k * cos) + (rotate_half_axis(k, head_axis) * sin)
        return q_out, k_out

    _apply_rope_safe._gptqmodel_safe = True  # type: ignore[attr-defined]
    _c2.apply_rotary_pos_emb = _apply_rope_safe  # type: ignore[assignment]
    log.debug("[GPTQModel] Replaced Cohere2.apply_rotary_pos_emb with safe version") 