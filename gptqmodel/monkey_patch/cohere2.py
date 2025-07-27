
import torch
from transformers.models.cohere2.modeling_cohere2 import Cohere2Attention
import inspect

_original_cohere2_attn_forward = Cohere2Attention.forward

def cohere2_attn_forward_patch(self, *args, **kwargs):
    # a HACK, but it works
    # when quantizing, the position_ids is not passed to the attention forward pass
    # but is required by the rotary embedding layer.
    # we grab it from the layer above and add it to the attention forward pass.
    # see: https://github.com/huggingface/transformers/blob/v4.41.2/src/transformers/models/cohere/modeling_cohere.py#L353
    if "position_ids" not in kwargs or kwargs["position_ids"] is None:
        # get position_ids from the call stack
        for frame_info in inspect.stack():
            if "position_ids" in frame_info.frame.f_locals:
                position_ids = frame_info.frame.f_locals["position_ids"]
                if isinstance(position_ids, torch.Tensor):
                    kwargs["position_ids"] = position_ids
                    break

    if "position_ids" in kwargs and kwargs["position_ids"] is not None:
        kwargs["position_ids"] = kwargs["position_ids"].to(self.q_proj.weight.device)
    return _original_cohere2_attn_forward(self, *args, **kwargs)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1) 