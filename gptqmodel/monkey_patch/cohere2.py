
import torch
from transformers.models.cohere2.modeling_cohere2 import Cohere2Attention

def cohere2_attn_forward_patch(self, *args, **kwargs):
    kwargs["position_ids"] = kwargs["position_ids"].to(self.q_proj.weight.device)
    return Cohere2Attention.forward(self, *args, **kwargs)

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