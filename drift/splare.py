"""SPLARE encoder pieces: frozen SAE head on resid_post of a layer + SPLADE pooling.

rep_j = max_t log(1 + z_tj),  z_t = SAE.encode(x_t)  (JumpReLU output, already >= 0)
over valid token positions t (non-pad, non-BOS). Scores are dot products of reps.
"""
import torch

from drift.modeling import bidirectional_mask


def encode(net, capture, sae, input_ids, attn, bos_id):
    """Sparse representation (B, W) for a batch. `net` is the (LoRA/DDP-wrapped) truncated backbone,
    `capture` a ResidCapture on the SAE layer. Gradients flow through the SAE into the backbone."""
    capture.out = None
    net(input_ids=input_ids, attention_mask=bidirectional_mask(attn, torch.bfloat16), use_cache=False)
    x = capture.out.float()
    with torch.autocast("cuda", enabled=False):  # SAE in fp32, as in the E1 analysis
        z = sae.encode(x)
    valid = (attn.bool() & (input_ids != bos_id)).unsqueeze(-1)
    return (torch.log1p(z) * valid).max(dim=1).values


def topk_sparsify(rep, k):
    """Keep the k largest entries per row (inference-time Top-K pooling); k=None or 0 keeps all."""
    if not k or k >= rep.shape[-1]:
        return rep
    v, i = rep.topk(k, dim=-1)
    return torch.zeros_like(rep).scatter_(-1, i, v)


def flops_reg(rep):
    """FLOPS regulariser (Paria et al. 2020): sum_j (mean over batch of rep_j)^2."""
    return rep.mean(0).pow(2).sum()


def l0(rep):
    """Mean number of non-zero dimensions per representation."""
    return (rep > 0).float().sum(-1).mean()
