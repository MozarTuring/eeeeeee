"""Sparse autoencoder loading (encoder + decoder) for drift analysis.

Only Gemma Scope (JumpReLU, residual stream) is implemented for the pilot.
Gemma Scope params are released with the activation normalisation folded in,
so they are applied to raw residual-stream activations.
"""
import numpy as np
import torch
from huggingface_hub import hf_hub_download


class JumpReLUSAE(torch.nn.Module):
    def __init__(self, W_enc, W_dec, b_enc, b_dec, threshold, meta=None):
        super().__init__()
        self.W_enc = torch.nn.Parameter(W_enc, requires_grad=False)  # (d, W)
        self.W_dec = torch.nn.Parameter(W_dec, requires_grad=False)  # (W, d)
        self.b_enc = torch.nn.Parameter(b_enc, requires_grad=False)  # (W,)
        self.b_dec = torch.nn.Parameter(b_dec, requires_grad=False)  # (d,)
        self.threshold = torch.nn.Parameter(threshold, requires_grad=False)  # (W,)
        self.meta = meta or {}

    @property
    def d_model(self):
        return self.W_enc.shape[0]

    @property
    def width(self):
        return self.W_enc.shape[1]

    def encode(self, x):
        pre = x @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec


def load_gemma_scope(repo="google/gemma-scope-2b-pt-res", layer=12, width="16k",
                     l0_dir="average_l0_82", device="cuda", dtype=torch.float32):
    """Load layer_{layer}/width_{width}/{l0_dir}/params.npz (must already be in the HF cache).

    SPLARE uses the released variant with average L0 closest to 100; for
    layer 12 / width 16k that is average_l0_82. Other layer/width combinations
    have different variants, so pass the matching l0_dir.
    """
    sae_dir = f"layer_{layer}/width_{width}/{l0_dir}"
    path = hf_hub_download(repo, f"{sae_dir}/params.npz")
    p = np.load(path)
    t = lambda k: torch.from_numpy(p[k]).to(device=device, dtype=dtype)
    sae = JumpReLUSAE(t("W_enc"), t("W_dec"), t("b_enc"), t("b_dec"), t("threshold"),
                      meta={"repo": repo, "dir": sae_dir, "layer": layer, "width": width})
    return sae


def load_sae(kind, **kw):
    if kind == "gemma_scope":
        return load_gemma_scope(**kw)
    if kind == "llama_scope":
        # TODO: Llama Scope (fnlp/Llama3_1-8B-Base-L*R-*x) for the 8B scale-up.
        # Needs its own loader: different checkpoint layout and input-norm handling.
        raise NotImplementedError("Llama Scope loader not implemented yet (8B scale-up).")
    raise ValueError(kind)
