"""Backbone loading, bidirectional attention and residual-stream capture.

`bidirectional_mask` is the single definition of SPLARE-style bidirectional
attention for this project. The (future) M1/M2 training code must import it
from here, so that analysis and training see exactly the same model.
"""
import torch
from transformers import AutoModel, AutoTokenizer


def load_backbone(name, adapters=(), layer=None, dtype=torch.bfloat16,
                  device="cuda", attn_impl="eager"):
    """Load a decoder backbone (no LM head), optionally merging LoRA adapters in order.

    adapters: PEFT adapter dirs trained on the same AutoModel class
              (e.g. [mntp_adapter] for M1, [mntp_adapter, retrieval_adapter] for M2).
    layer:    if given, drop all blocks after `layer` (we only read its output).
    """
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "right"
    # eager attention: Gemma-2 uses attention logit softcapping.
    model = AutoModel.from_pretrained(name, torch_dtype=dtype, attn_implementation=attn_impl)
    for a in adapters:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a).merge_and_unload()
    if layer is not None:
        n = len(model.layers)
        if not 0 <= layer < n:
            raise ValueError(f"layer {layer} out of range for {n} blocks")
        model.layers = torch.nn.ModuleList(list(model.layers)[: layer + 1])
    model.to(device).eval()
    return model, tok


def bidirectional_mask(attention_mask, dtype):
    """(B, T) padding mask -> (B, 1, T, T) additive mask: every query sees every non-pad key."""
    B, T = attention_mask.shape
    keep = attention_mask[:, None, None, :].bool().expand(B, 1, T, T)
    mask = torch.zeros(B, 1, T, T, dtype=dtype, device=attention_mask.device)
    return mask.masked_fill(~keep, torch.finfo(dtype).min)


class ResidCapture:
    """Forward hook that stores the output of block `layer` (= resid_post)."""

    def __init__(self, model, layer):
        self.out = None
        self.handle = model.layers[layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.out = output[0] if isinstance(output, tuple) else output

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def forward_resid(model, capture, input_ids, attention_mask, bidirectional):
    if bidirectional:
        mask = bidirectional_mask(attention_mask, model.dtype)
    else:
        mask = attention_mask
    capture.out = None
    model(input_ids=input_ids, attention_mask=mask, use_cache=False)
    return capture.out


@torch.no_grad()
def check_attention_mode(model, capture, tok, bidirectional, device="cuda"):
    """Fail loudly if the attention mode is not what we think it is.

    Perturbs the last token: under causal attention earlier positions must not
    change; under bidirectional attention they must. This guards against
    transformers versions silently ignoring the 4D mask.
    """
    ids = tok("The quick brown fox jumps over the lazy dog", return_tensors="pt").input_ids.to(device)
    ids2 = ids.clone()
    ids2[0, -1] = tok.convert_tokens_to_ids(tok.tokenize(" cat"))[0]
    am = torch.ones_like(ids)
    h1 = forward_resid(model, capture, ids, am, bidirectional)[0, 1:-1].float()
    h2 = forward_resid(model, capture, ids2, am, bidirectional)[0, 1:-1].float()
    diff = (h1 - h2).abs().max().item()
    if bidirectional and diff < 1e-3:
        raise RuntimeError(f"Bidirectional mask has no effect (max diff {diff:.2e}); "
                           "this transformers version probably rebuilds a causal mask.")
    if not bidirectional and diff > 1e-2:
        raise RuntimeError(f"Causal model leaks future tokens (max diff {diff:.2e}).")
    return diff
