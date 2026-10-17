"""E1: does the frozen SAE still fit the residual stream of a (fine-tuned) backbone?

For one model (M0 base, M1 bidirectional+MNTP, M2 SPLARE-trained, ...) this
script runs the fixed corpus through the backbone up to the SAE layer, encodes
resid_post with the frozen SAE and accumulates, per (source, half):

  * reconstruction: FVU, cosine(x, x_hat)
  * sparsity: L0 per token (mean + histogram)
  * activation norms (mean/std + histogram)
  * per-feature firing counts / activation sums / max (dead + dense features, and E2 input)

Two "views" are computed:
  raw     - activations as produced by the model
  scaled  - activations rescaled by a global factor so that their mean norm matches the
            reference run (--norm-ref). This separates "the norm changed" from
            "the direction changed" (only computed when --norm-ref is given).

BOS and padding positions are excluded (Gemma Scope was not trained on BOS).

Examples:
  # M0: base model, causal attention (the SAE's native setting)
  python e1_sae_fit.py --corpus data/e1_corpus --out runs/e1/m0 --label M0

  # M0-bidir: base weights, bidirectional attention, no training
  python e1_sae_fit.py --corpus data/e1_corpus --out runs/e1/m0_bidir --label M0-bidir \
      --bidirectional --norm-ref runs/e1/m0/summary.json

  # M2: SPLARE-trained (MNTP adapter, then retrieval adapter, merged in order)
  python e1_sae_fit.py --corpus data/e1_corpus --out runs/e1/m2 --label M2 --bidirectional \
      --adapter ckpt/mntp --adapter ckpt/splare --norm-ref runs/e1/m0/summary.json
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from tqdm import tqdm

from drift.corpus import Corpus
from drift.modeling import ResidCapture, check_attention_mode, forward_resid, load_backbone
from drift.sae import load_sae

L0_BINS = 2049  # 0..2047, last bin = overflow
NORM_EDGES = np.logspace(-1, 4, 201)  # log-spaced bins for ||x||


class Stats:
    """Streaming accumulators over G = n_sources * 2 groups (source, half)."""

    def __init__(self, n_groups, width, d_model, device):
        G, W = n_groups, width
        f64 = dict(dtype=torch.float64, device=device)
        self.G, self.W = G, W
        self.n = torch.zeros(G, **f64)
        self.sum_x = torch.zeros(G, d_model, **f64)
        self.sum_sq = torch.zeros(G, **f64)
        self.err2 = torch.zeros(G, **f64)
        self.cos = torch.zeros(G, **f64)
        self.norm = torch.zeros(G, **f64)
        self.norm_sq = torch.zeros(G, **f64)
        self.l0 = torch.zeros(G, **f64)
        self.l0_hist = torch.zeros(G * L0_BINS, dtype=torch.long, device=device)
        self.norm_hist = torch.zeros(G * (len(NORM_EDGES) + 1), dtype=torch.long, device=device)
        self.fire = torch.zeros(G * W, dtype=torch.long, device=device)
        self.act_sum = torch.zeros(G * W, **f64)
        self.act_max = torch.zeros(W, dtype=torch.float32, device=device)
        self.norm_edges = torch.tensor(NORM_EDGES, dtype=torch.float32, device=device)

    @torch.no_grad()
    def update(self, x, g, sae):
        """x: (N, d) float32 activations, g: (N,) group ids."""
        z = sae.encode(x)
        x_hat = sae.decode(z)
        G, W = self.G, self.W

        norm = x.norm(dim=-1)
        l0 = (z > 0).sum(-1)
        add = lambda acc, v: acc.index_add_(0, g, v.double())
        add(self.n, torch.ones_like(norm))
        self.sum_x.index_add_(0, g, x.double())
        add(self.sum_sq, norm ** 2)
        add(self.err2, (x - x_hat).pow(2).sum(-1))
        add(self.cos, torch.nn.functional.cosine_similarity(x, x_hat, dim=-1))
        add(self.norm, norm)
        add(self.norm_sq, norm ** 2)
        add(self.l0, l0.float())

        self.l0_hist += torch.bincount(g * L0_BINS + l0.clamp(max=L0_BINS - 1),
                                       minlength=G * L0_BINS)
        nb = torch.bucketize(norm, self.norm_edges)  # 0..len(edges)
        self.norm_hist += torch.bincount(g * (len(NORM_EDGES) + 1) + nb,
                                         minlength=G * (len(NORM_EDGES) + 1))

        rows, cols = z.nonzero(as_tuple=True)
        vals = z[rows, cols]
        flat = g[rows] * W + cols
        self.fire += torch.bincount(flat, minlength=G * W)
        self.act_sum += torch.bincount(flat, weights=vals.double(), minlength=G * W)
        self.act_max.scatter_reduce_(0, cols, vals, reduce="amax")

    def to_numpy(self, n_sources):
        S, W = n_sources, self.W
        return {
            "n": self.n.cpu().numpy().reshape(S, 2),
            "sum_x": self.sum_x.cpu().numpy().reshape(S, 2, -1),
            "sum_sq": self.sum_sq.cpu().numpy().reshape(S, 2),
            "err2": self.err2.cpu().numpy().reshape(S, 2),
            "cos": self.cos.cpu().numpy().reshape(S, 2),
            "norm": self.norm.cpu().numpy().reshape(S, 2),
            "norm_sq": self.norm_sq.cpu().numpy().reshape(S, 2),
            "l0": self.l0.cpu().numpy().reshape(S, 2),
            "l0_hist": self.l0_hist.cpu().numpy().reshape(S, 2, L0_BINS),
            "norm_hist": self.norm_hist.cpu().numpy().reshape(S, 2, -1),
            "fire": self.fire.cpu().numpy().reshape(S, 2, W),
            "act_sum": self.act_sum.cpu().numpy().reshape(S, 2, W).astype(np.float32),
            "act_max": self.act_max.cpu().numpy(),
        }


def hist_quantile(hist, q):
    c = np.cumsum(hist)
    return int(np.searchsorted(c, q * c[-1])) if c[-1] > 0 else float("nan")


def summarize(a, sel, dense_thresh):
    """Aggregate the numpy stats `a` over a boolean (S, 2) group selection."""
    n = a["n"][sel].sum()
    if n == 0:
        return None
    sum_x = a["sum_x"][sel].sum(0)
    total_var = a["sum_sq"][sel].sum() - (sum_x @ sum_x) / n
    mean_norm = a["norm"][sel].sum() / n
    l0_hist = a["l0_hist"][sel].sum(0)
    fire = a["fire"][sel].sum(0)
    freq = fire / n
    return {
        "n_tokens": int(n),
        "fvu": float(a["err2"][sel].sum() / total_var),
        "mse_per_token": float(a["err2"][sel].sum() / n),
        "mean_cos": float(a["cos"][sel].sum() / n),
        "mean_l0": float(a["l0"][sel].sum() / n),
        "median_l0": hist_quantile(l0_hist, 0.5),
        "p95_l0": hist_quantile(l0_hist, 0.95),
        "frac_l0_overflow": float(l0_hist[-1] / l0_hist.sum()),
        "mean_norm": float(mean_norm),
        "std_norm": float(np.sqrt(max(a["norm_sq"][sel].sum() / n - mean_norm ** 2, 0.0))),
        "frac_dead": float((fire == 0).mean()),
        "frac_dense": float((freq > dense_thresh).mean()),
        "n_alive": int((fire > 0).sum()),
    }


def summarize_all(a, sources, dense_thresh):
    S = len(sources)
    out = {}
    sel = np.ones((S, 2), bool)
    out["all"] = summarize(a, sel, dense_thresh)
    for h in range(2):
        sel = np.zeros((S, 2), bool)
        sel[:, h] = True
        out[f"half{h}"] = summarize(a, sel, dense_thresh)
    for s, name in enumerate(sources):
        sel = np.zeros((S, 2), bool)
        sel[s] = True
        out[name] = summarize(a, sel, dense_thresh)
    return out


@torch.no_grad()
def valid_activations(model, capture, ids, attn, src, half, bidirectional, bos_id, device):
    ids, attn = ids.to(device), attn.to(device)
    x = forward_resid(model, capture, ids, attn, bidirectional)
    valid = attn.bool() & (ids != bos_id)
    g = (src.to(device) * 2 + half.to(device))[:, None].expand_as(ids)
    return x[valid].float(), g[valid]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--adapter", action="append", default=[], help="PEFT adapter dir; repeat to merge in order")
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--sae-kind", default="gemma_scope")
    ap.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    ap.add_argument("--sae-width", default="16k")
    ap.add_argument("--sae-l0-dir", default="average_l0_82",
                    help="Gemma Scope variant; SPLARE uses the one with L0 closest to 100 (82 for layer 12/16k)")
    ap.add_argument("--norm-ref", default=None, help="summary.json of the reference run (M0)")
    ap.add_argument("--calib-docs", type=int, default=512)
    ap.add_argument("--max-batch-tokens", type=int, default=16384)
    ap.add_argument("--sae-chunk", type=int, default=4096, help="tokens per SAE call (memory: chunk x width x 4B)")
    ap.add_argument("--dense-thresh", type=float, default=0.1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    corpus = Corpus(args.corpus)
    if corpus.meta["tokenizer"] != args.model:
        print(f"NOTE: corpus tokenised with {corpus.meta['tokenizer']}, model is {args.model}")
    sources = corpus.sources
    bos_id, pad_id = corpus.meta["bos_token_id"], corpus.meta["pad_token_id"]

    model, tok = load_backbone(args.model, args.adapter, layer=args.layer,
                               dtype=getattr(torch, args.dtype), device=device)
    if tok.bos_token_id != bos_id:
        raise ValueError("Tokenizer mismatch between corpus and model.")
    capture = ResidCapture(model, args.layer)
    attn_check_diff = check_attention_mode(model, capture, tok, args.bidirectional, device)
    print(f"attention mode check ok (bidirectional={args.bidirectional}, max diff {attn_check_diff:.3g})")

    sae = load_sae(args.sae_kind, repo=args.sae_repo, layer=args.layer, width=args.sae_width,
                   l0_dir=args.sae_l0_dir, device=device)
    if sae.d_model != model.config.hidden_size:
        raise ValueError(f"SAE d_model {sae.d_model} != model hidden size {model.config.hidden_size}")
    print(f"SAE: {sae.meta['repo']}/{sae.meta['dir']} (width {sae.width})")

    # Calibration: mean activation norm on a fixed doc subset (same subset for every run).
    rng = np.random.default_rng(args.seed)
    calib_ids = np.sort(rng.choice(len(corpus), min(args.calib_docs, len(corpus)), replace=False))
    norm_sum, norm_n = 0.0, 0
    for ids, attn, src, half in corpus.batches(args.max_batch_tokens, pad_id, calib_ids):
        x, _ = valid_activations(model, capture, ids, attn, src, half, args.bidirectional, bos_id, device)
        norm_sum += x.norm(dim=-1).double().sum().item()
        norm_n += len(x)
    calib_mean_norm = norm_sum / norm_n

    views = {"raw": 1.0}
    if args.norm_ref:
        ref = json.loads(Path(args.norm_ref).read_text())
        views["scaled"] = ref["calib_mean_norm"] / calib_mean_norm
    print(f"calib mean norm {calib_mean_norm:.2f}; views: {views}")

    G = len(sources) * 2
    stats = {v: Stats(G, sae.width, sae.d_model, device) for v in views}
    n_batches = sum(1 for _ in corpus.batches(args.max_batch_tokens, pad_id))
    for ids, attn, src, half in tqdm(corpus.batches(args.max_batch_tokens, pad_id), total=n_batches):
        x, g = valid_activations(model, capture, ids, attn, src, half, args.bidirectional, bos_id, device)
        for v, scale in views.items():
            for i in range(0, len(x), args.sae_chunk):
                stats[v].update(x[i: i + args.sae_chunk] * scale, g[i: i + args.sae_chunk], sae)

    arrays = {v: s.to_numpy(len(sources)) for v, s in stats.items()}
    summary = {
        "label": args.label or out.name,
        "args": vars(args),
        "sae": sae.meta,
        "sources": sources,
        "calib_mean_norm": calib_mean_norm,
        "view_scales": views,
        "attn_check_max_diff": attn_check_diff,
        "env": {"torch": torch.__version__, "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(), "host": platform.node()},
        "runtime_sec": time.time() - t0,
        "views": {v: summarize_all(a, sources, args.dense_thresh) for v, a in arrays.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out / "feature_stats.npz", norm_edges=NORM_EDGES,
                        **{f"{v}__{k}": val for v, a in arrays.items() for k, val in a.items()})

    r = summary["views"]["raw"]["all"]
    print(f"[{summary['label']}] raw: FVU {r['fvu']:.4f} cos {r['mean_cos']:.4f} L0 {r['mean_l0']:.1f} "
          f"dead {r['frac_dead']:.3%} dense {r['frac_dense']:.3%} norm {r['mean_norm']:.1f}")
    if "scaled" in summary["views"]:
        s = summary["views"]["scaled"]["all"]
        print(f"[{summary['label']}] scaled: FVU {s['fvu']:.4f} cos {s['mean_cos']:.4f} L0 {s['mean_l0']:.1f}")
    if not args.adapter and not args.bidirectional and r["fvu"] > 0.3:
        print("WARNING: high FVU for the base causal model; check layer / SAE choice.")
    print(f"done in {summary['runtime_sec']/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
