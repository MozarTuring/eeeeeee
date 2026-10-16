"""Compare E1 runs against a reference run (normally M0) and produce tables + plots.

Feature-frequency drift is reported with a noise floor: we compare
  ref half0 vs ref half1   (same model, disjoint docs)       -> noise floor
  run half1 vs ref half0   (different model, disjoint docs)  -> drift
so both comparisons use the same number of tokens and disjoint data.

Example:
  python e1_compare.py --ref runs/e1/m0 \
      --runs runs/e1/m0 runs/e1/m0_bidir runs/e1/m1 runs/e1/m2 --out reports/e1
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

GLOBAL_COLS = ["fvu", "mean_cos", "mean_l0", "median_l0", "p95_l0", "mean_norm",
               "frac_dead", "frac_dense"]


def load_run(path):
    path = Path(path)
    summary = json.loads((path / "summary.json").read_text())
    z = np.load(path / "feature_stats.npz")
    arrays = {}
    for k in z.files:
        if "__" in k:
            view, name = k.split("__", 1)
            arrays.setdefault(view, {})[name] = z[k]
    return {"summary": summary, "arrays": arrays, "norm_edges": z["norm_edges"]}


def freq(a, half=None, source=None):
    fire, n = a["fire"], a["n"]
    if source is not None:
        fire, n = fire[source: source + 1], n[source: source + 1]
    if half is not None:
        fire, n = fire[:, half], n[:, half]
    return fire.reshape(-1, fire.shape[-1]).sum(0) / max(n.sum(), 1)


def freq_drift(f_a, f_b, n_a_tokens, min_fires):
    """Drift of per-feature firing frequency of b relative to a, over features alive in a."""
    alive = f_a * n_a_tokens >= min_fires
    if alive.sum() < 2:
        return None
    eps = 1.0 / max(n_a_tokens, 1)
    lr = np.log10((f_b[alive] + eps) / (f_a[alive] + eps))
    return {
        "n_ref_alive": int(alive.sum()),
        "spearman": float(spearmanr(f_a[alive], f_b[alive]).statistic),
        "frac_10x_change": float((np.abs(lr) > 1).mean()),
        "frac_newly_dead": float((f_b[alive] == 0).mean()),
        "median_abs_log10_ratio": float(np.median(np.abs(lr))),
    }


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-fires", type=int, default=20,
                    help="feature counts as alive in the reference half if it fires >= this many times")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref = load_run(args.ref)
    runs = [load_run(p) for p in args.runs]
    sources = ref["summary"]["sources"]
    for r in runs:
        if r["summary"]["sources"] != sources:
            raise ValueError("Runs were computed on different corpora.")
        if r["summary"]["sae"] != ref["summary"]["sae"]:
            raise ValueError("Runs use different SAEs.")
    labels = [r["summary"]["label"] for r in runs]
    report = {"ref": ref["summary"]["label"], "runs": labels, "sae": ref["summary"]["sae"]}
    md = [f"# E1: SAE fit — ref = {report['ref']}",
          f"SAE: `{ref['summary']['sae']['repo']}/{ref['summary']['sae']['dir']}`"]

    # 1. Global reconstruction / sparsity table.
    rows, report["global"] = [], {}
    for r, lab in zip(runs, labels):
        for view, s in r["summary"]["views"].items():
            rows.append([lab, view] + [s["all"][c] for c in GLOBAL_COLS])
            report["global"][f"{lab}/{view}"] = s["all"]
    md += ["\n## Global (all sources)", md_table(["run", "view"] + GLOBAL_COLS, rows)]

    # 2. Per-source FVU (raw and, when present, scaled).
    for view in ["raw", "scaled"]:
        present = [(r, lab) for r, lab in zip(runs, labels) if view in r["summary"]["views"]]
        if not present:
            continue
        rows = [[src] + [r["summary"]["views"][view][src]["fvu"] for r, _ in present] for src in sources]
        md += [f"\n## FVU per source ({view})", md_table(["source"] + [l for _, l in present], rows)]

    # 3. Feature-frequency drift vs noise floor (raw view).
    ra = ref["arrays"]["raw"]
    n_ref_h0 = ra["n"][:, 0].sum()
    f_ref0, f_ref1 = freq(ra, 0), freq(ra, 1)
    noise = freq_drift(f_ref0, f_ref1, n_ref_h0, args.min_fires)
    report["freq_drift"] = {"noise_floor(ref h0 vs h1)": noise}
    drift_cols = ["spearman", "frac_10x_change", "frac_newly_dead", "median_abs_log10_ratio"]
    rows = [["noise floor (ref h0 vs ref h1)"] + [noise[c] for c in drift_cols]]
    for r, lab in zip(runs, labels):
        # Note: the reference run itself appears here too; its row equals the noise floor.
        d = freq_drift(f_ref0, freq(r["arrays"]["raw"], 1), n_ref_h0, args.min_fires)
        report["freq_drift"][lab] = d
        rows.append([f"{lab} h1 vs ref h0"] + [d[c] for c in drift_cols])
    md += ["\n## Feature firing-frequency drift (raw view)",
           f"Features alive in ref half0 (>= {args.min_fires} fires): {noise['n_ref_alive']}",
           md_table(["comparison"] + drift_cols, rows)]

    # Per-source Spearman of feature frequencies (drift vs per-source noise floor).
    rows, report["freq_spearman_per_source"] = [], {}
    for s, src in enumerate(sources):
        n0 = ra["n"][s, 0]
        nf = freq_drift(freq(ra, 0, s), freq(ra, 1, s), n0, args.min_fires)
        row = [src, nf["spearman"] if nf else float("nan")]
        for r, lab in zip(runs, labels):
            d = freq_drift(freq(ra, 0, s), freq(r["arrays"]["raw"], 1, s), n0, args.min_fires)
            row.append(d["spearman"] if d else float("nan"))
        rows.append(row)
        report["freq_spearman_per_source"][src] = dict(zip(["noise"] + labels, row[1:]))
    md += ["\n## Per-source Spearman of feature frequency (ref h0 vs run h1)",
           md_table(["source", "noise"] + labels, rows)]

    (out / "e1_report.json").write_text(json.dumps(report, indent=2))
    (out / "e1_report.md").write_text("\n\n".join(md) + "\n")

    # 4. Plots.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for r, lab in zip(runs, labels):
        h = r["arrays"]["raw"]["l0_hist"].sum((0, 1))
        axes[0].plot(np.arange(len(h) - 1), h[:-1] / h.sum(), label=lab)
        nh = r["arrays"]["raw"]["norm_hist"].sum((0, 1))
        edges = r["norm_edges"]
        axes[1].plot(edges, nh[1:] / nh.sum(), label=lab)
    axes[0].set(xlabel="L0 per token", ylabel="fraction of tokens", xscale="log", title="L0 distribution (raw)")
    axes[1].set(xlabel="||x|| (resid_post)", ylabel="fraction of tokens", xscale="log", title="Activation norm")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "l0_norm_hist.png", dpi=150)

    n_plots = len(runs) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), sharex=True, sharey=True)
    eps = 1.0 / n_ref_h0
    pairs = [("noise floor", f_ref1)] + [(lab, freq(r["arrays"]["raw"], 1)) for r, lab in zip(runs, labels)]
    for ax, (lab, f) in zip(axes, pairs):
        ax.scatter(f_ref0 + eps, f + eps, s=1, alpha=0.3)
        ax.plot([eps, 1], [eps, 1], "k--", lw=0.8)
        ax.set(xscale="log", yscale="log", title=lab, xlabel="freq in ref (half0)")
    axes[0].set_ylabel("freq (half1)")
    fig.tight_layout()
    fig.savefig(out / "freq_scatter.png", dpi=150)

    fig, ax = plt.subplots(figsize=(max(6, len(sources) * 0.6), 4))
    w = 0.8 / len(runs)
    for i, (r, lab) in enumerate(zip(runs, labels)):
        ax.bar(np.arange(len(sources)) + i * w, [r["summary"]["views"]["raw"][s]["fvu"] for s in sources],
               w, label=lab)
    ax.set_xticks(np.arange(len(sources)) + 0.4 - w / 2, sources, rotation=60, ha="right")
    ax.set(ylabel="FVU (raw)", title="Reconstruction error per source")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fvu_per_source.png", dpi=150)

    print((out / "e1_report.md").read_text())


if __name__ == "__main__":
    main()
