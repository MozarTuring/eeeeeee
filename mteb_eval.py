"""MTEB(eng, v2) retrieval evaluation of a SPLARE model with the `mteb` package.

The model object gives MTEB Top-K-sparsified SPLARE vectors (default Top-K = (40, 400) for
(queries, documents), max 1024 tokens, no instructions, as in the SPLARE paper) and a dot-product
similarity. MTEB does data loading (pinned dataset revisions), search and nDCG@10.

Multi-GPU: every rank runs the same MTEB loop; inside encode() each rank encodes its strided share
of the length-sorted texts and the top-k (values, indices) are all-gathered, so all ranks hold the
same embeddings. Only rank 0 writes results.
    torchrun --nproc_per_node=gpu mteb_eval.py --adapter <mntp> --adapter <splare> --out output

Datasets are fetched beforehand (network) with `python mteb_eval.py --prefetch` (pkq_configs/common.sh);
the evaluation itself runs with HF_HUB_OFFLINE=1 unless --online is given.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# The 10 retrieval tasks of MTEB(eng, v2), as in SPLARE's Table (full results, MTEB English).
TASKS = ["ArguAna", "CQADupstackGamingRetrieval", "CQADupstackUnixRetrieval", "ClimateFEVERHardNegatives",
         "FEVERHardNegatives", "FiQA2018", "HotpotQAHardNegatives", "SCIDOCS", "TRECCOVID", "Touche2020Retrieval.v3"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="output")
    ap.add_argument("--label", default=None)
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--adapter", action="append", default=[], help="PEFT adapter dir; repeat to merge in order")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--sae-kind", default="gemma_scope")
    ap.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    ap.add_argument("--sae-width", default="16k")
    ap.add_argument("--sae-l0-dir", default="average_l0_82")
    ap.add_argument("--topk-q", type=int, default=40)
    ap.add_argument("--topk-d", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--prefetch", action="store_true", help="only download the task datasets (needs network), then exit")
    ap.add_argument("--online", action="store_true", help="allow network access during evaluation")
    return ap.parse_args()


def import_mteb():
    """mteb 1.x imports transformers.AutoModelForVision2Seq at start-up (for the GME vision model, unused here);
    transformers 5 renamed it to AutoModelForImageTextToText. Alias it so mteb imports; nothing we run uses it.
    sentence_transformers replaces sys.modules["transformers"] when imported, so import it first and alias
    on the module that is current afterwards."""
    import sentence_transformers  # noqa: F401
    tf = sys.modules["transformers"]
    if not hasattr(tf, "AutoModelForVision2Seq"):
        tf.AutoModelForVision2Seq = tf.AutoModelForImageTextToText
    import mteb
    return mteb


def prefetch(tasks):
    mteb = import_mteb()
    for t in mteb.get_tasks(tasks=tasks):
        t.load_data()
        print(f"cached {t.metadata.name}: {t.metadata.dataset['path']}@{t.metadata.dataset['revision']}", flush=True)


class SplareMTEB:
    """MTEB-compatible encoder: Top-K SPLARE vectors (dense numpy arrays) + dot-product similarity."""

    def __init__(self, args, rank, world, device):
        import torch
        from drift.modeling import ResidCapture, check_attention_mode, load_backbone
        from drift.sae import load_sae
        self.torch, self.rank, self.world, self.device, self.args = torch, rank, world, device, args
        self.model, self.tok = load_backbone(args.model, args.adapter, layer=args.layer, device=device)
        self.capture = ResidCapture(self.model, args.layer)
        check_attention_mode(self.model, self.capture, self.tok, bidirectional=True, device=device)
        self.sae = load_sae(args.sae_kind, repo=args.sae_repo, layer=args.layer, width=args.sae_width,
                            l0_dir=args.sae_l0_dir, device=device)
        self.width = self.sae.width
        self.n_similarity_calls = 0

    @staticmethod
    def _text(s):
        if isinstance(s, dict):
            t = (s.get("title") or "").strip()
            return (t + " " + s.get("text", "")).strip() if t else s.get("text", "")
        return s

    @staticmethod
    def _is_query(kwargs):
        pt = kwargs.get("prompt_type")
        return pt is not None and "query" in str(getattr(pt, "value", pt)).lower()

    def _encode(self, texts, is_query):
        torch, a = self.torch, self.args
        from drift.splare import encode, topk_sparsify
        k = a.topk_q if is_query else a.topk_d
        n = len(texts)
        order = sorted(range(n), key=lambda i: len(texts[i]))
        mine = order[self.rank::self.world]
        per = (n + self.world - 1) // self.world
        vals = torch.zeros(per, k, device=self.device)
        idx = torch.zeros(per, k, dtype=torch.long, device=self.device)
        pos = torch.full((per,), -1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            for s in range(0, len(mine), a.batch):
                ids_ = mine[s: s + a.batch]
                b = self.tok([texts[i] for i in ids_], truncation=True, max_length=a.max_len,
                             padding=True, return_tensors="pt")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    r = encode(self.model, self.capture, self.sae, b["input_ids"].to(self.device),
                               b["attention_mask"].to(self.device), self.tok.bos_token_id)
                v, i = topk_sparsify(r, k).topk(k, dim=-1)
                vals[s: s + len(ids_)], idx[s: s + len(ids_)] = v.float(), i
                pos[s: s + len(ids_)] = torch.tensor(ids_, device=self.device)
        if self.world > 1:
            import torch.distributed as dist
            gv = [torch.empty_like(vals) for _ in range(self.world)]
            gi = [torch.empty_like(idx) for _ in range(self.world)]
            gp = [torch.empty_like(pos) for _ in range(self.world)]
            dist.all_gather(gv, vals)
            dist.all_gather(gi, idx)
            dist.all_gather(gp, pos)
            vals, idx, pos = torch.cat(gv), torch.cat(gi), torch.cat(gp)
        keep = pos >= 0
        out = torch.zeros(n, self.width)
        out[pos[keep].cpu()] = torch.zeros(int(keep.sum()), self.width).scatter_(
            1, idx[keep].cpu(), vals[keep].cpu())
        return out.numpy()

    # MTEB entry points (different mteb versions call different ones).
    def encode(self, sentences, **kwargs):
        return self._encode([self._text(s) for s in sentences], self._is_query(kwargs))

    def encode_queries(self, queries, **kwargs):
        return self._encode([self._text(s) for s in queries], True)

    def encode_corpus(self, corpus, **kwargs):
        if isinstance(corpus, dict):
            corpus = [{"title": t, "text": x} for t, x in zip(corpus.get("title", [""] * len(corpus["text"])),
                                                             corpus["text"])]
        return self._encode([self._text(s) for s in corpus], False)

    def similarity(self, a, b):
        """Dot product (SPLARE's score), computed on GPU."""
        torch = self.torch
        self.n_similarity_calls += 1
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        b = torch.as_tensor(b, dtype=torch.float32, device=self.device)
        return (a @ b.T).cpu()

    def similarity_pairwise(self, a, b):
        torch = self.torch
        a = torch.as_tensor(a, dtype=torch.float32)
        b = torch.as_tensor(b, dtype=torch.float32)
        return (a * b).sum(-1)


def main():
    args = parse_args()
    if args.prefetch:
        prefetch(args.tasks)
        return
    if not args.online:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch
    from drift.dist import cleanup, setup_dist
    mteb = import_mteb()

    rank, world, local = setup_dist()
    device = f"cuda:{local}"
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = SplareMTEB(args, rank, world, device)
    if rank == 0:
        print(f"model ready ({world} GPU(s)); tasks: {args.tasks}", flush=True)

    scores = {}
    for task in mteb.get_tasks(tasks=args.tasks):
        name = task.metadata.name
        t = time.time()
        res = mteb.MTEB(tasks=[task]).run(model, output_folder=str(out / "mteb_results") if rank == 0 else None,
                                         overwrite_results=True, verbosity=0)
        if model.n_similarity_calls == 0:
            raise RuntimeError("MTEB did not call model.similarity; scores would use the wrong similarity function.")
        scores[name] = float(res[0].get_score())
        if rank == 0:
            print(f"{name}: nDCG@10 {100 * scores[name]:.2f} ({time.time() - t:.0f}s)", flush=True)

    if rank == 0:
        avg = 100 * sum(scores.values()) / len(scores)
        summary = {"label": args.label or out.name, "args": vars(args), "world_size": world,
                   "ndcg@10": {k: 100 * v for k, v in scores.items()}, "average": avg,
                   "mteb_version": mteb.__version__, "runtime_sec": time.time() - t0}
        (out / "mteb_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"MTEB(eng, v2) retrieval average nDCG@10: {avg:.2f} over {len(scores)} tasks "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    cleanup()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
