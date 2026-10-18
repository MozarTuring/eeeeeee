"""Summary statistics of the evaluation datasets: MTEB(eng, v2) retrieval (as loaded by mteb_eval.py) and NanoBEIR.

Per dataset: #queries (evaluated, i.e. with qrels), #documents, #relevant (query, doc) pairs, relevant docs per query,
relevance grades present, and lengths in Gemma tokens (no special tokens): queries (all, and share longer than the
query Top-K of 40) and documents ("title text"; share longer than the 1023 content tokens kept at max length 1024).
Document lengths use a random sample (default 20k docs per dataset, seed 0); counts are exact. CPU only, offline.

    python eval_data_stats.py --out output
"""
import argparse
import json
import os
import random
import statistics as st
from pathlib import Path


def doc_text(d):
    t = (d.get("title") or "").strip()
    return (t + " " + d.get("text", "")).strip() if t else d.get("text", "")


def stats(name, queries, docs, qrels, ntok, sample, query_topk, max_content):
    qids = [q for q in qrels if q in queries]
    rel = [sum(1 for v in qrels[q].values() if v > 0) for q in qids]
    ql = ntok([queries[q] for q in qids])
    ids = list(docs)
    random.seed(0)
    samp = random.sample(ids, min(sample, len(ids)))
    dl = ntok([docs[i] for i in samp])
    return {
        "dataset": name, "queries": len(qids), "docs": len(docs), "rel_pairs": sum(rel),
        "rel_per_query_mean": st.mean(rel), "rel_per_query_median": st.median(rel),
        "grades": sorted({v for q in qids for v in qrels[q].values()}),
        "query_tokens_mean": st.mean(ql), "query_tokens_median": st.median(ql),
        "query_frac_gt_topk": sum(x > query_topk for x in ql) / len(ql),
        "doc_tokens_mean": st.mean(dl), "doc_tokens_median": st.median(dl),
        "doc_tokens_p95": sorted(dl)[int(0.95 * (len(dl) - 1))],
        "doc_frac_truncated": sum(x > max_content for x in dl) / len(dl), "doc_length_sample": len(samp),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="output")
    ap.add_argument("--tokenizer", default="google/gemma-2-2b")
    ap.add_argument("--doc-sample", type=int, default=20000)
    ap.add_argument("--query-topk", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=1024)
    args = ap.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    from mteb_eval import TASKS, import_mteb
    from drift.nanobeir import NANOBEIR, load_nanobeir
    mteb = import_mteb()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ntok = lambda texts: [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]]
    kw = dict(ntok=ntok, sample=args.doc_sample, query_topk=args.query_topk, max_content=args.max_len - 1)

    rows = {"mteb_eng_v2": [], "nanobeir": []}
    for t in mteb.get_tasks(tasks=TASKS):
        t.load_data()
        s = t.metadata.eval_splits[0]
        q = {k: (v if isinstance(v, str) else str(v)) for k, v in t.queries[s].items()}
        d = {k: doc_text(v) for k, v in t.corpus[s].items()}
        rows["mteb_eng_v2"].append(stats(t.metadata.name, q, d, t.relevant_docs[s], **kw))
        print(f"done {t.metadata.name}", flush=True)
    for name in NANOBEIR:
        doc_ids, docs, qids, qtexts, rel = load_nanobeir("zeta-alpha-ai", name)
        rows["nanobeir"].append(stats(name, dict(zip(qids, qtexts)), dict(zip(doc_ids, docs)), rel, **kw))
        print(f"done {name}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_data_stats.json").write_text(json.dumps(rows, indent=2))
    cols = ["dataset", "queries", "docs", "rel_pairs", "rel_per_query_mean", "query_tokens_mean", "query_frac_gt_topk",
            "doc_tokens_mean", "doc_tokens_median", "doc_tokens_p95", "doc_frac_truncated"]
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    md = []
    for group, rs in rows.items():
        md += [f"## {group}", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        md += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rs]
        tot = {c: sum(r[c] for r in rs) for c in ["queries", "docs", "rel_pairs"]}
        md += [f"Total: {tot['queries']} queries, {tot['docs']} docs, {tot['rel_pairs']} relevant pairs.", ""]
    (out / "eval_data_stats.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
