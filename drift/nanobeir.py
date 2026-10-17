"""NanoBEIR nDCG@10 for a sparse encoder, multi-GPU (corpus sharded across ranks, top-k merged).

SPLARE used NanoBEIR nDCG@10 to select its temperature. Datasets: zeta-alpha-ai/Nano*, each with
configs corpus / queries / qrels (split "train").
"""
import math

import torch
import torch.distributed as dist
from datasets import load_dataset

from drift.dist import is_dist

NANOBEIR = ["NanoArguAna", "NanoClimateFEVER", "NanoDBPedia", "NanoFEVER", "NanoFiQA2018", "NanoHotpotQA",
            "NanoMSMARCO", "NanoNFCorpus", "NanoNQ", "NanoQuoraRetrieval", "NanoSCIDOCS", "NanoSciFact",
            "NanoTouche2020"]


def load_nanobeir(name):
    corpus = load_dataset(f"zeta-alpha-ai/{name}", "corpus", split="train")
    queries = load_dataset(f"zeta-alpha-ai/{name}", "queries", split="train")
    qrels = load_dataset(f"zeta-alpha-ai/{name}", "qrels", split="train")
    doc_ids = [str(d["_id"]) for d in corpus]
    docs = [(d.get("title") or "") + (" " if d.get("title") else "") + d["text"] for d in corpus]
    rel = {}
    for r in qrels:
        rel.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = float(r.get("score", 1) or 1)
    qs = [(str(q["_id"]), q["text"]) for q in queries if str(q["_id"]) in rel]
    return doc_ids, docs, [q for q, _ in qs], [t for _, t in qs], rel


def ndcg_at_10(ranked_doc_ids, rel):
    dcg = sum(rel.get(d, 0.0) / math.log2(i + 2) for i, d in enumerate(ranked_doc_ids[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


@torch.no_grad()
def evaluate_nanobeir(encode_texts, rank, world, names=NANOBEIR, topk_q=40, topk_d=400, k=10):
    """encode_texts(texts, is_query) -> (n, W) Top-K-sparsified reps on this rank's device.

    Each rank encodes every query and its strided shard of the corpus, keeps its local top-k per
    query, and the shards' top-k lists are merged on all ranks. Returns {name: nDCG@10, "mean": ...}.
    """
    results = {}
    for name in names:
        doc_ids, docs, qids, qtexts, rel = load_nanobeir(name)
        Q = encode_texts(qtexts, True)
        my = list(range(rank, len(docs), world))
        D = encode_texts([docs[i] for i in my], False)
        s = Q @ D.T  # (nq, n_my)
        kk = min(k, s.shape[1])
        vals, idx = s.topk(kk, dim=1)
        gidx = torch.tensor(my, device=s.device)[idx]
        if kk < k:  # pad so every rank contributes a (nq, k) block
            pad = k - kk
            vals = torch.cat([vals, torch.full((len(qids), pad), -1e9, device=s.device)], 1)
            gidx = torch.cat([gidx, torch.zeros((len(qids), pad), dtype=gidx.dtype, device=s.device)], 1)
        if is_dist():
            all_v = [torch.empty_like(vals) for _ in range(world)]
            all_i = [torch.empty_like(gidx) for _ in range(world)]
            dist.all_gather(all_v, vals.contiguous())
            dist.all_gather(all_i, gidx.contiguous())
            vals, gidx = torch.cat(all_v, 1), torch.cat(all_i, 1)
        order = vals.argsort(dim=1, descending=True)[:, :k]
        top = gidx.gather(1, order).tolist()
        scores = [ndcg_at_10([doc_ids[j] for j in top[qi]], rel[qid]) for qi, qid in enumerate(qids)]
        results[name] = sum(scores) / len(scores)
    results["mean"] = sum(results[n] for n in names) / len(names)
    return results
