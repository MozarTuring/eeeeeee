"""Build the fixed tokenised corpus used by every E1 run.

Every model (M0, M1, M2, ...) is evaluated on exactly the same token ids.
The corpus mixes English web text, multilingual web text and MS MARCO passages
(the retrieval training domain), so drift can be reported per source.
Sources are streamed, so only the sampled documents are downloaded.

Example:
    python build_corpus.py --tokenizer google/gemma-2-2b --out data/e1_corpus \
        --tokens-per-source 300000
"""
import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Languages follow the SPLARE paper's evaluation (MMTEB splits, MIRACL, XTREME-UP low-resource).
FINEWEB2_LANGS = ["fra_Latn", "deu_Latn", "spa_Latn", "rus_Cyrl", "cmn_Hani", "jpn_Jpan",
                  "arb_Arab", "hin_Deva", "swh_Latn", "ben_Beng", "tam_Taml", "yor_Latn"]


def iter_fineweb():
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    for ex in ds:
        yield ex["text"]


def iter_fineweb2(lang):
    ds = load_dataset("HuggingFaceFW/fineweb-2", name=lang, split="train", streaming=True)
    for ex in ds:
        yield ex["text"]


def iter_msmarco():
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True)
    for ex in ds:
        yield from ex["passages"]["passage_text"]


def source_specs(langs):
    specs = [("en_web", iter_fineweb), ("msmarco", iter_msmarco)]
    specs += [(f"web_{l}", (lambda l=l: iter_fineweb2(l))) for l in langs]
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="google/gemma-2-2b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens-per-source", type=int, default=300_000)
    ap.add_argument("--max-len", type=int, default=512, help="tokens per doc incl. BOS (truncated)")
    ap.add_argument("--min-len", type=int, default=16)
    ap.add_argument("--langs", nargs="*", default=FINEWEB2_LANGS)
    ap.add_argument("--shuffle-buffer", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    all_tokens, lengths, source_ids, halves, sources = [], [], [], [], []
    for name, make_iter in source_specs(args.langs):
        docs, n_tok = [], 0
        buf = []
        try:
            # Streaming datasets are ordered; sample through a shuffle buffer.
            for text in make_iter():
                buf.append(text)
                if len(buf) < args.shuffle_buffer:
                    continue
                text = buf.pop(rng.integers(len(buf)))
                ids = tok(text, truncation=True, max_length=args.max_len)["input_ids"]
                if len(ids) < args.min_len:
                    continue
                docs.append(np.asarray(ids, dtype=np.int32))
                n_tok += len(ids)
                if n_tok >= args.tokens_per_source:
                    break
        except Exception as e:  # e.g. a FineWeb-2 config name that does not exist
            print(f"WARNING: skipping source {name}: {type(e).__name__}: {e}")
            continue
        if n_tok < args.tokens_per_source:
            print(f"WARNING: {name} only has {n_tok} tokens (< {args.tokens_per_source})")
        sid = len(sources)
        sources.append(name)
        for j, ids in enumerate(docs):
            all_tokens.append(ids)
            lengths.append(len(ids))
            source_ids.append(sid)
            halves.append(j % 2)
        print(f"{name:>14}: {len(docs):6d} docs, {n_tok:9d} tokens")

    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.save(out / "tokens.npy", np.concatenate(all_tokens))
    np.save(out / "offsets.npy", offsets)
    np.save(out / "source_ids.npy", np.asarray(source_ids, dtype=np.int16))
    np.save(out / "halves.npy", np.asarray(halves, dtype=np.int8))
    meta = {"sources": sources, "tokenizer": args.tokenizer, "bos_token_id": tok.bos_token_id,
            "pad_token_id": tok.pad_token_id, **{k: v for k, v in vars(args).items() if k != "out"}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"total: {len(lengths)} docs, {offsets[-1]} tokens -> {out}")


if __name__ == "__main__":
    main()
