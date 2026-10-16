"""Fixed tokenised evaluation corpus, shared by every model being compared."""
import json
from pathlib import Path

import numpy as np
import torch


class Corpus:
    """Docs stored as concatenated token ids + offsets, with a source id and a half id per doc.

    half = doc index parity within its source; it gives the within-model noise floor.
    """

    def __init__(self, path):
        path = Path(path)
        self.tokens = np.load(path / "tokens.npy")
        self.offsets = np.load(path / "offsets.npy")  # (n_docs + 1,)
        self.source_ids = np.load(path / "source_ids.npy")
        self.halves = np.load(path / "halves.npy")
        self.meta = json.loads((path / "meta.json").read_text())
        self.sources = self.meta["sources"]

    def __len__(self):
        return len(self.source_ids)

    def doc(self, i):
        return self.tokens[self.offsets[i]: self.offsets[i + 1]]

    def lengths(self):
        return np.diff(self.offsets)

    def batches(self, max_tokens, pad_id, doc_ids=None):
        """Yield (input_ids, attention_mask, source_ids, halves) batches of length-sorted docs."""
        lens = self.lengths()
        ids = np.arange(len(self)) if doc_ids is None else np.asarray(doc_ids)
        ids = ids[np.argsort(lens[ids], kind="stable")]
        batch = []
        for i in ids:
            if batch and (len(batch) + 1) * lens[i] > max_tokens:
                yield self._collate(batch, pad_id)
                batch = []
            batch.append(i)
        if batch:
            yield self._collate(batch, pad_id)

    def _collate(self, batch, pad_id):
        T = max(len(self.doc(i)) for i in batch)
        input_ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), T), dtype=torch.long)
        for r, i in enumerate(batch):
            d = torch.from_numpy(self.doc(i).astype(np.int64))
            input_ids[r, : len(d)] = d
            attn[r, : len(d)] = 1
        src = torch.from_numpy(self.source_ids[batch].astype(np.int64))
        half = torch.from_numpy(self.halves[batch].astype(np.int64))
        return input_ids, attn, src, half
