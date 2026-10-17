"""Read datasets that pkq_configs/download.sh fetched with `hf download` into the default HF cache.

No network access: a dataset repo id is resolved to its cached snapshot (local_files_only=True).
A path to an existing local directory also works.
"""
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

PATTERNS = {"parquet": ["*.parquet"], "json": ["*.jsonl", "*.jsonl.gz", "*.json", "*.json.gz"]}


def resolve(repo_or_dir):
    """Local directory for a dataset repo id (cached snapshot) or an existing path."""
    p = Path(repo_or_dir)
    if p.is_dir():
        return p
    return Path(snapshot_download(repo_or_dir, repo_type="dataset", local_files_only=True))


def find_files(root, keyword=None):
    """(builder, files) for the data files under root.

    If keyword is given, keep only files whose relative path contains it (e.g. "train", "corpus").
    """
    root = Path(root)
    for builder, pats in PATTERNS.items():
        # Skip HF metadata dirs *inside* root (the HF cache itself lives under ~/.cache).
        files = sorted(p for pat in pats for p in root.rglob(pat) if ".cache" not in p.relative_to(root).parts)
        if keyword is not None:
            files = [p for p in files if keyword in p.relative_to(root).as_posix()]
        if files:
            return builder, [str(p) for p in files]
    present = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
    raise FileNotFoundError(f"no data files under {root}" + (f" matching {keyword!r}" if keyword else "") +
                            f"; present: {present[:50]}")


def load_local(repo_or_dir, keyword=None, streaming=False):
    builder, files = find_files(resolve(repo_or_dir), keyword)
    return load_dataset(builder, data_files=files, split="train", streaming=streaming)
