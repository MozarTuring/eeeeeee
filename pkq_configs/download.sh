#!/usr/bin/env bash
# Download the backbones and the Gemma Scope SAE into the HF cache.
# (Corpus text is streamed by build_corpus.py.)
# Gemma-2 is gated: accept the licence on HF and `hf auth login` (or set HF_TOKEN) first.
set -euo pipefail

WIDTH=${WIDTH:-16k}
LAYER=${LAYER:-12}

# Backbones (skip GGUF and other non-HF weight formats).
for m in google/gemma-2-2b google/gemma-2-2b-it; do
    hf download "$m" --include "*.json" "*.safetensors" "tokenizer*"
done

# SAE: all released L0 variants at this layer/width (e1_sae_fit.py picks the one closest to 100).
hf download google/gemma-scope-2b-pt-res --include "layer_${LAYER}/width_${WIDTH}/*/params.npz"
