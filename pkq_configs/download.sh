#!/usr/bin/env bash
# Pure downloads (hf download) into the default HF cache. Scripts refer to everything by repo id and
# resolve it from the cache without network access.
# Gemma-2 is gated: accept the licence on HF and `hf auth login` (or set HF_TOKEN) first.
set -euo pipefail

WIDTH=${WIDTH:-16k}
LAYER=${LAYER:-12}
SAE_L0=${SAE_L0:-average_l0_82}   # SPLARE: variant with L0 closest to 100 (82 for layer 12 / 16k)

# Backbones (skip GGUF and other non-HF weight formats).
for m in google/gemma-2-2b google/gemma-2-2b-it; do
    hf download "$m" --include "*.json" --include "*.safetensors" --include "tokenizer*"
done

# SAE used by e1_sae_fit.py / train_splare.py (--layer/--sae-width/--sae-l0-dir must match).
hf download google/gemma-scope-2b-pt-res "layer_${LAYER}/width_${WIDTH}/${SAE_L0}/params.npz"

# Cross-encoder teacher for train_splare.py.
hf download naver/trecdl22-crossencoder-debertav3

# MNTP corpus (train_mntp.py --data). Only needed to re-run MNTP; M1 is already trained.
# hf download Tevatron/msmarco-passage-corpus --repo-type dataset

# SPLARE training data: queries with positive / negative passages (train_splare.py --train-data).
hf download Tevatron/msmarco-passage --repo-type dataset

# NanoBEIR evaluation sets (train_splare.py --nanobeir-org).
for n in NanoArguAna NanoClimateFEVER NanoDBPedia NanoFEVER NanoFiQA2018 NanoHotpotQA NanoMSMARCO \
         NanoNFCorpus NanoNQ NanoQuoraRetrieval NanoSCIDOCS NanoSciFact NanoTouche2020; do
    hf download "zeta-alpha-ai/$n" --repo-type dataset
done
