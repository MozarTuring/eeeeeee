#!/usr/bin/env bash
# E1 pilot: Gemma-2-2B + Gemma Scope residual SAE at layer 12.
# Runs that need no trained checkpoint are enabled; M1/M2 are added once adapters exist.
set -euo pipefail

CORPUS=${CORPUS:-data/e1_corpus}
RUNS=${RUNS:-runs/e1}
WIDTH=${WIDTH:-16k}
LAYER=${LAYER:-12}
COMMON="--corpus $CORPUS --layer $LAYER --sae-width $WIDTH"

[ -f "$CORPUS/meta.json" ] || python build_corpus.py --tokenizer google/gemma-2-2b --out "$CORPUS"

# M0: base model, causal: the SAE's native setting (reference).
python e1_sae_fit.py $COMMON --out $RUNS/m0 --label M0

# Positive control: base -> chat transfer is known to mostly work.
python e1_sae_fit.py $COMMON --out $RUNS/m0_it --label M0-it \
    --model google/gemma-2-2b-it --norm-ref $RUNS/m0/summary.json

# Bidirectional attention with base weights, no training: drift from the attention change alone.
python e1_sae_fit.py $COMMON --out $RUNS/m0_bidir --label M0-bidir \
    --bidirectional --norm-ref $RUNS/m0/summary.json

# Once trained adapters exist:
# python e1_sae_fit.py $COMMON --out $RUNS/m1 --label M1 --bidirectional \
#     --adapter ckpt/mntp --norm-ref $RUNS/m0/summary.json
# python e1_sae_fit.py $COMMON --out $RUNS/m2 --label M2 --bidirectional \
#     --adapter ckpt/mntp --adapter ckpt/splare --norm-ref $RUNS/m0/summary.json

python e1_compare.py --ref $RUNS/m0 --out reports/e1 \
    --runs $RUNS/m0 $RUNS/m0_it $RUNS/m0_bidir
