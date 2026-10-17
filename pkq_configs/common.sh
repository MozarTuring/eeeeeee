pip install -r requirements.txt

CORPUS=${CORPUS:-data/e1_corpus}
RUNS=${RUNS:-runs/e1}
WIDTH=${WIDTH:-16k}
LAYER=${LAYER:-12}
COMMON="--corpus $CORPUS --layer $LAYER --sae-width $WIDTH"

[ -f "${PKQ_DATA_DIR}/backup/20262356/$CORPUS/meta.json" ] || python build_corpus.py --tokenizer google/gemma-2-2b --out "$CORPUS"

# MTEB(eng, v2) retrieval datasets at the revisions pinned by mteb (into the HF cache; mteb_eval.py then runs offline).
python mteb_eval.py --prefetch


