pip install -r requirements.txt

CORPUS=${CORPUS:-data/e1_corpus}
RUNS=${RUNS:-runs/e1}
WIDTH=${WIDTH:-16k}
LAYER=${LAYER:-12}
COMMON="--corpus $CORPUS --layer $LAYER --sae-width $WIDTH"

[ -f "${PKQ_DATA_DIR}/backup/20262356/$CORPUS/meta.json" ] || python build_corpus.py --tokenizer google/gemma-2-2b --out "$CORPUS"


