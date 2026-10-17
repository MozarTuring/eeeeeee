PKQ_PYTHON="3.10"
PKQ_SERVER_NAME=berzeliushopper
# PKQ_SERVER_NAME=arrhenius
# PKQ_SERVER_NAME=berzeliusampere
PKQ_GPU_NUM=1
PKQ_NODES_NUM=1
PKQ_RUN_TIME="1-00:00:00"
PKQ_SLURM_FILE=slurm.sh
PKQ_build_flashattn=
PKQ_NOTEBOOK=
PKQ_INTERACTIVE=

MEM_PER_TASK="$((80 * PKQ_GPU_NUM))G"
CPUS_PER_TASK=$((8 * PKQ_GPU_NUM))

CUDA_VISIBLE_DEVICES=1

# PKQ_RUN_COMMAND="python e1_sae_fit.py --corpus ${PKQ_DATA_DIR}/backup/20262356/data/e1_corpus --layer 12 --sae-width 16k --out output --label M0"

# PKQ_RUN_COMMAND="python e1_sae_fit.py --corpus ${PKQ_DATA_DIR}/backup/20262356/data/e1_corpus --layer 12 --sae-width 16k --model google/gemma-2-2b-it --norm-ref ${PKQ_DATA_DIR}/backup/20261017_092409/output/summary.json --out output --label M0-it"

# PKQ_RUN_COMMAND="python e1_sae_fit.py --corpus ${PKQ_DATA_DIR}/backup/20262356/data/e1_corpus --layer 12 --sae-width 16k --bidirectional --norm-ref ${PKQ_DATA_DIR}/backup/20261017_092409/output/summary.json --out output --label M0-bidir"

PKQ_RUN_COMMAND="python e1_sae_fit.py --corpus ${PKQ_DATA_DIR}/backup/20262356/data/e1_corpus --layer 12 --sae-width 16k --out output --label M0"
