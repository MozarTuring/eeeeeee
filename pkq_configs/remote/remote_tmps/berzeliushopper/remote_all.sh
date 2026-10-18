
set -e
# set following vars based on your setting, and then make sure this repo is cloned to /proj/berzelius-aiics-real/users/x_jinma/project_remote_pkq/eeeeeee_pikaq
export RUN_DIR_HOME=
export RUN_PROJ=

cd ${RUN_DIR_HOME}/project_remote_pkq/${RUN_PROJ}
export PKQ_PYTHON="3.10"
export PKQ_SERVER_NAME=berzeliushopper
export PKQ_GPU_NUM=1
export PKQ_NODES_NUM=1
export PKQ_RUN_TIME="1-00:00:00"
export PKQ_SLURM_FILE=slurm.sh
export PKQ_build_flashattn=
export PKQ_NOTEBOOK=
export PKQ_INTERACTIVE=
export MEM_PER_TASK="$((80 * PKQ_GPU_NUM))G"
export CPUS_PER_TASK=$((8 * PKQ_GPU_NUM))
export CUDA_VISIBLE_DEVICES=1
export PKQ_RUN_COMMAND="python eval_data_stats.py --out output"
if [[ -z ${PKQ_MODULES} ]]; then
export PKQ_MODULES="Miniforge3"
fi
export PKQ_LOGIN_MODULES=${PKQ_MODULES}
module --force purge
module load ${PKQ_LOGIN_MODULES}
PKQTMP=${RUN_DIR_HOME}/pkqcondaenv/pkqbase
if [[ ! -d ${PKQTMP} ]]; then
    conda create -p ${PKQTMP} pip -y
fi
conda activate ${PKQTMP}
which python
python --version
which pip
pip install -q huggingface_hub
if [ -z ${PKQ_CONDAENV} ]; then
    export PKQ_CONDAENV=${RUN_DIR_HOME}/pkqcondaenv/${RUN_PROJ}
    export PKQ_WHEELS=${RUN_DIR_HOME}/pkqwheels/${RUN_PROJ}
fi
echo "condaenv path ${PKQ_CONDAENV}"
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
module --force purge
module load ${PKQ_MODULES}

if [[ ! -d ${PKQ_CONDAENV}${PKQ_ARCH} ]]; then
    conda create -p ${PKQ_CONDAENV}${PKQ_ARCH} python=${PKQ_PYTHON} pip -y
fi
conda activate ${PKQ_CONDAENV}${PKQ_ARCH}
which python
python --version
which pip
pip install -r requirements.txt

CORPUS=${CORPUS:-data/e1_corpus}
RUNS=${RUNS:-runs/e1}
WIDTH=${WIDTH:-16k}
LAYER=${LAYER:-12}
COMMON="--corpus $CORPUS --layer $LAYER --sae-width $WIDTH"

[ -f "${PKQ_DATA_DIR}/backup/20262356/$CORPUS/meta.json" ] || python build_corpus.py --tokenizer google/gemma-2-2b --out "$CORPUS"

# MTEB(eng, v2) retrieval datasets at the revisions pinned by mteb (into the HF cache; mteb_eval.py then runs offline).
python mteb_eval.py --prefetch


pip list > pkq_configs/packages.txt
sbatch --signal=B:USR1@120 --time=1-00:00:00 --nodes=1 --output=pkq_configs/logs_20261018_092154/job-%j.out --error=pkq_configs/logs_20261018_092154/job-%j.out  --gpus=1 --cpus-per-task=8 --mem=24G  -A berzelius-2026-243 --partition=berzelius-hopper pkq_configs/remote/remote_tmps/berzeliushopper/slurm.sh
