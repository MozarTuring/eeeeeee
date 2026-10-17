
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
export PKQ_RUN_COMMAND="python e1_compare.py --ref ${PKQ_DATA_DIR}/backup/20261017_115544/output --runs ${PKQ_DATA_DIR}/backup/20261017_115544/output ${PKQ_DATA_DIR}/backup/20261017_121053/output ${PKQ_DATA_DIR}/backup/20261017_121130/output --out output"
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

[ -f "$CORPUS/meta.json" ] || python build_corpus.py --tokenizer google/gemma-2-2b --out "$CORPUS"


pip list > pkq_configs/packages.txt
sbatch --signal=B:USR1@120 --time=1-00:00:00 --nodes=1 --output=pkq_configs/logs_20261017_122039/job-%j.out --error=pkq_configs/logs_20261017_122039/job-%j.out  --gpus=1 --cpus-per-task=8 --mem=24G  -A berzelius-2026-243 --partition=berzelius-hopper pkq_configs/remote/remote_tmps/berzeliushopper/slurm.sh
