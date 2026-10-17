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
