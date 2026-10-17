export PKQ_PYTHON="3.10"
export PKQ_SERVER_NAME=berzeliushopper
export PKQ_GPU_NUM=4
export PKQ_NODES_NUM=1
export PKQ_RUN_TIME="1-00:00:00"
export PKQ_SLURM_FILE=slurm.sh
export PKQ_build_flashattn=
export PKQ_NOTEBOOK=
export PKQ_INTERACTIVE=1
export MEM_PER_TASK="$((80 * PKQ_GPU_NUM))G"
export CPUS_PER_TASK=$((8 * PKQ_GPU_NUM))
export CUDA_VISIBLE_DEVICES=1
export PKQ_RUN_COMMAND="torchrun --nproc_per_node=gpu mteb_eval.py --adapter ${PKQ_DATA_DIR}/backup/20261017_123949/output/adapter --adapter ${PKQ_DATA_DIR}/backup/20261017_200212/output/adapter --out output --label M2"
