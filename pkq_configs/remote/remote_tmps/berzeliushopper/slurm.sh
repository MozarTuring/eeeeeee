#!/bin/bash

early_warning() {
    echo "2 minutes left — saving checkpoint..."
    # save_checkpoint
    # optionally keep running, or exit gracefully
}

final_cleanup() {
    echo "Being killed — last-resort cleanup..."
}

trap early_warning SIGUSR1 # 120s before limit — your warning

trap final_cleanup SIGTERM # 0s — SLURM is killing you

module --force purge
module load ${PKQ_MODULES}

for _eroot in "$EBROOTCUDA/lib64" "$EBROOTCUDNN/lib" "$EBROOTCUSPARSELT/lib"; do
    [[ -d "$_eroot" ]] && export LD_LIBRARY_PATH="${_eroot}:${LD_LIBRARY_PATH:-}"
done

export LD_LIBRARY_PATH=${PKQ_CONDAENV}${PKQ_ARCH}/lib/${PKQ_PYTHON}/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

echo "LD_LIBRARY_PATH, ${LD_LIBRARY_PATH}"


conda activate ${PKQ_CONDAENV}${PKQ_ARCH}

which python
mkdir -p pkq_configs/packages
mkdir -p pkq_configs/logs_${PKQ_RUN_START_TIME}
python -m pip list > pkq_configs/packages/${PKQ_SERVER_NAME}.txt

bash ${RUN_DIR_HOME}/project_remote_pkq/common_tools_pikaq/resource_usage.sh >pkq_configs/logs_${PKQ_RUN_START_TIME}/resource_usage.log &

echo "TORCH_CUDA_ARCH_LIST ${TORCH_CUDA_ARCH_LIST}"

if [[ -n ${JWM_build_flashattn} ]]; then
    MAX_JOBS=${CPUS_PER_TASK} FLASH_ATTENTION_FORCE_BUILD=TRUE pip install ${PKQ_CONDAENV}/flash_attn_src/flash_attn*.tar.gz --no-build-isolation --no-cache-dir
    echo "flash attn build done"
    exit
fi
# Copy LIBRARY_PATH to LD_LIBRARY_PATH, but strip stubs dirs —
# CUDA module's stubs/lib64 has fake libnvidia-ml.so/libcuda.so
# that shadow the real driver and break GPU init.
_lib_path_no_stubs=$(echo "${LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/stubs/' | paste -sd ':')
export LD_LIBRARY_PATH=${_lib_path_no_stubs:+${_lib_path_no_stubs}:}${LD_LIBRARY_PATH:-}
echo "PWD, ${PWD}"
echo "PKQ_RUN_COMMAND, ${PKQ_RUN_COMMAND}"
srun --ntasks=1 ${PKQ_RUN_COMMAND} &

wait $!

exit_code=$?
remote_dir=${PWD}
echo "slurm remote dir, ${remote_dir}"

if [ $exit_code -eq 0 ]; then
    echo "Command succeeded."
    mkdir -p "${RUN_DIR_HOME}/project_remote_pkq/remote_data/${RUN_PROJ_NAME}/backup/${PKQ_RUN_START_TIME}" && rsync -a ./ "${RUN_DIR_HOME}/project_remote_pkq/remote_data/${RUN_PROJ_NAME}/backup/${PKQ_RUN_START_TIME}/" 2>&1 && touch ./pkq_run_succeed.flag

else
    echo "Command failed with exit code $exit_code."
fi

