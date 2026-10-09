# vLLM 0.30 / Triton 3.7 on RTX 5090 (sm_120) with driver 575 (CUDA <= 12.9); source with the vLLM environment active.
# Triton's bundled CUDA 13.1 ptxas makes the 12.9 driver segfault, so use the 12.9 ptxas from the nvidia-cuda-nvcc
# wheel, separate caches (the defaults may hold 13.1 cubins) and no FlashInfer sampler (its JIT rejects sm_120).
export TRITON_PTXAS_BLACKWELL_PATH=${VIRTUAL_ENV:?activate the vLLM environment}/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas
export TRITON_PTXAS_PATH=$TRITON_PTXAS_BLACKWELL_PATH
export TRITON_CACHE_DIR=${UBT_WORK:-$PWD/work}/cache/triton129
export VLLM_CACHE_ROOT=${UBT_WORK:-$PWD/work}/cache/vllm129
export VLLM_USE_FLASHINFER_SAMPLER=0
# NCCL's cuMem path fails with "peer access is not supported" between processes that each see a different GPU
# (vLLM server <-> TRL weight sync, --isolate-gpus DDP); the legacy path works.
export NCCL_CUMEM_ENABLE=0
