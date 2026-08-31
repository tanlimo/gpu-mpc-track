# DeepDTAGen affinity-path experiment — standalone, out-of-tree Makefile.
#
# These sources live in the PROJECT repo (idash/mpc/gpu_mpc) and treat the
# EzPC/GPU-MPC checkout as a READ-ONLY external dependency: nothing here is
# copied into or edited inside that upstream tree. Point GPU_MPC_ROOT at your
# EzPC/GPU-MPC checkout (default: ~/EzPC/GPU-MPC).
#
# Usage examples:
#
#   # Local RTX 4060 (SM 8.9), 32-bit ring, local toolkit install:
#   make GPU_MPC_ROOT=$HOME/EzPC/GPU-MPC BW=32 GPU_ARCH=89 deepdtagen_inference
#
#   # H100 (SM 9.0a), 64-bit production ring:
#   make GPU_MPC_ROOT=$HOME/EzPC/GPU-MPC BW=64 GPU_ARCH=90a deepdtagen_inference
#
#   # Dry-run (check command, no compile):
#   make ... --dry-run
#
# NVCC resolution order (first found wins):
#   1. NVCC_PATH  — explicit override
#   2. $(CUDA_HOME)/bin/nvcc   — if CUDA_HOME is set
#   3. nvcc on PATH
#   4. ~/.local/cuda-$(CUDA_VERSION)/usr/local/cuda-$(CUDA_VERSION)/bin/nvcc
#      (local .run toolkit install, no sudo)
#
# BW=32 default: 32-bit ring halves share bandwidth for rapid testing and
# matches the reference/share_data.py bw=32 mode.  Switch to BW=64 for the
# final production submission.

GPU_MPC_ROOT ?= $(HOME)/EzPC/GPU-MPC
CUDA_VERSION ?= 12.1
GPU_ARCH     ?= 89
BW           ?= 32

CUTLASS_PATH      = $(GPU_MPC_ROOT)/ext/cutlass
SYTORCH_PATH      = $(GPU_MPC_ROOT)/ext/sytorch
SYTORCH_BUILD_PATH = $(SYTORCH_PATH)/build
LLAMA_PATH        = $(SYTORCH_PATH)/ext/llama
HERE = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# ── nvcc path resolution ─────────────────────────────────────────────────────
ifdef NVCC_PATH
  CXX := $(NVCC_PATH)
else ifdef CUDA_HOME
  CXX := $(CUDA_HOME)/bin/nvcc
else ifneq ($(shell which nvcc 2>/dev/null),)
  CXX := $(shell which nvcc)
else
  CXX := $(HOME)/.local/cuda-$(CUDA_VERSION)/usr/local/cuda-$(CUDA_VERSION)/bin/nvcc
endif

# ── gencode: always include both virtual (PTX fallback) and real arch ─────────
# For SM_90a (H100 SXM/PCIe) we must use compute_90a (not compute_90) to
# unlock the Hopper-specific TMA / wgmma ISA.
ifeq ($(GPU_ARCH),90a)
  GENCODE := -gencode arch=compute_90a,code=[sm_90a,compute_90a]
else
  GENCODE := -gencode arch=compute_$(GPU_ARCH),code=[sm_$(GPU_ARCH),compute_$(GPU_ARCH)]
endif

# ── compile flags ─────────────────────────────────────────────────────────────
# -DInfType selects the ring element type: u32 (BW=32) or u64 (BW=64).
FLAGS := -O3 $(GENCODE) -std=c++17 -m64 \
         -DInfType=$(if $(filter 32,$(BW)),u32,u64) \
         -DBW=$(BW) \
         -Xcompiler="-O3,-w,-std=c++17,-fpermissive,-fpic,-pthread,-fopenmp,-march=native" \
         -Xlinker=--wrap=initGPUMemPool,--wrap=gpuMalloc
LIBS  := -lsytorch -lcryptoTools -lLLAMA -lbitpack -lcuda -lcudart -lcurand

# ── include / link paths ──────────────────────────────────────────────────────
UTIL_FILES   := $(GPU_MPC_ROOT)/utils/gpu_mem.cu \
                $(GPU_MPC_ROOT)/utils/gpu_file_utils.cpp \
                $(HERE)/utils/sigma_comms.cpp
OBJ_INCLUDES := -I '$(HERE)' \
                -I '$(GPU_MPC_ROOT)' \
                -I '$(GPU_MPC_ROOT)/utils' \
                -I '$(CUTLASS_PATH)/include' \
                -I '$(CUTLASS_PATH)/tools/util/include' \
                -I '$(SYTORCH_PATH)/include' \
                -I '$(LLAMA_PATH)/include' \
                -I '$(SYTORCH_PATH)/ext/cryptoTools'
INCLUDES     := $(OBJ_INCLUDES) \
                -L$(CUTLASS_PATH)/build/tools/library \
                -L$(SYTORCH_BUILD_PATH) \
                -L$(SYTORCH_BUILD_PATH)/ext/cryptoTools \
                -L$(SYTORCH_BUILD_PATH)/ext/llama \
                -L$(SYTORCH_BUILD_PATH)/ext/bitpack \
                -L$(SYTORCH_BUILD_PATH)/lib

# ── targets ───────────────────────────────────────────────────────────────────
# pool_override.o: replaces initGPUMemPool() to skip the 40GiB warmup probe
# that OOMs on RTX 4060 (8 GiB).  Compiled with plain gcc so the __wrap_
# symbol lands in the object as a standard C extern.
pool_override.o: pool_override.c
	gcc -O2 -fPIC -I/usr/local/cuda-$(CUDA_VERSION)/include \
	    -c $(HERE)/pool_override.c -o $(HERE)/pool_override.o

deepdtagen_inference: deepdtagen_inference.cu deepdtagen.h gcn_layer.h masked_maxpool.h secure_adj_norm.h ddg_orca.h ddg_orca_base.h ddg_orca_opt.h pool_override.o
	$(CXX) $(FLAGS) $(INCLUDES) $(HERE)/deepdtagen_inference.cu $(UTIL_FILES) \
	    $(HERE)/pool_override.o $(LIBS) \
	    -o $(HERE)/deepdtagen_inference

clean:
	rm -f $(HERE)/deepdtagen_inference $(HERE)/pool_override.o
