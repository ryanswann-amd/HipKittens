# HipKittens BF16 GEMM Library — MI300X (gfx942, CDNA3)
# Builds multiple tile-size kernels as separate pybind11 .so modules

GPU_TARGET=CDNA3

# ROCm
ROCM_INSTALL_DIR := $(shell [ -d /opt/rocm ] && echo /opt/rocm || echo /opt/rocm-7.2.0)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip
ROCM_INCLUDE_DIR := $(ROCM_INSTALL_DIR)/include
HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc

# Architecture flags
HIPFLAGS = -DKITTENS_CDNA3 --offload-arch=gfx942

# Common flags
CXX_STD   := c++20
PY_INCLUDES := $(shell python3 -m pybind11 --includes)
PY_LDFLAGS  := $(shell python3-config --ldflags)
PY_EXT      := $(shell python3-config --extension-suffix)

TORCH_INCLUDES := $(shell python3 -c "import torch; from pathlib import Path; p=Path(torch.__file__).parent; print(f'-I{p}/include -I{p}/include/torch/csrc/api/include')" 2>/dev/null)
TORCH_LDFLAGS  := $(shell python3 -c "import torch; from pathlib import Path; p=Path(torch.__file__).parent/'lib'; print(f'-L{p} -ltorch -ltorch_python -lc10 -lc10_hip')" 2>/dev/null)

CXXFLAGS = -std=$(CXX_STD) -w -shared -fPIC \
           -I${THUNDERKITTENS_ROOT}/include \
           -I${THUNDERKITTENS_ROOT}/prototype \
           -I$(ROCM_INCLUDE_DIR) -I$(HIP_INCLUDE_DIR) \
           $(PY_INCLUDES) $(PY_LDFLAGS) \
           $(TORCH_INCLUDES) $(TORCH_LDFLAGS) \
           -Rpass-analysis=kernel-resource-usage

# Tile configurations: name -> BLOCK_M, BLOCK_N, K_STEP, WARPS_M, WARPS_N
TILES = tile_256x256 tile_128x256 tile_256x128 tile_128x128 tile_64x64

# Target .so names
TARGETS = $(addsuffix $(PY_EXT), $(TILES))

all: $(TARGETS)

tile_256x256$(PY_EXT): gemm_kernel.cpp
	$(HIPCXX) $< $(HIPFLAGS) $(CXXFLAGS) \
		-DBLOCK_M=256 -DBLOCK_N=256 -DK_STEP_SIZE=16 -DWARPS_M=2 -DWARPS_N=4 \
		-DMODULE_NAME=tile_256x256 \
		-o $@ 2>&1

tile_128x256$(PY_EXT): gemm_kernel.cpp
	$(HIPCXX) $< $(HIPFLAGS) $(CXXFLAGS) \
		-DBLOCK_M=128 -DBLOCK_N=256 -DK_STEP_SIZE=64 -DWARPS_M=2 -DWARPS_N=4 \
		-DMODULE_NAME=tile_128x256 \
		-o $@ 2>&1

tile_256x128$(PY_EXT): gemm_kernel.cpp
	$(HIPCXX) $< $(HIPFLAGS) $(CXXFLAGS) \
		-DBLOCK_M=256 -DBLOCK_N=128 -DK_STEP_SIZE=64 -DWARPS_M=4 -DWARPS_N=2 \
		-DMODULE_NAME=tile_256x128 \
		-o $@ 2>&1

tile_128x128$(PY_EXT): gemm_kernel.cpp
	$(HIPCXX) $< $(HIPFLAGS) $(CXXFLAGS) \
		-DBLOCK_M=128 -DBLOCK_N=128 -DK_STEP_SIZE=16 -DWARPS_M=2 -DWARPS_N=4 \
		-DMODULE_NAME=tile_128x128 \
		-o $@ 2>&1

tile_64x64$(PY_EXT): gemm_kernel.cpp
	$(HIPCXX) $< $(HIPFLAGS) $(CXXFLAGS) \
		-DBLOCK_M=64 -DBLOCK_N=64 -DK_STEP_SIZE=32 -DWARPS_M=2 -DWARPS_N=2 \
		-DMODULE_NAME=tile_64x64 \
		-o $@ 2>&1

clean:
	rm -f *.so *.hip *.o

.PHONY: all clean

# CDNA3 branch optimized kernel (uses cdna3 branch headers)
CDNA3_HEADERS = /tmp/hk_cdna3
hk_gemm$(PY_EXT): cdna3_kernel.cpp
	$(HIPCXX) $< $(HIPFLAGS) -std=$(CXX_STD) -w -shared -fPIC \
		-I$(CDNA3_HEADERS)/include -I$(CDNA3_HEADERS)/prototype \
		-I$(ROCM_INCLUDE_DIR) -I$(HIP_INCLUDE_DIR) \
		$(PY_INCLUDES) $(PY_LDFLAGS) \
		-Rpass-analysis=kernel-resource-usage \
		-o $@ 2>&1

# CDNA3 optimized kernels (multiple tile sizes)
hk_128x128$(PY_EXT): cdna3_kernel.cpp
	$(HIPCXX) $< $(HIPFLAGS) -std=$(CXX_STD) -w -shared -fPIC \
		-I$(CDNA3_HEADERS)/include -I$(CDNA3_HEADERS)/prototype \
		-I$(ROCM_INCLUDE_DIR) -I$(HIP_INCLUDE_DIR) \
		$(PY_INCLUDES) $(PY_LDFLAGS) \
		-DBLOCK_SIZE_VAL=128 -DHK_MODULE_NAME=hk_128x128 \
		-o $@ 2>&1

hk_256x256$(PY_EXT): cdna3_kernel.cpp
	$(HIPCXX) $< $(HIPFLAGS) -std=$(CXX_STD) -w -shared -fPIC \
		-I$(CDNA3_HEADERS)/include -I$(CDNA3_HEADERS)/prototype \
		-I$(ROCM_INCLUDE_DIR) -I$(HIP_INCLUDE_DIR) \
		$(PY_INCLUDES) $(PY_LDFLAGS) \
		-DBLOCK_SIZE_VAL=256 -DHK_MODULE_NAME=hk_256x256 \
		-o $@ 2>&1

# CDNA3 NN (non-transposed) kernels — C = A @ B where B is KxN
CDNA3_NN_FLAGS = $(HIPFLAGS) -std=$(CXX_STD) -w -shared -fPIC \
	-I$(CDNA3_HEADERS)/include -I$(CDNA3_HEADERS)/prototype \
	-I$(ROCM_INCLUDE_DIR) -I$(HIP_INCLUDE_DIR) \
	$(PY_INCLUDES) $(PY_LDFLAGS) \
	-Rpass-analysis=kernel-resource-usage

hk_nn_256x256x64$(PY_EXT): nn_kernel.cpp
	$(HIPCXX) $< $(CDNA3_NN_FLAGS) \
		-DBLOCK_SIZE_VAL=256 -DK_STEP_VAL=64 -DHK_MODULE_NAME=hk_nn_256x256x64 \
		-o $@ 2>&1

hk_nn_128x128x64$(PY_EXT): nn_kernel.cpp
	$(HIPCXX) $< $(CDNA3_NN_FLAGS) \
		-DBLOCK_SIZE_VAL=128 -DK_STEP_VAL=64 -DHK_MODULE_NAME=hk_nn_128x128x64 \
		-o $@ 2>&1

hk_nn_192x192x64$(PY_EXT): nn_kernel.cpp
	$(HIPCXX) $< $(CDNA3_NN_FLAGS) \
		-DBLOCK_SIZE_VAL=192 -DK_STEP_VAL=64 -DHK_MODULE_NAME=hk_nn_192x192x64 \
		-o $@ 2>&1

nn: hk_nn_256x256x64$(PY_EXT) hk_nn_128x128x64$(PY_EXT) hk_nn_192x192x64$(PY_EXT)

# FP16 NT kernels — same structure as BF16 NT but with half type
FP16_FLAGS = $(HIPFLAGS) -std=$(CXX_STD) -w -shared -fPIC \
	-I$(CDNA3_HEADERS)/include -I$(CDNA3_HEADERS)/prototype \
	-I$(ROCM_INCLUDE_DIR) -I$(HIP_INCLUDE_DIR) \
	$(PY_INCLUDES) $(PY_LDFLAGS) \
	-Rpass-analysis=kernel-resource-usage

hk_fp16_nt_256x256x64$(PY_EXT): fp16_nt_kernel.cpp
	$(HIPCXX) $< $(FP16_FLAGS) \
		-DBLOCK_SIZE_VAL=256 -DK_STEP_VAL=64 -DHK_MODULE_NAME=hk_fp16_nt_256x256x64 \
		-o $@ 2>&1

hk_fp16_nt_128x128x64$(PY_EXT): fp16_nt_kernel.cpp
	$(HIPCXX) $< $(FP16_FLAGS) \
		-DBLOCK_SIZE_VAL=128 -DK_STEP_VAL=64 -DHK_MODULE_NAME=hk_fp16_nt_128x128x64 \
		-o $@ 2>&1

fp16_nt: hk_fp16_nt_256x256x64$(PY_EXT) hk_fp16_nt_128x128x64$(PY_EXT)

# FP16 NN kernels — same transposed B load as BF16 NN
hk_fp16_nn_128x128x64$(PY_EXT): fp16_nn_kernel.cpp
	$(HIPCXX) $< $(FP16_FLAGS) \
		-DBLOCK_SIZE_VAL=128 -DK_STEP_VAL=64 -DHK_MODULE_NAME=hk_fp16_nn_128x128x64 \
		-o $@ 2>&1

hk_fp16_nn_192x192x64$(PY_EXT): fp16_nn_kernel.cpp
	$(HIPCXX) $< $(FP16_FLAGS) \
		-DBLOCK_SIZE_VAL=192 -DK_STEP_VAL=64 -DHK_MODULE_NAME=hk_fp16_nn_192x192x64 \
		-o $@ 2>&1

fp16_nn: hk_fp16_nn_128x128x64$(PY_EXT) hk_fp16_nn_192x192x64$(PY_EXT)

# BF16 TT kernels — transposed A load, normal B load
hk_tt_128x128x64$(PY_EXT): tt_kernel.cpp
	$(HIPCXX) $< $(CDNA3_NN_FLAGS) \
		-DBLOCK_SIZE_VAL=128 -DK_STEP_VAL=64 -DHK_MODULE_NAME=hk_tt_128x128x64 \
		-o $@ 2>&1

hk_tt_192x192x64$(PY_EXT): tt_kernel.cpp
	$(HIPCXX) $< $(CDNA3_NN_FLAGS) \
		-DBLOCK_SIZE_VAL=192 -DK_STEP_VAL=64 -DHK_MODULE_NAME=hk_tt_192x192x64 \
		-o $@ 2>&1

tt: hk_tt_128x128x64$(PY_EXT) hk_tt_192x192x64$(PY_EXT)

# BF16 TN kernels — both A and B transposed
hk_tn_128x128x64$(PY_EXT): tn_kernel.cpp
	$(HIPCXX) $< $(CDNA3_NN_FLAGS) \
		-DBLOCK_SIZE_VAL=128 -DK_STEP_VAL=64 -DHK_MODULE_NAME=hk_tn_128x128x64 \
		-o $@ 2>&1

tn: hk_tn_128x128x64$(PY_EXT)
