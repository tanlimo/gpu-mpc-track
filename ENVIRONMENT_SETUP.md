# Environment Setup

This document describes the complete environment configuration
used for DeepDTAGen GPU-MPC deployment and evaluation.

The content is extracted from the original README environment
setup section.


---

## 1 Validated development environment

The current code has been successfully built and tested with:

```text
OS                  Ubuntu 22.04
Python              3.12.3
GCC / G++           11.4
CUDA Toolkit        12.8

PyTorch             2.8.0+cu128
PyTorch CUDA        12.8
NumPy               2.3.2
pandas              3.0.5
torch_geometric     2.8.0.post1
RDKit               2026.03.5

Development GPU     NVIDIA H800 PCIe 80 GB
```

Validated EzPC revision:

```text
13592590466bbe19fa6f13384e3e896a0a4323b5
```

The same commit is used as the project `GPU-MPC` dependency.

---

## 2 Check the base system

Run:

```bash
nvidia-smi

nvcc --version

gcc --version
g++ --version

cmake --version

python3 --version
```

Verify PyTorch CUDA access:

```bash
python3 - <<'PY'
import torch

print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu =", torch.cuda.get_device_name(0))
PY
```

Expected:

```text
cuda available = True
```

---

## 3 Basic system packages

Install missing packages when necessary:

```bash
apt-get update

apt-get install -y \
  git \
  cmake \
  build-essential \
  libssl-dev \
  libgmp-dev \
  libomp-dev
```

Skip packages already present in the target environment.

---

## 4 Create a workspace

Choose a writable workspace. The filesystem used for generated FSS keys must
have sufficient free space.

For example:

```bash
export WORKSPACE="$HOME/idash-track3"
export DDG_WORK="$WORKSPACE/runtime"

mkdir -p "$WORKSPACE"
mkdir -p "$DDG_WORK"

cd "$WORKSPACE"

df -h "$WORKSPACE" "$DDG_WORK"
```

`DDG_WORK` is the default location used by the examples in Sections 5–8 for
prepared datasets, serialized model weights, and generated FSS keys.

If `$WORKSPACE` is located on a small container overlay, set `DDG_WORK` to a
larger mounted writable filesystem instead.

---

## 5 Obtain this repository

```bash
cd "$WORKSPACE"

git clone https://github.com/tanlimo/gpu-mpc-track.git

cd gpu-mpc-track
```

During development:

```bash
git checkout compliance-timing-v1
```

Verify:

```bash
git status
git log -5 --oneline
```

For the final submitted package, use the submission branch/archive instead of relying on a development branch.

---

## 6 Obtain EzPC/GPU-MPC

Clone EzPC:

```bash
cd "$WORKSPACE"

git clone https://github.com/mpc-msri/EzPC.git
```

Checkout the currently validated revision and initialize all required
submodules:

```bash
cd "$WORKSPACE/EzPC"

git checkout 13592590466bbe19fa6f13384e3e896a0a4323b5

git submodule update --init --recursive
```

Verify the EzPC revision:

```bash
git rev-parse HEAD
```

Expected:

```text
13592590466bbe19fa6f13384e3e896a0a4323b5
```

Optionally inspect submodule state:

```bash
git submodule status --recursive
```

Set:

```bash
export GPU_MPC_ROOT="$WORKSPACE/EzPC/GPU-MPC"
```

Verify:

```bash
test -d "$GPU_MPC_ROOT" \
  && echo "GPU_MPC_ROOT: PASS" \
  || echo "GPU_MPC_ROOT: FAIL"
```

---

## 7 Configure CUDA

The following environment variables must be configured in every new shell
before building or running the MPC programs.

Find the installed CUDA toolkit:

```bash
ls -d /usr/local/cuda*
```

The validated CUDA 12.8 configuration is:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_VERSION=12.8

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export NVCC_PATH="$CUDA_HOME/bin/nvcc"

export GPU_MPC_ROOT="$WORKSPACE/EzPC/GPU-MPC"
```

Verify:

```bash
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VERSION=$CUDA_VERSION"
echo "NVCC_PATH=$NVCC_PATH"
echo "GPU_MPC_ROOT=$GPU_MPC_ROOT"

which nvcc
nvcc --version

test -d "$GPU_MPC_ROOT" \
  && echo "GPU_MPC_ROOT: PASS" \
  || echo "GPU_MPC_ROOT: FAIL"
```

These exports are shell-local. If a new shell or container terminal is opened,
configure them again before building or running the MPC programs.
---

## 8 Python dependencies

Inspect installed modules first:

```bash
python3 - <<'PY'
mods = [
    "torch",
    "numpy",
    "pandas",
    "torch_geometric",
    "rdkit",
]

for name in mods:
    try:
        mod = __import__(name)
        print(
            name,
            getattr(mod, "__version__", "OK")
        )
    except Exception as exc:
        print(name, "MISSING:", exc)
PY
```

Install missing non-PyTorch dependencies when necessary:

```bash
python3 -m pip install \
  numpy \
  pandas \
  torch-geometric \
  rdkit
```

PyTorch should match the CUDA environment.

If a CUDA-compatible PyTorch installation is already included in the base environment, prefer keeping the provided build instead of reinstalling it unnecessarily.

Verify:

```bash
python3 - <<'PY'
import sys
import torch
import numpy
import pandas
import torch_geometric
import rdkit

print("python =", sys.version.split()[0])
print("torch =", torch.__version__)
print("torch cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
print("numpy =", numpy.__version__)
print("pandas =", pandas.__version__)
print("torch_geometric =", torch_geometric.__version__)
print("rdkit =", rdkit.__version__)
PY
```

The validated development environment reports:

```text
python = 3.12.3
torch = 2.8.0+cu128
torch cuda = 12.8
numpy = 2.3.2
pandas = 3.0.5
torch_geometric = 2.8.0.post1
rdkit = 2026.03.5
```

---

## 9 Sytorch / SCI compatibility

The current EzPC/GPU-MPC dependency includes Sytorch/SCI components.

On the validated CUDA 12.8 + GCC 11 environment, the bundled SEAL source required an explicit `<mutex>` include.

Define:

```bash
LOCKS="$GPU_MPC_ROOT/ext/sytorch/ext/sci/extern/SEAL/native/src/seal/util/locks.h"
```

Apply only when it is missing:

```bash
grep -q '^#include <mutex>$' "$LOCKS" || \
  sed -i '/#include <shared_mutex>/a #include <mutex>' "$LOCKS"
```

Verify:

```bash
grep -n -A3 -B2 "shared_mutex" "$LOCKS" | head -10
```

The working source contains both:

```cpp
#include <shared_mutex>
#include <mutex>
```

Also check:

```bash
grep SEAL_POLY_MOD_DEGREE_MAX \
  "$GPU_MPC_ROOT/ext/sytorch/ext/sci/extern/SEAL/native/src/seal/util/defines.h"
```

The validated environment uses:

```text
SEAL_POLY_MOD_DEGREE_MAX 65536
```

Configure Sytorch:

```bash
cd "$GPU_MPC_ROOT/ext/sytorch"

rm -rf build

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/install" \
  -DCUDAToolkit_ROOT="$CUDA_HOME"
```

Build:

```bash
cmake --build build -j"$(nproc)"
```

If the target environment already contains a compatible built dependency, rebuilding may not be necessary.

---

## 10 Build DeepDTAGen MPC

Return to the repository:

```bash
cd "$WORKSPACE/gpu-mpc-track"
```

Build the current 64-bit configuration:

```bash
cd gpu_mpc

make \
  GPU_MPC_ROOT="$GPU_MPC_ROOT" \
  BW=64 \
  GPU_ARCH=90a \
  deepdtagen_inference

BUILD_RC=$?

cd ..

echo "BUILD_RC=$BUILD_RC"
```

Expected:

```text
BUILD_RC=0
```

`GPU_ARCH=90a` is used for the tested Hopper-class H800/H100 environment.

Adjust it when using a different GPU architecture.

---

## 11 Required external files

The following large files may not be stored in Git.

### Model checkpoint

Place under:

```text
model/
```

For example:

```text
model/deepdtagen_model_kiba.pth
```

### Raw dataset

Place under:

```text
data/
```

For example:

```text
data/new_test.csv
```

Absolute paths may also be supplied to the preparation scripts.

---

## 12 Final environment smoke test

A fresh environment is considered successfully configured only after the following full chain works:

```text
prepare dataset
      ↓
prepare model
      ↓
generate offline keys
      ↓
run local two-party online inference
      ↓
AFFINITY_GLOBAL
      ↓
PASS
```

Use Sections 5–8 for the exact commands.

---

## 13 Common setup problems

### `nvcc` not found

```bash
ls -d /usr/local/cuda*

export CUDA_HOME=/usr/local/cuda-XX.Y
export PATH="$CUDA_HOME/bin:$PATH"
```

---

### Wrong CUDA selected

```bash
which nvcc
nvcc --version
echo "$CUDA_HOME"
```

---

### PyTorch does not see CUDA

```bash
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

---

### Invalid `GPU_MPC_ROOT`

```bash
echo "$GPU_MPC_ROOT"

test -d "$GPU_MPC_ROOT" \
  && echo PASS \
  || echo FAIL
```

---

### SEAL `shared_mutex` / locking build error

Check:

```bash
grep -n -A3 -B2 "shared_mutex" \
  "$GPU_MPC_ROOT/ext/sytorch/ext/sci/extern/SEAL/native/src/seal/util/locks.h"
```

The validated source contains:

```cpp
#include <shared_mutex>
#include <mutex>
```

---

### GPU architecture error

Use a `GPU_ARCH` supported by both the installed CUDA toolkit and the target GPU.

For the current H800/H100 configuration:

```text
GPU_ARCH=90a
```

---