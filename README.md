# GPU-Accelerated 2PC Inference for DeepDTAGen

GPU-accelerated two-party secure inference for the **drug-target affinity prediction branch of DeepDTAGen**, developed for **iDASH Privacy & Security Workshop 2026 — Track 3: Accelerating MPC-Based Deep Learning for Drug-Target Interaction Prediction**.

The current implementation protects the **drug input** using two-party additive secret sharing, treats the **protein sequence and model parameters as public**, and outputs a continuous drug-target affinity prediction.

Current production candidate:

```text
MPC parties          2
Offline Dealer       semi-honest trusted setup
Ring                 64 bit
Fixed-point scale    12
Drug                 private
Protein              public
Model parameters     public
A_norm               securely computed online
FSS keys             generated offline
Evaluator key mode   direct full-RAM preload
```

> **Current compliance status: `PRE_COMPLIANCE`**
>
> Preprocessing, Dealer generation, key preload, setup, online computation, and end-to-end timing are reported separately. The final competition treatment of model-specific FSS preprocessing should be confirmed with the organizers.

---

# 0. Quick Start

The complete execution pipeline is:

```text
Raw dataset
    │
    ▼
Dataset preparation
    │
    ├── private drug → additive shares
    └── public protein → target_ids.dat
    │
    ▼
Model preparation
    │
    ├── weights.bin for MPC drug/fusion path
    └── original .pth for public Protein Gated-CNN
    │
    ▼
Trusted Offline Dealer
    │
    ├── K0
    └── K1
    │
    └──────────── Dealer exits ────────────
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     Online Server 0     Online Server 1
       share0 + K0         share1 + K1
          │                   │
          └──── GPU-MPC ──────┘
                    │
                    ▼
           affinity prediction
```

A typical local workflow is shown below.

First, choose a writable filesystem with sufficient capacity for generated
datasets, model binaries, and especially FSS key files:

```bash
export DDG_WORK="$WORKSPACE/runtime"

mkdir -p "$DDG_WORK"

df -h "$DDG_WORK"
```

FSS key files can be several GB per internal micro-batch. Do not place large
key directories on a small container overlay or `/tmp` unless sufficient free
space has first been verified with `df -h`.

```bash
# ------------------------------------------------------------
# 1. Prepare dataset
# ------------------------------------------------------------

python3 reference/prepare_dataset.py \
  --csv data/davis_test.csv \
  --output "$DDG_WORK/davis3_prepare" \
  --bw 64 \
  --scale 12 \
  --limit 3


# ------------------------------------------------------------
# 2. Prepare model
# ------------------------------------------------------------

python3 reference/prepare_model.py \
  --checkpoint model/deepdtagen_model_kiba.pth \
  --output "$DDG_WORK/kiba_model" \
  --scale 12


# ------------------------------------------------------------
# 3. Generate complete offline FSS keys
# ------------------------------------------------------------

python3 gpu_mpc/run_offline_dealer.py \
  "$DDG_WORK/davis3_prepare" \
  "$DDG_WORK/kiba_model/weights.bin" \
  "$DDG_WORK/davis3_keys" \
  --num-samples 3 \
  --micro-batch 8 \
  --bw 64 \
  --scale 12 \
  --gpu 0


# ------------------------------------------------------------
# 4. Run local two-party online inference
# ------------------------------------------------------------

python3 gpu_mpc/run_offline_online_local.py \
  "$DDG_WORK/davis3_prepare" \
  "$DDG_WORK/kiba_model/weights.bin" \
  "$DDG_WORK/davis3_keys" \
  --protein-checkpoint model/deepdtagen_model_kiba.pth \
  --num-samples 3 \
  --micro-batch 8 \
  --bw 64 \
  --scale 12 \
  --gpu0 0 \
  --gpu1 1 \
  --ip 127.0.0.1 \
  --full-key-ram
```

A successful run should end with output similar to:

```text
AFFINITY_GLOBAL[0]=...
AFFINITY_GLOBAL[1]=...
AFFINITY_GLOBAL[2]=...

PASS: N=3, fixed B=8, chunks=1, padded=8, returned=3
OFFLINE/ONLINE SEPARATION: PASS
```

---

# 1. Project Overview

## 1.1 Task

This repository implements the **affinity prediction branch** of DeepDTAGen.

The molecule-generation branch is not part of the MPC inference path.

The model returns a continuous affinity value:

```text
drug + protein
      │
      ▼
DeepDTAGen affinity branch
      │
      ▼
continuous affinity prediction
```

The implemented network contains:

```text
Drug:
    GCN ×3
    → masked/global max pooling
    → Drug FC
    → 128-D vector

Protein:
    public Gated-CNN
    → 128-D vector

Fusion:
    concat(128,128)
    → FC stack
    → affinity
```

---

## 1.2 Privacy model

| Value | Privacy |
|---|---|
| Drug molecule | Private |
| Drug node features | Private |
| Drug adjacency | Private |
| Drug pooling mask | Private |
| Protein sequence | Public |
| Model parameters | Public |
| MPC intermediate values | Private |
| Final affinity | Output |

The private drug representation is additively secret-shared between two non-colluding online parties.

The two online servers must not receive both shares of the private drug input.

---

## 1.3 Dealer and online-party architecture

The current system contains one logical trusted Offline Dealer and two online MPC Evaluators.

```text
                         OFFLINE

                    Trusted Dealer
                         │
                generate FSS material
                         │
                  ┌──────┴──────┐
                  │             │
                 K0            K1
                  │             │
                  ▼             ▼
              Server P0     Server P1

                    Dealer exits


                         ONLINE

          Server P0                    Server P1
       ───────────────               ───────────────
       private share0                private share1
       FSS key K0                    FSS key K1
       public weights                public weights
       public protein                public protein

                  │                       │
                  └──── GPU-MPC / 2PC ────┘
                              │
                              ▼
                     affinity prediction
```

The Dealer is **not** a third online MPC participant.

Possible Dealer deployments include:

- a temporary trusted third machine;
- a trusted preprocessing machine;
- a data-owner-controlled trusted setup environment.

After generating and distributing `K0` and `K1`, the Dealer exits.

The intended online data separation is:

```text
Server P0:
    drug share0
    K0

Server P1:
    drug share1
    K1
```

---

# 2. Method

## 2.1 Drug branch

The private drug graph is processed using GPU-MPC.

```text
Node features: 94
      │
      ▼
GCN 94 → 188
      │
    ReLU
      │
      ▼
GCN 188 → 282
      │
    ReLU
      │
      ▼
GCN 282 → 376
      │
    ReLU
      │
      ▼
Masked / global max pooling
      │
      ▼
FC 376 → 1024
      │
    ReLU
      │
      ▼
FC 1024 → 128
```

---

## 2.2 Protein branch

The protein sequence is public.

The original DeepDTAGen Gated-CNN is therefore executed using ordinary FP32 GPU computation.

```text
Protein token sequence [1000]
        │
        ▼
Embedding
        │
        ▼
Conv / Gate 1
        │
        ▼
Conv / Gate 2
        │
        ▼
Conv / Gate 3
        │
        ▼
Flatten
        │
        ▼
FC → 128
```

The original `.pth` checkpoint remains required for this branch.

No `protein_emb.dat` is precomputed during dataset preparation.

---

## 2.3 Fusion branch

```text
Drug vector       Protein vector
    128                128
      \                /
       \              /
        ▼            ▼
          Concat 256
              │
              ▼
        FC 256 → 1024
              │
            ReLU
              │
              ▼
       FC 1024 → 512
              │
            ReLU
              │
              ▼
        FC 512 → 256
              │
            ReLU
              │
              ▼
          FC 256 → 1
              │
              ▼
           affinity
```

---

## 2.4 Secure adjacency normalization

Dataset preparation outputs the raw binary adjacency matrix with self-loops.

It does **not** precompute normalized adjacency.

The online MPC protocol computes:

```text
A_raw
  │
  ▼
degree D
  │
  ▼
D^(-1/2)
  │
  ▼
A_norm = D^(-1/2) A D^(-1/2)
```

Therefore the graph degree and normalization factors are derived from secret-shared adjacency during inference.

---

## 2.5 Fixed-point configuration

The current validated configuration is:

```text
BW    = 64
SCALE = 12
```

`BW=64, SCALE=12` should be used unless a different configuration has been separately validated for accuracy.

---

## 2.6 Offline FSS keys and full-RAM Evaluator

The current fallback generates the complete FSS key stream before online inference.

```text
Offline Dealer:
    generate complete K0
    generate complete K1
           │
           ▼
     persistent key files

Online Evaluator setup:
    K0/K1 → final FSS RAM buffer
           │
           ▼
        READY
           │
           ▼
        START
           │
           ▼
     MPC inference
```

In `--full-key-ram` mode:

- the complete key file is loaded directly into the final FSS host buffer;
- no second full-size RAM copy is created;
- no per-chunk SSD key read occurs during MPC execution;
- micro-batches access different regions through pointer offsets.

---

# 3. Repository Structure

The main production-relevant files are:

```text
gpu-mpc-track/
│
├── README.md
│
├── gpu_mpc/
│   ├── deepdtagen_inference.cu
│   │     Main Dealer / Evaluator CUDA program
│   │
│   ├── deepdtagen.h
│   │     DeepDTAGen affinity graph
│   │
│   ├── gcn_layer.h
│   ├── masked_maxpool.h
│   ├── secure_adj_norm.h
│   │     Secure drug graph operations
│   │
│   ├── ddg_orca.h
│   ├── ddg_orca_base.h
│   ├── ddg_orca_batched.h
│   │     DeepDTAGen GPU-MPC backend
│   │
│   ├── run_offline_dealer.py
│   │     Complete offline key generation
│   │
│   ├── run_offline_online_local.py
│   │     Local two-GPU online runner
│   │
│   └── Makefile
│
├── reference/
│   ├── prepare_dataset.py
│   │     External dataset → MPC input format
│   │
│   ├── prepare_model.py
│   │     .pth → weights.bin
│   │
│   ├── run_protein_chunks.py
│   │     Public Protein Gated-CNN
│   │
│   ├── dense_graph.py
│   └── ...
│
├── model/
│   └── *.pth
│
├── data/
│   └── datasets
│
└── ...
```

Large model files, datasets, generated shares, and FSS keys may be intentionally excluded from Git.

---

# 4. Environment Setup

This section is intended as a **fresh-environment recovery guide**.

Its purpose is to make it possible to enter a new Linux GPU environment or future competition container, configure all project dependencies, build the code, and finish with a successful minimal inference test.

---

## 4.1 Validated development environment

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

## 4.2 Check the base system

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

## 4.3 Basic system packages

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

## 4.4 Create a workspace

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

## 4.5 Obtain this repository

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

## 4.6 Obtain EzPC/GPU-MPC

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

## 4.7 Configure CUDA

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

## 4.8 Python dependencies

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

## 4.9 Sytorch / SCI compatibility

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

## 4.10 Build DeepDTAGen MPC

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

## 4.11 Required external files

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

## 4.12 Final environment smoke test

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

## 4.13 Common setup problems

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

# 5. Dataset Preparation

## 5.1 External input format

The preferred external dataset interface is CSV.

By default, each row contains:

| Column | Required | Meaning |
|---|---|---|
| `compound_iso_smiles` | yes | drug SMILES |
| `target_sequence` | yes | protein amino-acid sequence |
| `affinity` | optional | ground-truth affinity |

Example:

```csv
compound_iso_smiles,target_sequence,affinity
CCO,MKT...,10.21
CCN,MSA...,11.04
```

The `affinity` column is not consumed by MPC inference and may be absent on hidden evaluation data.

Alternative column names may be supplied using:

```text
--smiles-column
--protein-column
--affinity-column
```

---

## 5.2 Prepare a new dataset

```bash
python3 reference/prepare_dataset.py \
  --csv data/new_test.csv \
  --output "$DDG_WORK/new_test_prepared" \
  --bw 64 \
  --scale 12
```

For a small smoke test:

```bash
python3 reference/prepare_dataset.py \
  --csv data/new_test.csv \
  --output "$DDG_WORK/new_test_prepared" \
  --bw 64 \
  --scale 12 \
  --limit 3
```

Show all options with:

```bash
python3 reference/prepare_dataset.py --help
```

---

## 5.3 Prepared private drug format

For each sample:

### Node features

```text
shape      [138, 94]
scale      12
privacy    secret-shared
```

### Raw adjacency

```text
shape      [138, 138]
values     {0,1}
self-loop  included for real atoms
scale      0
privacy    secret-shared
```

### Pooling mask

```text
shape      [138, 376]
scale      0
privacy    secret-shared
```

The adjacency is intentionally stored as raw binary adjacency.

`A_norm` is not precomputed.

---

## 5.4 Public protein format

The protein sequence is encoded using the DeepDTAGen sequence mapping and padded/truncated to:

```text
length = 1000
```

It is written to:

```text
target_ids.dat
```

as little-endian `int64`.

Dataset preparation does not run the Protein Gated-CNN.

---

## 5.5 Prepared directory

```text
new_test_prepared/
├── x_share0.dat
├── x_share1.dat
├── adj_share0.dat
├── adj_share1.dat
├── mask_share0.dat
├── mask_share1.dat
├── target_ids.dat
├── metadata.json
└── affinity.npy       # only if labels exist
```

Online deployment:

```text
Server P0:
    x_share0.dat
    adj_share0.dat
    mask_share0.dat

Server P1:
    x_share1.dat
    adj_share1.dat
    mask_share1.dat

Both:
    target_ids.dat
```

---

## 5.6 Expected output

A successful preparation prints:

```text
PREPARE DATASET: PASS
samples = ...
output = ...
protein = public target_ids.dat; NO protein_emb.dat
adj = raw binary + self-loops, scale=0
```

Inspect:

```bash
cat "$DDG_WORK/new_test_prepared/metadata.json"
```

---

## 5.7 Secret-sharing randomness

By default, the production preparation path uses OS cryptographic randomness for additive secret splitting.

The option:

```text
--deterministic-seed
```

is intended only for debugging and regression tests.

It should not be used for production secret sharing.

---

# 6. Model Preparation

## 6.1 Model execution split

The original `.pth` checkpoint serves two purposes:

```text
Original .pth
    │
    ├── public Protein Gated-CNN
    │
    └── model preparation
            │
            ▼
        weights.bin
            │
            ▼
       MPC model path
```

Therefore both are required:

```text
original .pth
weights.bin
```

---

## 6.2 Prepare weights

```bash
python3 reference/prepare_model.py \
  --checkpoint model/deepdtagen_model_kiba.pth \
  --output "$DDG_WORK/kiba_model_prepared" \
  --scale 12
```

Generated files:

```text
$DDG_WORK/kiba_model_prepared/
├── weights.bin
├── weights.bin.json
└── model_metadata.json
```

The MPC binary contains parameters for:

```text
GCN ×3
Drug FC ×2
Fusion FC ×4
```

for 9 MPC parameter groups in total.

---

## 6.3 Expected output

```text
PREPARE MODEL: PASS
checkpoint = ...
weights = ...
scale = 12
MPC layers = 9
Protein = original .pth cnn.* (timed FP32 GPU)
```

Inspect:

```bash
cat "$DDG_WORK/kiba_model_prepared/model_metadata.json"
```

---

## 6.4 Replacing model weights

For a compatible replacement checkpoint:

```bash
python3 reference/prepare_model.py \
  --checkpoint model/new_model.pth \
  --output "$DDG_WORK/new_model_prepared" \
  --scale 12
```

Then use:

```text
$DDG_WORK/new_model_prepared/weights.bin
```

for the MPC path, while passing:

```text
model/new_model.pth
```

to the public protein runner.

A compatible replacement checkpoint does not require C++ source modification.

A checkpoint with a different network architecture is not automatically compatible.

---

# 7. Offline Dealer

## 7.1 Purpose

The trusted Offline Dealer generates the complete correlated FSS key streams
required by the two online Evaluators.

```text
prepared private shares + public model
                │
                ▼
         trusted Offline Dealer
                │
          ┌─────┴─────┐
          ▼           ▼
         K0          K1
          │           │
          ▼           ▼
      Server P0   Server P1

          Dealer exits
```

The current implementation uses **one logical Dealer**.

The Dealer generates the two party-specific key streams sequentially:

```text
Dealer:
    generate K0
        ↓
    generate K1
        ↓
    exit
```

The Dealer script does not start either online Evaluator.

---

## 7.2 Storage requirement

FSS key files are large.

For the current configuration:

```text
BW     = 64
SCALE  = 12
B      = 8
A_norm = secure online
```

one FSS key chunk occupies:

```text
2,954,940,416 bytes / party
```

which is approximately:

```text
2.955 GB / party
2.752 GiB / party
```

Therefore a single B8 chunk requires approximately:

```text
K0 + K1 ≈ 5.91 GB
```

of persistent storage across the two key files.

Before generating keys, verify the selected filesystem:

```bash
df -h "$DDG_WORK"
```

Do not place large FSS key directories on a small container overlay or `/tmp`
unless sufficient free space has explicitly been verified.

For example:

```bash
export DDG_WORK="$WORKSPACE/runtime"

mkdir -p "$DDG_WORK"

df -h "$DDG_WORK"
```

The filesystem containing `$DDG_WORK` must have enough capacity for both party
key files.

---

## 7.3 Generate complete offline keys

A validated example for:

```text
logical N   = 3
micro-batch = 8
```

is:

```bash
python3 gpu_mpc/run_offline_dealer.py \
  "$DDG_WORK/davis3_prepare" \
  "$DDG_WORK/kiba_model/weights.bin" \
  "$DDG_WORK/davis3_keys" \
  --num-samples 3 \
  --micro-batch 8 \
  --bw 64 \
  --scale 12 \
  --gpu 0
```

The general form is:

```bash
python3 gpu_mpc/run_offline_dealer.py \
  PREPARED_DATASET_DIR \
  PREPARED_WEIGHTS_BIN \
  OUTPUT_KEY_DIR \
  --num-samples N \
  --micro-batch B \
  --bw 64 \
  --scale 12 \
  --gpu GPU_ID
```

Show all available options with:

```bash
python3 gpu_mpc/run_offline_dealer.py --help
```

Important options include:

```text
--num-samples
--micro-batch
--bw
--scale
--gpu
--keybuf-cap-gb
--force
--keep-work
```

The option:

```text
--legacy-precomputed-adj
```

is intended only for debug/regression experiments.

The default path generates key material for secure online computation of:

```text
A_raw
  ↓
degree
  ↓
D^(-1/2)
  ↓
A_norm
```

---

## 7.4 Output files

For `BW=64`, `SCALE=12`, the generated directory has the form:

```text
davis3_keys/
├── DeepDTAGen_64_12_party0_inference_key0.dat
├── DeepDTAGen_64_12_party1_inference_key1.dat
├── metadata.json
└── logs/
    ├── dealer_keygen_k0.log
    └── dealer_keygen_k1.log
```

For one B8 chunk:

```text
K0 = 2,954,940,416 bytes
K1 = 2,954,940,416 bytes
```

The metadata records fields including:

```text
samples
micro_batch
chunks
padded_samples
bw
scale
secure_adj_norm
key_chunk_bytes_per_party
key_total_bytes_per_party
logical_dealer_count
online_evaluator_started
status
```

For the validated N=3 example:

```text
samples                    = 3
micro_batch                = 8
chunks                     = 1
padded_samples             = 5
bw                         = 64
scale                      = 12
secure_adj_norm            = true
logical_dealer_count       = 1
online_evaluator_started   = false
status                     = PRE_COMPLIANCE
```

---

## 7.5 Fixed micro-batches

Logical dataset size `N` and internal MPC micro-batch size `B` are separate.

For example:

```text
N = 17
B = 8
```

produces:

```text
chunk 0: 8 real samples
chunk 1: 8 real samples
chunk 2: 1 real sample + 7 padding samples
```

The number of FSS chunks is:

```text
chunks = ceil(N / B)
```

For the current B8 computation graph:

```text
key bytes / party
=
ceil(N / 8) × 2,954,940,416
```

For example:

| Logical N | B8 chunks | Approx. key size / party |
|---:|---:|---:|
| 8 | 1 | 2.95 GB |
| 16 | 2 | 5.91 GB |
| 128 | 16 | 47.28 GB |
| 512 | 64 | 189.12 GB |

---

## 7.6 Key distribution

After generation:

```text
K0 → Server P0 only
K1 → Server P1 only
```

The intended online separation is:

```text
Server P0:
    private share0
    K0

Server P1:
    private share1
    K1
```

The Dealer terminates before online inference and is not an online MPC
participant.

The intended lifecycle is therefore:

```text
OFFLINE

Trusted Dealer
    │
    ├── generate K0
    ├── generate K1
    ├── distribute K0 → P0
    ├── distribute K1 → P1
    │
    └── terminate


ONLINE

Server P0  <──── GPU-MPC / 2PC ────>  Server P1
share0 + K0                           share1 + K1
```

---

## 7.7 Validated Dealer smoke test

The current implementation has been validated with:

```text
N      = 3
B      = 8
BW     = 64
SCALE  = 12
A_norm = secure online
```

A validated run produced:

```text
[offline-dealer] generate K0: chunks=1 B=8 GPU=0
[offline-dealer] K0 complete: 6.120873 s

[offline-dealer] generate K1: chunks=1 B=8 GPU=0
[offline-dealer] K1 complete: 6.121321 s

key bytes/chunk/party = 2,954,940,416
key bytes/party       = 2,954,940,416
total K0+K1 bytes     = 5,909,880,832

OFFLINE DEALER: PASS
Dealer lifecycle complete; no online evaluator was started.
```

The same run reported approximately:

```text
K0 generation wall = 6.121 s
K1 generation wall = 6.121 s
Dealer total wall  = 12.243 s
```

These values are development-machine smoke-test measurements and are not final
competition performance results.

---

## 7.8 Timing and compliance note

Dealer execution is reported explicitly:

```text
[DDG_TIME][OFFLINE_INPUT_FORMAT]
[DDG_TIME][OFFLINE_DEALER_KEYGEN]
[DDG_TIME][OFFLINE_DEALER_TOTAL]
[DDG_TIME][OFFLINE_KEY_DISTRIBUTION]
```

The current implementation retains:

```text
status=PRE_COMPLIANCE
```

because the final competition timing treatment of model-specific correlated
FSS preprocessing should be confirmed with the organizers.

This README does not assume that model-specific Dealer preprocessing is
automatically excluded from the final competition runtime metric.
---

# 8. Running Inference

## 8.1 Single-machine two-party run

The currently validated local topology uses two GPUs on one physical host:

```text
Evaluator P0 → GPU 0
Evaluator P1 → GPU 1
peer IP      → 127.0.0.1
peer port    → 42003
```

Run:

```bash
python3 gpu_mpc/run_offline_online_local.py \
  "$DDG_WORK/davis3_prepare" \
  "$DDG_WORK/kiba_model/weights.bin" \
  "$DDG_WORK/davis3_keys" \
  --protein-checkpoint model/deepdtagen_model_kiba.pth \
  --num-samples 3 \
  --micro-batch 8 \
  --bw 64 \
  --scale 12 \
  --gpu0 0 \
  --gpu1 1 \
  --ip 127.0.0.1 \
  --full-key-ram
```

For debugging:

```text
--keep-work
```

retains temporary fixed-micro-batch inputs and evaluator logs.
---

## 8.2 One-sample smoke test

The external logical sample count may be smaller than the MPC micro-batch.

Example:

```text
N = 1
B = 8
```

internally becomes:

```text
1 real sample
7 padding samples
```

but only one logical affinity result is returned.

One complete B8 FSS key chunk is still required.

---

## 8.3 Full-RAM lifecycle

With:

```text
--full-key-ram
```

the Evaluator lifecycle is:

```text
SETUP
    │
    ├── start P0/P1
    ├── load complete K0/K1
    ├── initialize backend
    ├── connect MPC peer
    │
    ▼
READY0 + READY1
    │
    ▼
----- ONLINE COMPUTE -----
    │
public Protein Gated-CNN
    │
    ▼
START
    │
    ▼
secure A_norm
    │
    ▼
MPC inference
    │
    ▼
affinity
----- ONLINE COMPLETE -----
```

Evaluator logs should show:

```text
[DDG_PRELOAD][KEY]
[DDG_BARRIER][READY]
[DDG_BARRIER][START]
[DDG_PROFILE][EVAL]
```

in that order.

---

## 8.4 Online runner parameters

Show all options:

```bash
python3 gpu_mpc/run_offline_online_local.py --help
```

Important parameters:

| Option | Meaning |
|---|---|
| `--protein-checkpoint` | original DeepDTAGen `.pth` containing `cnn.*` |
| `--num-samples` | logical sample count |
| `--micro-batch` | fixed internal MPC batch |
| `--bw` | ring bit width |
| `--scale` | fixed-point scale |
| `--gpu0` | local GPU for P0 |
| `--gpu1` | local GPU for P1 |
| `--ip` | `GpuPeer` address |
| `--full-key-ram` | preload complete key file |
| `--keep-work` | retain debug files |

The current backend uses TCP port:

```text
42003
```

---

## 8.5 Two-server deployment

The intended deployment is:

```text
                 TRUSTED OFFLINE SETUP

                         Dealer
                        /      \
                      K0        K1
                      │          │
                      ▼          ▼

                    ONLINE

             Server P0 <----> Server P1
             ---------        ---------
             share0           share1
             K0               K1
             public model     public model
             public protein   public protein
```

Before online execution:

1. prepare the dataset;
2. generate the two drug shares;
3. generate `K0` and `K1` on the trusted Dealer environment;
4. copy only share0 and `K0` to Server P0;
5. copy only share1 and `K1` to Server P1;
6. copy public model/protein assets to both servers;
7. confirm network access on the MPC port;
8. start the two Evaluators.

> **TODO before final submission**
>
> Insert the final copy-paste commands for the physical two-server runner after the current READY/START full-RAM architecture has been validated on two independent machines.
>
> Do not present the current localhost launcher as a verified two-server production launcher until that test is complete.

---
## 8.6 Validated local two-party smoke test

The complete README execution path has been validated using:

```text
logical N       = 3
micro-batch B   = 8
BW / SCALE      = 64 / 12
GPU P0 / P1     = 0 / 1
peer IP         = 127.0.0.1
key mode        = offline full file -> direct full RAM buffer
Protein         = timed FP32 GPU
A_norm          = secure online
```

The online runner confirmed:

```text
Dealer          = NOT STARTED
key mode        = offline full file -> direct full RAM buffer
Protein         = timed FP32 GPU
A_norm          = secure online
```

The Evaluators reached the READY barrier before online computation:

```text
[driver] SETUP: both Evaluators READY
```

The public Protein Gated-CNN was then executed, followed by the START release:

```text
[driver] ONLINE: start timed public Protein GatedCNN
[driver] ONLINE: Protein complete
[driver] ONLINE: release Evaluators with START marker
```

Both Evaluators completed successfully:

```text
[driver] evaluator return codes: E0=0 E1=0
```

The logical outputs were:

```text
AFFINITY_GLOBAL[0]=11.827393
AFFINITY_GLOBAL[1]=11.977295
AFFINITY_GLOBAL[2]=11.341064
```

and the driver finished with:

```text
PASS: N=3, fixed B=8, chunks=1, padded=8, returned=3
OFFLINE/ONLINE SEPARATION: PASS
Dealer process was never started by this online runner.
```

The same run confirmed that the complete FSS keys had already been loaded into
RAM before MPC chunk execution:

```text
P0 key_read_us = 0
P1 key_read_us = 0
```

Representative timing from this functional run was:

```text
SETUP_TOTAL     ≈ 3.513 s
PROTEIN         ≈ 2.618 s
MPC compute     ≈ 0.372 s / party
ONLINE_COMPUTE  ≈ 3.624 s
END_TO_END      ≈ 7.137 s
```

This test uses **Davis input rows with the KIBA checkpoint** in order to validate
the external input interface and complete execution lifecycle.

It is **not a Davis model-accuracy benchmark**.

The small `N=3` timing values are also not representative throughput results,
because five of the eight internal B8 positions are padding and process/CUDA
startup overhead dominates such a small workload.
---

# 9. Timing

The runner currently reports:

```text
PREPROCESS_UNTIMED
OFFLINE_DEALER
OFFLINE_KEY_DISTRIBUTION
KEY_PRELOAD
SETUP_TOTAL
PROTEIN
EVALUATOR
ONLINE_COMPUTE
END_TO_END
```

---

## 9.1 Offline Dealer

```text
OFFLINE_DEALER
```

is the complete trusted Dealer key-generation wall time.

It is reported separately even when Dealer execution occurs before the two online servers start.

---

## 9.2 Key preload

```text
KEY_PRELOAD
```

measures loading one complete party key file into the final FSS host buffer.

In the current lifecycle:

```text
included_in_online_wall = 0
included_in_setup_wall  = 1
```

---

## 9.3 Setup

```text
SETUP_TOTAL
```

currently contains Evaluator startup work such as:

```text
process startup
CUDA/backend initialization
key preload
peer connection
READY synchronization
```

---

## 9.4 Online compute

```text
ONLINE_COMPUTE
```

contains:

```text
public Protein Gated-CNN
        +
secure adjacency normalization
        +
two-party affinity MPC
```

---

## 9.5 End-to-end

Current:

```text
END_TO_END
```

means:

```text
Evaluator setup
    +
online computation
```

while input preprocessing and Dealer generation remain separately reported.

---

## 9.6 Compliance note

The timing records describe what the current implementation measures; they do not independently redefine the competition timing boundary.

Model-specific FSS preprocessing may require additional organizer clarification.

For this reason the current code reports:

```text
status=PRE_COMPLIANCE
```

rather than claiming final timing compliance.

---

# 10. Correctness and Performance

## 10.1 Functional regression

The current `BW=64, SCALE=12` KIBA smoke test produced:

```text
AFFINITY_GLOBAL[0]=10.361816
AFFINITY_GLOBAL[1]=11.346680
AFFINITY_GLOBAL[2]=11.227295
```

with:

```text
PASS
OFFLINE/ONLINE SEPARATION: PASS
```

The direct full-RAM implementation has also been validated across multiple B8 chunks.

A logical:

```text
N = 17
B = 8
```

test successfully executed three internal chunks.

---

## 10.2 Current smoke-test timing

The latest validated `N=3, B=8` local functional timing is reported in
Section 8.6.

The timing values are intentionally not duplicated here so that this section
does not retain stale measurements after lifecycle or implementation changes.

The `N=3` run is a functional smoke test, **not a representative throughput
benchmark**. Five of the eight internal B8 positions are padding, and
Python/PyTorch/CUDA process-startup overhead is significant at this size.

Final performance measurements should use substantially larger logical sample
counts after micro-batch and startup-overhead tuning.

---

## 10.3 Accuracy

The MPC program must output a **continuous affinity regression value** for each
drug-target pair.

The competition accuracy metric is the average of sensitivity and specificity:

```text
BalancedAccuracy = (Sensitivity + Specificity) / 2
```

For a threshold `t`, the current evaluation convention is:

```text
truth_class = (ground_truth_affinity >= t)
ref_class   = (reference_prediction >= t)
mpc_class   = (mpc_prediction >= t)
```

The reference and MPC predictions are evaluated against the same ground-truth
classes:

```text
BA_ref = (sensitivity_ref + specificity_ref) / 2
BA_mpc = (sensitivity_mpc + specificity_mpc) / 2
```

The qualification requirement is interpreted as:

```text
BA_ref - BA_mpc <= 0.02
```

that is, the MPC implementation should be no more than **2 percentage points**
below the reference model on the test data.

The implementation should not be tuned to a single affinity threshold. The
competition FAQ states that the organizers expect regression outputs and may
evaluate accuracy at several reasonable thresholds.

Therefore:

- continuous affinity values must be preserved as the submitted output;
- a simple relative regression error such as `|mpc-ref| / |ref| < 2%` is **not**
  the competition accuracy criterion;
- MAE/max-error measurements are useful engineering diagnostics, but they do
  not replace the competition accuracy gate.

---

# 11. Resource Requirements and Limitations

## 11.1 Current FSS key size

For:

```text
BW = 64
SCALE = 12
B = 8
secure online adjacency normalization
```

one FSS key chunk currently occupies approximately:

```text
2,954,940,416 bytes
```

per party.

For B8:

```text
chunks = ceil(N / 8)
```

so:

```text
key bytes / party
≈ ceil(N / 8) × 2,954,940,416
```

Example:

| Logical N | Chunks | Approx. key / party |
|---:|---:|---:|
| 8 | 1 | 2.95 GB |
| 16 | 2 | 5.91 GB |
| 128 | 16 | 47.28 GB |
| 512 | 64 | 189.12 GB |

These values are per online server.

---

## 11.2 Full-RAM mode

Advantages:

```text
no online per-chunk SSD key reads
no second complete RAM copy
simple persistent Evaluator
key_read_us = 0 during chunk MPC
```

Limitation:

```text
host RAM consumption grows linearly with padded chunk count
```

Before running a large logical batch, leave sufficient memory for:

```text
OS
CUDA/runtime
network buffers
input data
temporary host allocations
filesystem/page cache
```

---

## 11.3 Large datasets

The currently validated production path uses complete offline FSS key files
and preloads the complete per-party key stream into host RAM before online MPC
execution.

Therefore the complete per-party key file must fit in available host memory,
with additional memory reserved for the OS, CUDA/runtime state, input data,
network buffers, temporary allocations, and filesystem/page cache.

Memory-bounded SSD/key streaming is **not part of the currently validated
production workflow**.

For logical datasets whose complete FSS key stream does not fit safely in host
RAM, an additional memory-bounded key-consumption strategy would be required
and must be validated separately before it is presented as a supported
production mode.

---

## 11.4 Remaining performance work

Current major performance work includes:

```text
1. reduce public Protein worker startup overhead
2. remove unnecessary Evaluator startup delay
3. tune MPC micro-batch B
4. benchmark larger N
5. validate on two physical servers
6. benchmark on final target GPU/network environment
```

---

## 11.5 Cryptographic randomness

Production input secret sharing uses cryptographically appropriate OS randomness.

Before final submission, the Dealer/FSS randomness configuration must also be audited for the competition's required security level.

Fixed deterministic seeds must remain debug/regression-only.

---

# 12. Contact

**Team:** `<TEAM_NAME>`

**Institution:** `<INSTITUTION>`

**Contact:** `<CONTACT_NAME>`

**Email:** `<CONTACT_EMAIL>`

---

# Appendix: Final Validation Checklist

Before final submission:

```text
[ ] Fresh environment builds successfully
[ ] CUDA/PyTorch GPU execution works

[ ] Replacement CSV can be prepared
[ ] Replacement compatible .pth can be prepared

[ ] Drug is secret-shared
[ ] Protein remains public
[ ] A_norm is computed securely online

[ ] Dealer generates K0/K1 before online execution
[ ] Dealer exits before online MPC
[ ] P0 receives only share0 + K0
[ ] P1 receives only share1 + K1

[ ] Full-RAM path reports key_read_us=0
[ ] READY occurs before START
[ ] START occurs before MPC chunks

[ ] N=1 smoke test passes
[ ] multi-chunk regression passes
[ ] large-N benchmark passes

[ ] two physical server execution passes
[ ] accuracy gate passes
[ ] final timing interpretation is confirmed
[ ] cryptographic randomness audit passes

[ ] README contains final contact email
```