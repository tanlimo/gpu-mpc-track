# GPU-Accelerated 2PC Inference for DeepDTAGen

## Documentation

- Environment setup:
  `ENVIRONMENT_SETUP.md`

- Competition submission description:
  `TRACK3_SUBMISSION.md`

- Validation and benchmark results:
  `docs/VALIDATION_RESULTS.md`

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

## 0.1 Prepare dataset

python3 reference/prepare_dataset.py \
  --csv data/davis_test.csv \
  --output "$DDG_WORK/davis3_prepare" \
  --bw 64 \
  --scale 12 \
  --limit 3

## 0.2 Prepare model

python3 reference/prepare_model.py \
  --checkpoint model/deepdtagen_model_kiba.pth \
  --output "$DDG_WORK/kiba_model" \
  --scale 12

## 0.3 Generate complete offline FSS keys

python3 gpu_mpc/run_offline_dealer.py \
  "$DDG_WORK/davis3_prepare" \
  "$DDG_WORK/kiba_model/weights.bin" \
  "$DDG_WORK/davis3_keys" \
  --num-samples 3 \
  --micro-batch 8 \
  --bw 64 \
  --scale 12 \
  --gpu 0

## 0.4 Run local two-party online inference

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

The current implementation generates the complete FSS key stream during the offline phase before online inference.

The online Evaluator uses the direct full-RAM key loading mode:


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
- no external key storage access occurs during online MPC execution;
- all required FSS key material is loaded before online computation starts;
- micro-batches access the preloaded RAM buffer through offsets.

---

# 3. Repository Structure

The repository is organized into source code, model preparation tools,
 runtime scripts, and submission documentation.

```
gpu-mpc-track/
│
├── README.md
│     Main project overview and usage guide
│
├── ENVIRONMENT_SETUP.md
│     Environment installation, Docker setup, and build instructions
│
├── TRACK3_SUBMISSION.md
│     Competition submission description and execution instructions
│
├── METHOD_DESCRIPTION.md
│     Detailed technical method description
│
├── docs/
│   │
│   ├── VALIDATION_RESULTS.md
│   │     Benchmark results and validation records
│   │
│   └── archive/
│         Historical development notes and intermediate documents
│
├── gpu_mpc/
│   │
│   ├── deepdtagen_inference.cu
│   │     Main Dealer / Evaluator CUDA implementation
│   │
│   ├── deepdtagen.h
│   │     DeepDTAGen affinity inference graph
│   │
│   ├── gcn_layer.h
│   │
│   ├── masked_maxpool.h
│   │
│   ├── secure_adj_norm.h
│   │     Secure adjacency normalization
│   │
│   ├── ddg_orca.h
│   ├── ddg_orca_base.h
│   ├── ddg_orca_batched.h
│   │     GPU-MPC backend implementation
│   │
│   ├── run_offline_dealer.py
│   │     Offline FSS key generation
│   │
│   ├── run_offline_online_local.py
│   │     Local two-party GPU inference launcher
│   │
│   ├── run_offline_online_party.py
│   │     Independent online party launcher for two-server deployment
│   │
│   └── Makefile
│
├── reference/
│   │
│   ├── prepare_dataset.py
│   │     External dataset → MPC input format
│   │
│   ├── prepare_model.py
│   │     DeepDTAGen checkpoint → MPC weights
│   │
│   ├── affinity_model.py
│   │
│   ├── dense_graph.py
│   │
│   └── fixed_forward.py
│
├── model/
│   └── DeepDTAGen checkpoints
│
├── data/
│   └── External datasets
│
└── scripts/
    └── Development and validation scripts
```

Large generated files are intentionally excluded from Git:

```
datasets

prepared secret shares

FSS key files

compiled CUDA binaries

large model checkpoints
```

These files should be generated inside the target execution environment.

------

# 4. Dataset Preparation

The submission supports replacing the evaluation dataset through the standard
 CSV preparation pipeline.

The complete workflow is:

```
External CSV dataset

        │

        ▼

reference/prepare_dataset.py

        │

        ▼

Prepared MPC input directory

        │

        ▼

Offline FSS key generation

        │

        ▼

Online GPU-MPC inference
```

------

## 4.1 External input format

The expected external dataset format is CSV.

Required columns:

| Column                | Required | Description                 |
| --------------------- | -------- | --------------------------- |
| `compound_iso_smiles` | Yes      | Drug SMILES representation  |
| `target_sequence`     | Yes      | Protein amino-acid sequence |
| `affinity`            | Optional | Ground-truth affinity label |

Example:

```
compound_iso_smiles,target_sequence,affinity
CCO,MKT...,10.21
CCN,MSA...,11.04
```

The `affinity` column is only used for accuracy evaluation.

It is not consumed during MPC inference.

Hidden evaluation datasets can be processed using the same format.

------

## 4.2 Prepare a new dataset

For a new evaluation dataset:

```
python3 reference/prepare_dataset.py \
  --csv data/new_dataset.csv \
  --output "$DDG_WORK/new_dataset_prepare" \
  --bw 64 \
  --scale 12
```

For debugging:

```
python3 reference/prepare_dataset.py \
  --csv data/new_dataset.csv \
  --output "$DDG_WORK/new_dataset_prepare" \
  --bw 64 \
  --scale 12 \
  --limit 3
```

Available options:

```
python3 reference/prepare_dataset.py --help
```

Important parameters:

| Option     | Description                  |
| ---------- | ---------------------------- |
| `--csv`    | input CSV dataset            |
| `--output` | generated prepared directory |
| `--bw`     | MPC ring bit width           |
| `--scale`  | fixed-point scale            |
| `--nmax`   | maximum molecule graph size  |
| `--limit`  | debug subset only            |

------

## 4.3 Prepared private drug format

Each drug molecule is converted into a fixed-size graph representation.

Current configuration:

```
nmax = 138

feature dimension = 94
```

The generated private components are:

```
Node features:

shape:
[138, 94]

privacy:
secret-shared
Adjacency:

shape:
[138, 138]

values:
{0,1}

self-loop:
included

privacy:
secret-shared
Pooling mask:

shape:
[138,376]

privacy:
secret-shared
```

The dataset preparation stage stores raw adjacency.

Normalized adjacency is computed securely during MPC inference.

------

## 4.4 Public protein format

Protein sequences are treated as public input.

The sequence processing follows the DeepDTAGen encoding:

```
Protein sequence

        │

        ▼

DeepDTAGen token mapping

        │

        ▼

target_ids.dat
```

Current configuration:

```
sequence length = 1000
```

The Protein Gated-CNN is executed online using the original checkpoint.

No precomputed:

```
protein_emb.dat
```

is required.

------

## 4.5 Prepared directory

The generated directory contains:

```
new_dataset_prepare/

├── x_share0.dat

├── x_share1.dat

├── adj_share0.dat

├── adj_share1.dat

├── mask_share0.dat

├── mask_share1.dat

├── target_ids.dat

├── metadata.json

└── affinity.npy
```

Deployment mapping:

```
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

------

## 4.6 Expected output

Successful preparation:

```
PREPARE DATASET: PASS

samples = ...

output = ...

protein = public target_ids.dat; NO protein_emb.dat

adj = raw binary + self-loops, scale=0
```

Metadata:

```
cat "$DDG_WORK/new_dataset_prepare/metadata.json"
```

contains:

```
number of samples

feature dimension

graph configuration

sharing method

protein format

fixed-point configuration
```

------

## 4.7 Secret-sharing randomness

The production dataset preparation path uses operating-system cryptographic
 randomness for additive secret sharing.

The option:

```
--deterministic-seed
```

is provided only for debugging and regression testing.

It should not be used for production dataset preparation.

---

# 5. Model Preparation

The submitted solution separates the DeepDTAGen model into:

```
Public Protein branch

        +

Private Drug / Fusion MPC branch
```

The original DeepDTAGen checkpoint is used for both:

```
Original .pth checkpoint

        │

        ├── Public Protein Gated-CNN
        │
        └── MPC model conversion

                │

                ▼

             weights.bin
```

Therefore, a complete deployment requires:

```
1. Original DeepDTAGen checkpoint (.pth)

2. MPC-compatible weights.bin
```

------

## 5.1 Model execution split

The current inference architecture is:

```
Drug branch:

private input

        │

        ▼

GPU-MPC inference

        │

        ▼

128-dimensional drug representation



Protein branch:

public sequence

        │

        ▼

FP32 GPU Gated-CNN

        │

        ▼

128-dimensional protein representation



Fusion:

drug vector + protein vector

        │

        ▼

GPU-MPC affinity prediction
```

The Protein Gated-CNN remains public and uses the original checkpoint.

The MPC backend uses converted model parameters stored in:

```
weights.bin
```

------

## 5.2 Prepare weights

Convert a DeepDTAGen checkpoint:

```
python3 reference/prepare_model.py \
  --checkpoint model/deepdtagen_model_kiba.pth \
  --output "$DDG_WORK/kiba_model" \
  --scale 12
```

The generated directory:

```
$DDG_WORK/kiba_model/

├── weights.bin

├── weights.bin.json

└── model_metadata.json
```

Show available options:

```
python3 reference/prepare_model.py --help
```

------

## 5.3 Expected output

Successful model preparation:

```
PREPARE MODEL: PASS

checkpoint = ...

weights = ...

scale = 12

MPC layers = 9

Protein = original .pth cnn.* (timed FP32 GPU)
```

The generated `weights.bin` contains parameters required by the MPC path:

```
GCN layers

Drug fully-connected layers

Fusion fully-connected layers
```

The original `.pth` checkpoint remains required for:

```
Public Protein Gated-CNN inference
```

------

## 5.4 Replacing model weights

A compatible DeepDTAGen checkpoint can be converted using:

```
python3 reference/prepare_model.py \
  --checkpoint model/new_model.pth \
  --output "$DDG_WORK/new_model" \
  --scale 12
```

Then use:

```
$DDG_WORK/new_model/weights.bin
```

for MPC inference.

The corresponding checkpoint:

```
model/new_model.pth
```

is passed to the Protein Gated-CNN execution path.

A replacement checkpoint must preserve:

```
DeepDTAGen affinity branch architecture

layer dimensions

parameter naming compatibility
```

A checkpoint with a different network architecture requires additional MPC
 implementation changes.

------

## 5.5 Model submission requirements

For final deployment, the following files must be provided:

```
DeepDTAGen checkpoint:

model/*.pth


Converted MPC weights:

weights.bin

weights.bin.json
```

The model replacement procedure is also described in:

```
TRACK3_SUBMISSION.md
```

---

# 6. Offline Dealer

## 6.1 Purpose

The trusted Offline Dealer generates the complete correlated FSS key material
 required by the two online MPC Evaluators.

The current workflow is:

```
Prepared private shares
        +
Public model weights
        │
        ▼
 Trusted Offline Dealer
        │
        ├── generate K0
        │
        └── generate K1
        │
        ▼
 Distribute party-specific keys

        K0 → Server P0

        K1 → Server P1
```

The Dealer is an offline preprocessing component.

It does not participate in online MPC inference.

After generating and distributing the keys:

```
Dealer exits

        ↓

Server P0  <──── GPU-MPC / 2PC ────>  Server P1
```

The current implementation uses one logical trusted Dealer:

```
Dealer:

generate K0

generate K1

exit
```

------

## 6.2 Storage requirement

FSS key files can be large.

The current production configuration:

```
BW              = 64

SCALE           = 12

micro-batch     = 8

secure A_norm   = enabled
```

generates approximately:

```
K0:

2,954,940,416 bytes


K1:

2,954,940,416 bytes
```

Total:

```
K0 + K1 ≈ 5.91 GB
```

The required storage depends on the number of internal MPC chunks:

```
chunks = ceil(N / B)
```

For example:

| Logical N | Micro-batch B | Chunks | Approx. key size / party |
| --------- | ------------- | ------ | ------------------------ |
| 8         | 8             | 1      | 2.95 GB                  |
| 16        | 8             | 2      | 5.91 GB                  |
| 64        | 8             | 8      | 23.64 GB                 |
| 128       | 8             | 16     | 47.28 GB                 |

Before key generation:

```
df -h "$DDG_WORK"
```

should be used to verify available storage.

Large generated key directories should be placed on a filesystem with
 sufficient capacity.

------

## 6.3 Generate complete offline keys

The current production path generates complete FSS key streams offline.

Example:

```
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

General format:

```
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

Available options:

```
python3 gpu_mpc/run_offline_dealer.py --help
```

Important parameters:

| Option          | Description                   |
| --------------- | ----------------------------- |
| `--num-samples` | logical input sample count    |
| `--micro-batch` | internal fixed MPC batch size |
| `--bw`          | ring bit width                |
| `--scale`       | fixed-point scale             |
| `--gpu`         | GPU used by Dealer            |
| `--keep-work`   | retain temporary files        |

The generated key material corresponds to the secure online computation path:

```
A_raw adjacency

        ↓

degree computation

        ↓

normalization

        ↓

A_norm

        ↓

GPU-MPC inference
```

------

## 6.4 Output files

Generated key directory:

```
davis3_keys/

├── DeepDTAGen_64_12_party0_inference_key0.dat

├── DeepDTAGen_64_12_party1_inference_key1.dat

├── metadata.json

└── logs/

    ├── dealer_keygen_k0.log

    └── dealer_keygen_k1.log
```

The metadata records:

```
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

Example:

```
samples                  = 3

micro_batch              = 8

chunks                   = 1

bw                       = 64

scale                    = 12

secure_adj_norm          = true

logical_dealer_count     = 1

online_evaluator_started = false
```

------

## 6.5 Fixed micro-batches

Logical sample number `N` and internal MPC micro-batch size `B` are independent.

Example:

```
N = 17

B = 8
```

The execution becomes:

```
chunk 0:

8 real samples


chunk 1:

8 real samples


chunk 2:

1 real sample

7 padding samples
```

The backend removes padding outputs before returning final predictions.

The number of generated key chunks is:

```
chunks = ceil(N / B)
```

------

## 6.6 Key distribution

After Dealer generation:

```
K0 → Server P0

K1 → Server P1
```

Each online server receives only its own key and private input share.

Online separation:

```
Server P0:

private drug share0

FSS key K0


Server P1:

private drug share1

FSS key K1
```

The Dealer terminates before online inference.

Complete lifecycle:

```
OFFLINE

Trusted Dealer

    generate K0

    generate K1

    distribute keys

    terminate


ONLINE

Server P0  <──── GPU-MPC / 2PC ────>  Server P1
```

------

## 6.7 Validated Dealer smoke test

The current implementation has been validated with:

```
N      = 3

B      = 8

BW     = 64

SCALE  = 12

A_norm = secure online
```

Example output:

```
[offline-dealer] generate K0

[offline-dealer] generate K1


key bytes/chunk/party = 2,954,940,416

key bytes/party       = 2,954,940,416

total K0+K1 bytes     = 5,909,880,832


OFFLINE DEALER: PASS
```

Detailed benchmark information is maintained in:

```
docs/VALIDATION_RESULTS.md
```

------

## 6.8 Timing and compliance note

Dealer timing is reported separately:

```
[DDG_TIME][OFFLINE_INPUT_FORMAT]

[DDG_TIME][OFFLINE_DEALER_KEYGEN]

[DDG_TIME][OFFLINE_DEALER_TOTAL]

[DDG_TIME][OFFLINE_KEY_DISTRIBUTION]
```

The current implementation separates:

```
Offline:

Dataset preparation

Dealer key generation

Key distribution


Online:

Evaluator setup

GPU-MPC inference

Affinity prediction
```

The final competition timing boundary depends on the official evaluation
 procedure defined by the organizers.

Therefore, the repository reports each lifecycle stage explicitly without
 assuming a specific treatment of offline preprocessing.


---


# 7. Running Inference

The inference workflow consists of:

```
Prepared dataset
        │
        ▼
Offline FSS key generation
        │
        ▼
Online two-party GPU-MPC inference
        │
        ▼
Continuous affinity prediction
```

The current production path uses:

```
Evaluator key mode:

offline generated FSS keys
        │
        ▼
direct full-RAM preload
        │
        ▼
online MPC execution
```

The complete inference lifecycle is:

```
OFFLINE

Trusted Dealer
    │
    ├── generate K0
    ├── generate K1
    │
    ├── distribute K0 → Server P0
    └── distribute K1 → Server P1


ONLINE

Server P0                    Server P1
---------                    ---------
drug share0                 drug share1
K0                          K1
public weights              public weights
                            public protein checkpoint

        │                         │
        └──── GPU-MPC / 2PC ──────┘

                    │
                    ▼

          affinity prediction
```

------

## 7.1 Single-machine two-party run

The local validation environment runs two MPC parties on one machine.

Topology:

```
GPU 0                     GPU 1
P0                        P1
│                         │
share0                    share1
K0                        K1
│                         │
└────── GPU-MPC 2PC ──────┘
```

The validated local entry point is:

```
gpu_mpc/run_offline_online_local.py
```

Example:

```
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

Important parameters:

```
--num-samples
    logical inference sample count

--micro-batch
    fixed internal MPC batch size

--bw
    ring bit width

--scale
    fixed-point scale

--gpu0 / --gpu1
    GPUs assigned to the two MPC parties

--full-key-ram
    preload complete FSS key files into evaluator memory
```

------

## 7.2 One-sample smoke test

The external logical sample count does not need to equal the internal MPC
 micro-batch size.

Example:

```
logical samples:

N = 1


internal MPC execution:

B = 8

1 real sample
7 padding samples
```

The MPC backend executes a fixed-size batch, then removes padded outputs before
 returning the final affinity prediction.

------

## 7.3 Full-RAM key lifecycle

The current implementation uses direct full-RAM FSS evaluation.

Lifecycle:

```
START

Evaluator P0/P1 process launch
            │
            ▼
Load complete K0/K1
            │
            ▼
Initialize GPU-MPC backend
            │
            ▼
Establish MPC peer connection
            │
            ▼
READY synchronization
            │
            ▼
ONLINE COMPUTATION
            │
            ▼
Affinity output

END
```

During online computation:

- FSS keys are accessed from memory;
- No external key streaming is used.
- The online phase operates entirely on the preloaded FSS RAM buffer.
- no key-slot management is required;
- micro-batches access preloaded key material through offsets.

------

## 7.4 Online runner parameters

Show available options:

```
python3 gpu_mpc/run_offline_online_local.py --help
```

Main options:

| Option                 | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| `--protein-checkpoint` | DeepDTAGen `.pth` checkpoint containing public Protein Gated-CNN parameters |
| `--num-samples`        | logical input sample number                                  |
| `--micro-batch`        | internal MPC batch size                                      |
| `--bw`                 | ring bit width                                               |
| `--scale`              | fixed-point scale                                            |
| `--gpu0`               | GPU assigned to party P0                                     |
| `--gpu1`               | GPU assigned to party P1                                     |
| `--ip`                 | MPC peer address                                             |
| `--full-key-ram`       | enable complete FSS key preload                              |
| `--keep-work`          | keep temporary debugging files                               |

------

## 7.5 Two-server deployment

For physical deployment, each online party runs independently.

The entry point is:

```
gpu_mpc/run_offline_online_party.py
```

Each server receives only its own private share and FSS key.

Deployment model:

```
                 OFFLINE

              Trusted Dealer
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
         K0                  K1
          │                   │
          ▼                   ▼


                 ONLINE

        Server P0              Server P1

        share0                 share1
        K0                     K1
        weights                weights
                               protein checkpoint

             │                    │
             └──── GPU-MPC 2PC ───┘
```

------

## 7.5.1 Server P0

Run:

```
python3 gpu_mpc/run_offline_online_party.py \
  "$DDG_WORK/prepared_dataset" \
  "$DDG_WORK/model/weights.bin" \
  "$DDG_WORK/keys" \
  --party 0 \
  --peer-ip SERVER_P0_IP \
  --num-samples N \
  --micro-batch B \
  --bw 64 \
  --scale 12 \
  --gpu 0 \
  --full-key-ram
```

P0 provides:

```
private drug share0

K0

public MPC parameters
```

------

## 7.5.2 Server P1

Run:

```
python3 gpu_mpc/run_offline_online_party.py \
  "$DDG_WORK/prepared_dataset" \
  "$DDG_WORK/model/weights.bin" \
  "$DDG_WORK/keys" \
  --party 1 \
  --peer-ip SERVER_P0_IP \
  --num-samples N \
  --micro-batch B \
  --bw 64 \
  --scale 12 \
  --gpu 0 \
  --protein-checkpoint model/deepdtagen_model_kiba.pth \
  --full-key-ram
```

P1 provides:

```
private drug share1

K1

public Protein Gated-CNN checkpoint
```

------

## 7.5.3 Network convention

The two-party deployment requires:

```
P0:
    listens for MPC peer connection


P1:
    connects to P0
```

The control and MPC communication ports are configured by the runner.

Verify that:

```
P0 ↔ P1
```

network communication is available before starting inference.

------

## 7.6 Validation entry points

Available execution checks:

Dataset:

```
python3 reference/prepare_dataset.py --help
```

Model:

```
python3 reference/prepare_model.py --help
```

Offline Dealer:

```
python3 gpu_mpc/run_offline_dealer.py --help
```

Local two-party inference:

```
python3 gpu_mpc/run_offline_online_local.py --help
```

Two-server party inference:

```
python3 gpu_mpc/run_offline_online_party.py --help
```

Detailed validation results are recorded in:

```
docs/VALIDATION_RESULTS.md
```

---

# 8. Timing

The implementation reports different execution stages separately to distinguish
 offline preprocessing, setup overhead, and online MPC computation.

The current timing categories are:

```
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

Detailed benchmark results and hardware-specific measurements are maintained
 separately:

```
docs/VALIDATION_RESULTS.md
```

------

## 8.1 Offline Dealer

```
OFFLINE_DEALER
```

represents the complete trusted Dealer key-generation runtime.

The Dealer phase includes:

```
prepared private shares
        +
public model weights
        │
        ▼
 trusted Offline Dealer
        │
        ├── generate K0
        │
        └── generate K1
```

The Dealer runs before online inference and is reported separately.

The current submission workflow treats the Dealer as an offline trusted setup
 component.

------

## 8.2 Key preload

```
KEY_PRELOAD
```

measures loading the complete party-specific FSS key file into the evaluator
 memory buffer.

The current production path uses:

```
Evaluator key mode:

offline generated FSS key
        │
        ▼
direct full-RAM preload
        │
        ▼
online MPC execution
```

During online computation:

- No external key streaming is used.
- The online phase operates entirely on the preloaded FSS RAM buffer.
- no per-chunk key file reading occurs;
- evaluator accesses key material through memory offsets.

Current timing classification:

```
KEY_PRELOAD:

included_in_online_wall = 0

included_in_setup_wall = 1
```

------

## 8.3 Setup

```
SETUP_TOTAL
```

contains evaluator initialization before online computation.

The setup phase includes:

```
process startup

CUDA/backend initialization

FSS key preload

MPC peer connection

READY synchronization
```

The setup phase completes before the online computation begins.

------

## 8.4 Online computation

```
ONLINE_COMPUTE
```

represents the actual inference execution after setup completion.

It includes:

```
Public Protein Gated-CNN
        +
Secure adjacency normalization
        +
Two-party GPU-MPC inference
        +
Affinity prediction
```

The online computation excludes:

```
Offline Dealer key generation

Offline key distribution

Dataset preparation
```

------

## 8.5 End-to-end runtime

```
END_TO_END
```

represents:

```
Evaluator setup

+

Online computation
```

The current reported scope is:

```
END_TO_END =
SETUP_TOTAL + ONLINE_COMPUTE
```

The following stages are reported separately:

```
Dataset preparation

Offline Dealer

Offline key distribution
```

to maintain clear offline/online separation.

------

## 8.6 Timing compliance note

The current timing output reports all major lifecycle stages explicitly.

Example:

```
[DDG_TIME][OFFLINE_DEALER]

[DDG_TIME][KEY_PRELOAD]

[DDG_TIME][SETUP_TOTAL]

[DDG_TIME][ONLINE_COMPUTE]

[DDG_TIME][END_TO_END]
```

The final competition timing boundary depends on the official evaluation
 procedure defined by the organizers.

Therefore, this repository reports measured components separately and does not
 assume any specific treatment of offline cryptographic preprocessing.

Detailed validation measurements are provided in:

```
docs/VALIDATION_RESULTS.md
```

---

# 9. Correctness and Performance

## 9.1 Functional regression

The implementation has been validated with:

```
BW     = 64
SCALE  = 12
```

and two-party GPU-MPC execution using direct full-RAM FSS key evaluation.

The validation covers:

- correct continuous affinity prediction output;
- offline/online lifecycle separation;
- fixed micro-batch execution;
- multi-chunk inference;
- padding and output trimming correctness.

Detailed validation cases and measured outputs are recorded in:

```
docs/VALIDATION_RESULTS.md
```

------

## 9.2 Performance evaluation

Performance depends on:

- logical sample count `N`;
- internal micro-batch size `B`;
- GPU hardware;
- FSS key configuration;
- online/offline timing scope.

The implementation reports:

```
Offline Dealer time
Key preload time
Evaluator setup time
Protein inference time
Online MPC computation time
End-to-end runtime
```

Detailed benchmark results are maintained separately:

```
docs/VALIDATION_RESULTS.md
```

The current performance results are engineering validation measurements.
 They are intended to verify correctness, lifecycle behavior, and runtime
 characteristics of the implementation.

Final competition performance depends on the official evaluation environment,
 hardware configuration, and organizer-defined timing boundary.

------

## 9.3 Accuracy

The MPC program outputs a continuous affinity regression value for each
 drug-target pair.

The competition accuracy metric is the average of sensitivity and specificity:

```
BalancedAccuracy = (Sensitivity + Specificity) / 2
```

For threshold `t`:

```
truth_class = (ground_truth_affinity >= t)

ref_class = (reference_prediction >= t)

mpc_class = (mpc_prediction >= t)
```

The reference and MPC predictions are evaluated using the same ground-truth
 classes:

```
BA_ref = (sensitivity_ref + specificity_ref) / 2

BA_mpc = (sensitivity_mpc + specificity_mpc) / 2
```

The qualification requirement is:

```
BA_ref - BA_mpc <= 0.02
```

Therefore:

- submitted outputs remain continuous affinity values;
- relative regression error is not the competition accuracy criterion;
- MAE and maximum prediction error are engineering diagnostics only.

The final accuracy evaluation should be performed using the official hidden test dataset and evaluation procedure provided by the organizers.

---

# 10. Resource Requirements and Limitations

## 10.1 Current execution environment requirements

The current implementation is designed for GPU-enabled Linux environments.

Validated development environment:

```
OS                  Ubuntu 22.04
CUDA                12.8
Compiler            GCC/G++ 11+
GPU                 NVIDIA CUDA GPU
Python              3.12
```

The submitted solution is expected to run inside the provided competition Docker environment.

The detailed environment setup, build process, and dependency installation are documented separately:

```
ENVIRONMENT_SETUP.md
```

------

## 10.2 GPU and memory requirements

The current GPU-MPC implementation uses GPU acceleration for:

```
- FSS key generation
- secure MPC computation backend
- drug graph inference
- public Protein Gated-CNN inference
```

A typical deployment requires:

```
Server P0:
    NVIDIA GPU
    private drug share0
    FSS key K0

Server P1:
    NVIDIA GPU
    private drug share1
    FSS key K1
    public protein checkpoint
```

The exact GPU memory requirement depends on:

```
- logical sample number N
- micro-batch size B
- FSS key configuration
- CUDA runtime overhead
- model execution path
```

------

## 10.3 FSS key storage requirement

The current implementation uses:

```
Evaluator key mode:
    direct full-RAM preload
```

The complete party-specific FSS key file is loaded into the final evaluator
 buffer before online computation.

For the validated configuration:

```
BW              = 64
SCALE           = 12
micro-batch     = 8
secure A_norm   = enabled
```

the key size is approximately:

```
K0:
    2,954,940,416 bytes

K1:
    2,954,940,416 bytes
```

Total:

```
K0 + K1 ≈ 5.91 GB
```

For multiple internal micro-batch chunks:

```
chunks = ceil(N / B)
```

the generated key storage increases approximately linearly with the number of
 chunks.

Example:

| Logical N | Micro-batch B | Chunks | Approx. key size / party |
| --------- | ------------- | ------ | ------------------------ |
| 8         | 8             | 1      | 2.95 GB                  |
| 16        | 8             | 2      | 5.91 GB                  |
| 64        | 8             | 8      | 23.64 GB                 |
| 128       | 8             | 16     | 47.28 GB                 |

Before generating offline keys, ensure the selected filesystem has sufficient
 capacity.

------

## 10.4 Full-RAM evaluation mode limitations

The current production path intentionally uses:

```
full-key-ram
```

The current implementation uses direct full-RAM FSS key loading.

The online Evaluator does not perform external key streaming during MPC execution.

Advantages:

- no external key loading overhead during online MPC computation;
- simpler evaluator lifecycle;
- deterministic key access through memory offsets;
- lower implementation complexity.

Limitations:

- the complete key file must fit into available memory;
- memory usage increases with the number of FSS chunks;
Very large datasets require sufficient storage and memory capacity because complete FSS key material is loaded before online computation.

The current implementation targets competition-scale evaluation rather than
 Dataset size is currently limited by available storage and memory resources.

------

## 10.5 Dataset and model replacement limitations

The submitted solution supports replacing:

```
Dataset:
    CSV input
    ↓
reference/prepare_dataset.py
    ↓
prepared MPC input directory

Model:
    DeepDTAGen compatible checkpoint
    ↓
reference/prepare_model.py
    ↓
weights.bin
```

The replacement dataset must provide:

```
compound_iso_smiles
target_sequence
(optional) affinity
```

The replacement model checkpoint must preserve the expected DeepDTAGen
 architecture.

A checkpoint with different layer dimensions or network structure requires
 additional model conversion and MPC implementation changes.

------

## 10.6 Current validation scope

The current implementation has been validated for:

```
- local two-party GPU execution
- offline Dealer generation
- direct full-RAM key loading
- fixed micro-batch execution
- multi-chunk inference
- continuous affinity output
```

Additional validation required before final competition execution:

```
- final Docker submission environment
- official hidden evaluation dataset
- final competition timing measurement
- physical two-server network deployment
```

---

# 11. Contact

**Team:** `<TEAM_NAME>`

**Email:** `<CONTACT_EMAIL>`

---

# Appendix: Final Validation Checklist

This checklist summarizes the validation items required before submitting the
 final Track 3 solution.

------

## Environment

-  Competition Docker environment can be entered successfully.
-  CUDA-enabled GPU environment is available.
-  Required Python dependencies are installed.
-  CUDA backend and MPC components can be compiled successfully.

Detailed environment instructions:

```
ENVIRONMENT_SETUP.md
```

------

## Dataset preparation

-  External CSV dataset format is supported.
-  Drug SMILES and protein sequence columns are correctly parsed.
-  Private drug features are converted into additive secret shares.
-  Raw adjacency with self-loops is generated.
-  Public protein sequence is converted into `target_ids.dat`.

Validation command:

```
python3 reference/prepare_dataset.py --help
```

------

## Model preparation

-  DeepDTAGen checkpoint conversion is supported.
-  MPC-compatible `weights.bin` can be generated.
-  Public Protein Gated-CNN checkpoint remains available.

Validation command:

```
python3 reference/prepare_model.py --help
```

------

## Offline Dealer

-  Offline Dealer generates party-specific FSS keys.
-  Dealer generates keys independently from online inference.
-  Generated keys can be loaded by online Evaluators.
-  Key distribution follows:

```
K0 → Server P0

K1 → Server P1
```

Validation command:

```
python3 gpu_mpc/run_offline_dealer.py --help
```

------

## Online two-party inference

-  Local two-party GPU inference is validated.
-  Direct full-RAM FSS key evaluation is validated.
-  Fixed micro-batch execution is validated.
-  Multi-chunk inference is validated.
-  Padding samples are correctly removed from final outputs.

Local validation entry:

```
python3 gpu_mpc/run_offline_online_local.py --help
```

Two-server deployment entry:

```
python3 gpu_mpc/run_offline_online_party.py --help
```

------

## Correctness

-  Continuous affinity prediction values are produced.
-  Output count matches logical input sample count.
-  Offline and online phases are separated.
-  Secret-shared drug input remains distributed between two parties.

Detailed validation records:

```
docs/VALIDATION_RESULTS.md
```

------

## Performance measurement

-  Offline Dealer runtime is reported separately.
-  Key preload time is reported separately.
-  Setup runtime is reported separately.
-  Online computation runtime is reported separately.
-  End-to-end runtime is reported separately.

Benchmark tables:

```
docs/VALIDATION_RESULTS.md
```

------

## Final submission check

Before submission:

-  Rebuild inside the final Docker image.
-  Run the complete pipeline from a clean environment.
-  Replace demonstration dataset with official evaluation dataset.
-  Prepare required model weights.
-  Verify generated output format.
-  Confirm contact information in submission documents.

Submission description:

```
TRACK3_SUBMISSION.md
```

------

This checklist only records the final validated workflow.
Historical development notes and intermediate experiments are archived under:
docs/archive/

 separately and are not part of the submission path.