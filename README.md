# GPU-Accelerated 2PC DeepDTAGen Affinity Inference

GPU-accelerated two-party secure inference for the **drug-target affinity prediction branch** of DeepDTAGen, developed for **iDASH 2026 Track 3: Accelerating MPC-Based Deep Learning for Drug-Target Interaction Prediction**.

> **Current status:** `persistent-v2`, engineering / pre-compliance baseline.
>
> The current implementation already supports persistent arbitrary-size logical batches, bounded FSS-key storage, buffered ephemeral key transport, and BW64/S12 fixed-point inference.
>
> It is **not yet claimed to be a fully submission-compliant implementation**. The remaining competition-compliance items are documented explicitly in [Known Limitations / Submission Compliance](#20-known-limitations--submission-compliance).

---

## 1. Competition Scope

This project targets the **affinity prediction branch only**.

The drug-generation Transformer branch of DeepDTAGen is outside the current implementation.

The target affinity pipeline is conceptually:

```text
Private drug graph
    |
    v
GCN -> ReLU
    |
GCN -> ReLU
    |
GCN -> ReLU
    |
Masked Global Max Pool
    |
Drug FC
    |
128-d drug embedding
    |
    +----------------------+
                           |
Public protein sequence    |
    |                      |
Gated-CNN                  |
    |                      |
128-d protein embedding ---+
                           |
                           v
                     Concatenation
                           |
                        FC 1024
                           |
                        FC 512
                           |
                        FC 256
                           |
                         FC 1
                           |
                           v
                  Continuous affinity
```

The competition expects a **continuous regression output**, rather than a binary model optimized for one particular affinity threshold.

---

## 2. Privacy Model

The current implementation follows the Track 3 privacy model:

| Item | Privacy status |
|---|---|
| Drug / chemical compound | **Private** |
| Protein sequence | **Public** |
| Model parameters | **Public** |
| MPC intermediate values | **Confidential** |
| Final affinity prediction | Revealed output |

The private drug graph is represented using additive shares over a fixed-width ring.

Because the protein sequence is public, the protein encoder does **not** require MPC confidentiality. However, protein-model computation is still model computation and therefore must be handled correctly in the final competition timing boundary.

---

## 3. Current Arithmetic Configuration

The current validated production candidate is:

```text
Ring bit width : 64
Fraction bits  : 12
Ring           : Z_(2^64)
Default B      : 8
```

The relevant defaults are:

```text
reference/mpc_config.py

BW    = 64
SCALE = 12
```

and:

```text
gpu_mpc/run_persistent_local.py

--bw          default: 64
--scale       default: 12
--micro-batch default: 8
```

BW=32 remains available for experiments and debugging, but it is **not the current correctness candidate**.

The validated strict reference path is therefore referred to throughout this README as:

```text
BW64 / SCALE12
```

---

## 4. Implemented Secure Model

The current C++/CUDA MPC model contains the following secured computation.

### Drug branch

```text
GCN 94 -> 188
ReLU

GCN 188 -> 282
ReLU

GCN 282 -> 376
ReLU

Secret masked global max pool

FC 376 -> 1024
ReLU

FC 1024 -> 128
```

This produces a 128-dimensional drug representation.

### Protein branch

The current engineering baseline computes:

```text
Public protein sequence
    |
Plaintext Gated-CNN
    |
128-d protein representation
```

outside the C++ MPC model.

The result is currently stored in:

```text
protein_emb.dat
```

### Fusion branch

```text
128-d drug representation
        +
128-d protein representation
        |
        v
Concat -> 256
        |
FC 256 -> 1024
ReLU
        |
FC 1024 -> 512
ReLU
        |
FC 512 -> 256
ReLU
        |
FC 256 -> 1
        |
        v
Affinity
```

The current `weights.bin` contains the MPC-secured GCN, drug-FC, and fusion layers.

The plaintext protein Gated-CNN is currently not serialized into this MPC weight blob.

---

## 5. Repository Layout

The files most relevant to the current `persistent-v2` implementation are:

```text
.
├── README.md
│
├── gpu_mpc/
│   ├── deepdtagen_inference.cu
│   ├── deepdtagen.h
│   ├── ddg_orca.h
│   ├── ddg_orca_base.h
│   ├── ddg_orca_batched.h
│   ├── ddg_orca_opt.h
│   ├── dpf_dcf_adapter.h
│   ├── gcn_layer.h
│   ├── masked_maxpool.h
│   ├── masked_maxpool_layer.h
│   ├── Makefile
│   │
│   ├── run_persistent_local.py
│   ├── run_chunked_local.py
│   └── run_local_2pc.sh
│
├── reference/
│   ├── affinity_model.py
│   ├── csv_runner.py
│   ├── dense_gcn.py
│   ├── dense_graph.py
│   ├── export_weights.py
│   ├── fixed_forward.py
│   ├── fixedpoint.py
│   ├── masked_maxpool.py
│   ├── metrics.py
│   ├── mpc_config.py
│   ├── offline_prepare.py
│   ├── protein_plaintext.py
│   └── share_data.py
│
├── docs/
│   ├── COMPLETION_SUMMARY.md
│   ├── FILE_GUIDE_CN.md
│   ├── README.md
│   ├── SETUP_GUIDE_CN.md
│   └── correctness_davis_v2.md
│
├── real_gpu_2pc_benchmark.py
└── run_davis_multibatch.sh
```

The current main arbitrary-N validation entry point is:

```text
gpu_mpc/run_persistent_local.py
```

`run_chunked_local.py` is the older process-per-chunk arbitrary-N implementation and is retained mainly for regression and development purposes.

---

## 6. External Dependency

The GPU MPC implementation is based on:

```text
EzPC
GPU-MPC
Sytorch
```

The project expects an external GPU-MPC checkout:

```bash
export GPU_MPC_ROOT=/path/to/EzPC/GPU-MPC
```

The current validated project also depends on local arithmetic changes in the EzPC/Sytorch environment.

Therefore:

> A completely vanilla upstream EzPC checkout is not guaranteed to reproduce the current validated results without the corresponding project-side / Sytorch patches.

The project Makefile treats the EzPC tree as an external dependency rather than copying this repository into EzPC.

---

## 7. Build

### 7.1 Example Hopper build environment

The current local H800 environment uses:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_VERSION=12.8

export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export NVCC_PATH=/usr/local/cuda-12.8/bin/nvcc

export GPU_MPC_ROOT=/path/to/EzPC/GPU-MPC
```

### 7.2 Build BW64

```bash
make -C gpu_mpc \
  GPU_MPC_ROOT="$GPU_MPC_ROOT" \
  CUDA_VERSION=12.8 \
  CUDA_HOME="$CUDA_HOME" \
  NVCC_PATH="$NVCC_PATH" \
  GPU_ARCH=90a \
  BW=64 \
  deepdtagen_inference
```

The resulting executable is:

```text
gpu_mpc/deepdtagen_inference
```

### 7.3 Hardware note

Current local validation was performed on:

```text
2 x NVIDIA H800 PCIe
```

The official Track 3 evaluation environment uses:

```text
NVIDIA H100 PCIe 80 GB
```

Therefore the local H800 performance numbers in this README are **engineering measurements**, not official H100 results.

---

## 8. MPC Input Format

### 8.1 Current source directory contract

`run_persistent_local.py` expects a **sample-major contiguous** directory:

```text
source_dir/
├── x_share0.dat
├── x_share1.dat
├── adj_share0.dat
├── adj_share1.dat
├── mask_share0.dat
├── mask_share1.dat
└── protein_emb.dat
```

For N samples, each file stores fixed-size sample records consecutively:

```text
sample 0
sample 1
sample 2
...
sample N-1
```

All seven files must represent the same number of samples.

---

### 8.2 Fixed dimensions

Current dimensions are:

```text
NMAX       = 138
FEAT_DIM   = 94
POOL_DIM   = 376
PROTEIN    = 128
```

For BW64, each ring element occupies:

```text
8 bytes
```

Therefore the current per-sample storage contract is:

| File | Shape/sample | Bytes/sample |
|---|---:|---:|
| `x_share0.dat` | 138 × 94 | 103,776 |
| `x_share1.dat` | 138 × 94 | 103,776 |
| `adj_share0.dat` | 138 × 138 | 152,352 |
| `adj_share1.dat` | 138 × 138 | 152,352 |
| `mask_share0.dat` | 138 × 376 | 415,104 |
| `mask_share1.dat` | 138 × 376 | 415,104 |
| `protein_emb.dat` | 128 | 1,024 |

`run_chunked_local.py` uses these fixed strides to infer how many samples exist in a contiguous input directory.

---

### 8.3 Secret sharing

The current offline drug-input path produces additive shares:

```text
share0 + share1
    =
fixed-point plaintext
mod 2^BW
```

for:

```text
X
adjacency representation
tiled node mask
```

The share files are little-endian unsigned ring-element arrays.

For BW64:

```text
uint64 / <u8
```

For BW32 experimental mode:

```text
uint32 / <u4
```

---

### 8.4 Current adjacency representation

The **current pre-compliance implementation** shares:

```text
A_hat = D^(-1/2) A D^(-1/2)
```

rather than the original raw private adjacency matrix `A`.

This is an important known compliance limitation and is discussed separately in Section 20.

---

### 8.5 Protein representation

The current input contract uses:

```text
protein_emb.dat
```

containing a public 128-dimensional fixed-point protein representation.

Conceptually:

```text
Public protein sequence
        |
        v
Plaintext Gated-CNN
        |
        v
128-d protein vector
        |
        v
protein_emb.dat
```

In the current MPC fusion path:

```text
Party 0 : zero protein contribution
Party 1 : public protein representation
```

so the public representation enters the arithmetic computation without being treated as a confidential protein input.


### 8.6 New Dataset Integration Contract

The competition may evaluate the program using an additional dataset that is
not included in this repository. Therefore it is important to distinguish
between:

```text
raw evaluation dataset
        |
        v
input preparation / secret splitting
        |
        v
processed MPC input directory
        |
        v
run_persistent_local.py
```

#### Raw dataset format

The current Python reference and preparation code follows the DeepDTAGen
challenge-style CSV format.

The semantically required fields for affinity inference are:

| Column | Required for inference | Meaning |
|---|---|---|
| `compound_iso_smiles` | yes | Drug / compound SMILES string |
| `target_sequence` | yes | Public protein amino-acid sequence |
| `affinity` | no for inference; used for local validation/evaluation | Ground-truth affinity |

A minimal inference-oriented input CSV may therefore contain:

```csv
compound_iso_smiles,target_sequence
CCO...,MKT...
CCN...,VVK...
```

For local correctness or accuracy evaluation, the CSV may additionally contain
the ground-truth affinity:

```csv
compound_iso_smiles,target_sequence,affinity
CCO...,MKT...,7.42
CCN...,VVK...,10.31
```

Additional columns may also be present. They can be ignored if they are not
required by the affinity prediction branch.

> **Current helper limitation**
>
> `reference/offline_prepare.py` currently reuses the local baseline dataset
> loader and therefore expects an `affinity` field, even though the affinity
> value is not mathematically required to perform inference.
>
> A final submission-oriented dataset adapter should therefore make
> `affinity` optional.

---

#### Where the raw dataset may be stored

The raw evaluation dataset does not need to be stored in a hard-coded
repository directory.

For example, an evaluation CSV may be located at:

```text
/input/hidden_test.csv
```

or:

```text
/data/evaluation/test.csv
```

The important requirement is that the preparation script receives the correct
path.

Likewise, the processed MPC input directory may be placed at an arbitrary
location such as:

```text
/work/hidden_test/inputs/
```

The processed directory path is passed directly to:

```text
gpu_mpc/run_persistent_local.py
```

as the first positional argument.

For example:

```bash
python3 gpu_mpc/run_persistent_local.py \
  /work/hidden_test/inputs \
  /work/hidden_model/weights.bin \
  /work/mpc_keys \
  --num-samples 1000 \
  --micro-batch 8 \
  --bw 64 \
  --scale 12 \
  --allow-many-chunks
```

Therefore the inference runtime itself does not depend on a hard-coded dataset
directory.

---

#### Current per-sample preparation code

The current repository contains:

```text
reference/offline_prepare.py
```

with the function:

```python
prepare_sample(
    dataset,
    csv_path,
    row_idx,
    out_dir,
    scale=12,
    bw=64,
)
```

Conceptually, the current helper performs:

```text
one raw CSV row
      |
      v
prepare_sample(...)
      |
      v
sample_<row_idx>/
├── x_share0.dat
├── x_share1.dat
├── adj_share0.dat
├── adj_share1.dat
├── mask_share0.dat
├── mask_share1.dat
└── protein_emb.dat
```

The helper currently performs the main low-level preparation steps required by
the engineering baseline:

```text
drug SMILES
    |
    v
drug graph construction
    |
    v
fixed-size dense representation
    |
    v
fixed-point conversion
    |
    v
additive secret splitting
```

and:

```text
public protein sequence
    |
    v
plaintext Gated-CNN
    |
    v
protein_emb.dat
```

---

#### Current persistent-runner input format

The persistent arbitrary-N runner does **not** consume:

```text
sample_0/
sample_1/
sample_2/
...
```

as separate directories.

Instead, it expects one **sample-major contiguous input directory**:

```text
prepared_input/
├── x_share0.dat
├── x_share1.dat
├── adj_share0.dat
├── adj_share1.dat
├── mask_share0.dat
├── mask_share1.dat
└── protein_emb.dat
```

Each file contains all samples concatenated in the same order.

For example:

```text
x_share0.dat

sample 0 X-share
||
sample 1 X-share
||
sample 2 X-share
||
...
||
sample N-1 X-share
```

The same sample ordering must be used in every input file.

Thus:

```text
record i in x_share0.dat
record i in x_share1.dat
record i in adj_share0.dat
record i in adj_share1.dat
record i in mask_share0.dat
record i in mask_share1.dat
record i in protein_emb.dat
```

must all refer to the same drug-target pair.

---

#### Current processed input contract

For the current BW64 / SCALE12 pre-compliance baseline, the processed input is:

```text
private drug input
    |
    +-- x_share0.dat
    +-- x_share1.dat
    |
    +-- adj_share0.dat
    +-- adj_share1.dat
    |
    +-- mask_share0.dat
    +-- mask_share1.dat

public protein input
    |
    +-- protein_emb.dat
```

The current meaning of these files is:

```text
x_share{0,1}.dat
    additive shares of the fixed-size drug node-feature matrix X

adj_share{0,1}.dat
    additive shares of the currently precomputed normalized adjacency A_hat

mask_share{0,1}.dat
    additive shares of the tiled valid-node mask

protein_emb.dat
    public 128-dimensional protein representation produced by Gated-CNN
```

The current persistent runner accepts this processed representation regardless
of whether the raw dataset was Davis, KIBA, BindingDB, or an additional hidden
evaluation dataset.

---

#### Processed input validation

Before execution, the runner validates the processed files.

For BW64, the expected dimensions are:

```text
NMAX       = 138
FEAT_DIM   = 94
POOL_DIM   = 376
PROTEIN    = 128
element    = 8 bytes
```

Therefore each sample occupies:

| File | Bytes per sample |
|---|---:|
| `x_share0.dat` | 103,776 |
| `x_share1.dat` | 103,776 |
| `adj_share0.dat` | 152,352 |
| `adj_share1.dat` | 152,352 |
| `mask_share0.dat` | 415,104 |
| `mask_share1.dat` | 415,104 |
| `protein_emb.dat` | 1,024 |

The runner checks that:

1. every required file exists;
2. every file size is divisible by its expected per-sample stride; and
3. every file contains the same number of samples.

For example, after loading the processed directory the runner may report:

```text
available N = 10000
```

A request such as:

```text
--num-samples 5000
```

is valid, while:

```text
--num-samples 12000
```

is invalid because it exceeds the available processed sample count.

---

#### Current limitation: no final one-command dataset adapter yet

The repository currently contains:

```text
raw-row preparation logic
+
secret-sharing logic
+
protein preparation logic
+
persistent contiguous-input runtime
```

but it does **not yet contain a final submission-quality one-command script**
that converts an arbitrary new raw evaluation CSV directly into the complete
sample-major persistent-runner directory.

A planned final interface is:

```text
reference/prepare_dataset.py
```

with usage conceptually similar to:

```bash
python3 reference/prepare_dataset.py \
  --input-csv /input/hidden_test.csv \
  --output-dir /work/hidden_test \
  --bw 64 \
  --scale 12
```

and expected output:

```text
/work/hidden_test/
└── inputs/
    ├── x_share0.dat
    ├── x_share1.dat
    ├── adj_share0.dat
    ├── adj_share1.dat
    ├── mask_share0.dat
    ├── mask_share1.dat
    └── protein_emb.dat
```

This final adapter has intentionally not been frozen yet because two parts of
the input contract are expected to change during the remaining competition
compliance work.

---

#### Target submission input contract

The current pre-compliance representation contains:

```text
adj_share{0,1}.dat
    shares of precomputed A_hat
```

and:

```text
protein_emb.dat
    precomputed public Gated-CNN output
```

The intended submission-oriented representation is instead conceptually:

```text
private drug
    |
    +-- x_share{0,1}.dat
    |
    +-- adj_share{0,1}.dat
    |       shares of raw/private adjacency A
    |
    +-- mask_share{0,1}.dat

public protein
    |
    +-- target_sequence
            or an equivalent lossless encoded public sequence
```

The corresponding timed execution will become:

```text
raw secret-shared A
        |
        v
secure online normalization
        |
        v
A_norm
        |
        v
GCN
```

and:

```text
public target_sequence
        |
        v
timed plaintext Gated-CNN
        |
        v
128-d protein representation
        |
        v
fusion
```

Therefore the final `prepare_dataset.py` should be implemented **after**:

1. secure online `A -> A_norm` is completed; and
2. the public protein Gated-CNN is moved into the timed inference path.

This avoids freezing a dataset-conversion interface that would immediately
become obsolete.

---

#### Recommended interpretation for a new hidden dataset

When a new evaluation dataset is provided, the intended workflow is:

```text
new hidden dataset
        |
        | compound_iso_smiles
        | target_sequence
        | affinity (optional)
        v
dataset adapter
        |
        +-- validate SMILES
        +-- validate protein sequence
        +-- construct fixed-size graph representation
        +-- fixed-point conversion
        +-- secret splitting of private drug input
        +-- serialize public protein input
        v
processed MPC input
        |
        v
run_persistent_local.py
        |
        v
continuous affinity predictions
```

No dataset-specific C++ inference code should be required as long as the new
dataset follows the same DeepDTAGen input semantics and can be converted into
the documented fixed-shape MPC representation.

The dataset name itself is not part of the MPC runtime contract.

---

## 9. Public Weight Format

The model parameters are public.

The MPC-secured layers are exported into:

```text
weights.bin
weights.bin.json
```

### 9.1 `weights.bin`

The current format is:

```text
headerless
little-endian
int64
SCALE=12
```

The layer groups are serialized in forward order:

```text
GCN layers
Drug FC layers
Fusion layers
```

For each layer:

```text
weight matrix
bias vector
```

PyTorch stores a linear weight as:

```text
(out_features, in_features)
```

Before serialization, the project transposes it to:

```text
(in_features, out_features)
```

because the C++/Sytorch matrix multiplication computes conceptually:

```text
Y = X * W
```

using the `(in, out)` orientation.

---

### 9.2 `weights.bin.json`

The manifest contains metadata such as:

```text
scale
bitwidth
total element count

for each layer:
    name
    weight shape
    bias shape
    weight offset
    bias offset
```

This makes it possible to verify the raw binary layout independently.

### 9.3 New Model Weight Integration Contract

The evaluation may use a DeepDTAGen checkpoint different from the checkpoints
used during local development.

It is therefore important to distinguish between:

```text
original DeepDTAGen checkpoint
        |
        v
*.pth
        |
        +-------------------------------+
        |                               |
        v                               v
MPC weight conversion             public protein model
        |                               |
        v                               |
weights.bin                       Gated-CNN parameters
weights.bin.json                        |
        |                               |
        +---------------+---------------+
                        |
                        v
                    inference
```

A new model checkpoint does **not** need to be manually converted into C++
source code as long as it uses the same DeepDTAGen affinity architecture.

---

#### Original checkpoint format

The expected source model is a PyTorch checkpoint:

```text
*.pth
```

Existing development checkpoints follow names such as:

```text
model/deepdtagen_model_davis.pth

model/deepdtagen_model_kiba.pth
```

The current helper:

```text
reference/offline_prepare.py
```

uses the naming convention:

```text
model/deepdtagen_model_<dataset>.pth
```

when a dataset name is supplied.

However, the lower-level model loader and exporter do not fundamentally depend
on this filename.

A new checkpoint may therefore be stored at an arbitrary location, for example:

```text
/input/models/deepdtagen_model_hidden.pth
```

or:

```text
/work/models/new_model.pth
```

as long as the actual path is passed to the model conversion procedure.

---

#### Compatible checkpoint architecture

The current C++/CUDA model uses a fixed DeepDTAGen affinity architecture.

The expected layer dimensions are:

```text
Drug GCN
--------------------------------
94 -> 188
188 -> 282
282 -> 376


Drug FC
--------------------------------
376 -> 1024
1024 -> 128


Protein representation
--------------------------------
128


Fusion
--------------------------------
256 -> 1024
1024 -> 512
512 -> 256
256 -> 1
```

The current Python checkpoint loader expects affinity parameters corresponding
to:

```text
encoder.GraphConv1
encoder.GraphConv2
encoder.GraphConv3

encoder.Drug_FCs.0
encoder.Drug_FCs.3

fc.FC_layers.0
fc.FC_layers.3
fc.FC_layers.6
fc.FC_layers.9
```

The public protein Gated-CNN parameters are loaded from checkpoint entries under:

```text
cnn.*
```

For GCN layers, the Python loader supports the relevant PyTorch-Geometric
weight naming variants such as:

```text
*.lin.weight
```

and:

```text
*.weight
```

depending on the checkpoint / PyG version.

---

#### Same architecture, different numerical weights

If a new checkpoint changes only the trained numerical parameter values while
keeping the same architecture:

```text
same GCN layer count
same GCN dimensions
same Drug-FC dimensions
same 128-d protein representation
same fusion dimensions
```

then the current implementation can use the new model by regenerating its
weight artifacts.

In this case:

```text
C++ model source modification
    not required

C++ architecture modification
    not required

new weights.bin
    required

new protein-model parameters
    required
```

The inference binary does not need to contain dataset-specific numerical model
weights.

---

#### Incompatible architecture changes

The current implementation does **not** automatically support a checkpoint
whose network topology changes.

Examples include:

```text
different number of GCN layers

94 -> 256 instead of 94 -> 188

different final GCN dimension

different drug embedding dimension

different protein embedding dimension

additional / removed FC layers

different fusion widths
```

Such a model would no longer match the statically constructed C++/CUDA model
and would require corresponding implementation changes before inference.

Therefore:

```text
new parameter values
        |
        v
re-export only
```

is supported, while:

```text
new architecture
        |
        v
C++ / CUDA model adaptation required
```

is not currently automatic.

---

#### Current model conversion components

The current conversion path uses:

```text
reference/affinity_model.py
```

to load the DeepDTAGen checkpoint and:

```text
reference/export_weights.py
```

to serialize the MPC-secured layer parameters.

Conceptually:

```text
new_model.pth
       |
       v
AffinityModel.from_pth(...)
       |
       v
extract:
    GCN layers
    Drug FC layers
    Fusion layers
       |
       v
fixed-point quantization
       |
       v
dump_mpc_weights(...)
       |
       +------------------+
       |                  |
       v                  v
weights.bin       weights.bin.json
```

The currently validated conversion uses:

```text
BW    = 64
SCALE = 12
```

---

#### Exporting a new checkpoint

A compatible checkpoint can currently be converted directly using the existing
Python APIs.

Example:

```bash
cd /path/to/gpu-mpc-track

PTH=/input/models/deepdtagen_model_hidden.pth
OUTDIR=/work/hidden_model

mkdir -p "$OUTDIR"

python3 - "$PTH" "$OUTDIR" <<'PY'
import sys
from pathlib import Path

from reference.affinity_model import AffinityModel
from reference.export_weights import dump_mpc_weights

pth = Path(sys.argv[1]).resolve()
outdir = Path(sys.argv[2]).resolve()

if not pth.is_file():
    raise FileNotFoundError(
        f"checkpoint not found: {pth}"
    )

outdir.mkdir(
    parents=True,
    exist_ok=True,
)

model = AffinityModel.from_pth(
    str(pth),
    device="cpu",
)

out = outdir / "weights.bin"

manifest = dump_mpc_weights(
    model,
    str(out),
    scale=12,
)

print("checkpoint =", pth)
print("weights    =", out)
print("manifest   =", str(out) + ".json")
print("scale      =", manifest["scale"])
print("bitwidth   =", manifest["bitwidth"])
print("elements   =", manifest["total_elements"])
PY
```

Expected output artifacts:

```text
/work/hidden_model/
├── weights.bin
└── weights.bin.json
```

---

#### Recommended model asset directory

For a new evaluation checkpoint, the recommended directory layout is:

```text
/work/hidden_model/
├── model.pth
├── weights.bin
└── weights.bin.json
```

where:

```text
model.pth
```

is the original DeepDTAGen checkpoint, and:

```text
weights.bin
weights.bin.json
```

are the MPC-compatible exported artifacts.

The original checkpoint should be retained even after `weights.bin` has been
created.

---

#### Why the original `.pth` must currently be retained

The MPC `weights.bin` currently contains only the layers executed by the
MPC-secured affinity path:

```text
GCN layers

Drug FC layers

Fusion FC layers
```

It intentionally does **not** contain the public protein Gated-CNN.

The protein encoder parameters remain in the original checkpoint under:

```text
cnn.*
```

Therefore a complete new DeepDTAGen model consists conceptually of:

```text
new checkpoint
      |
      +----------------------------+
      |                            |
      v                            v
MPC-secured layers             public Gated-CNN
      |                            |
      v                            v
weights.bin                  protein representation
      |                            |
      +-------------+--------------+
                    |
                    v
                  fusion
```

Using only a new `weights.bin` while continuing to use the protein encoder from
a different checkpoint would create an inconsistent model.

For example, the following is invalid:

```text
Davis / old protein Gated-CNN
        +
new hidden-model MPC weights
```

unless those parameters are known to belong to the same trained checkpoint.

---

#### Current pre-compliance protein behavior

In the current engineering baseline:

```text
model.pth
    |
    v
cnn.* parameters
    |
    v
plaintext Gated-CNN
    |
    v
protein_emb.dat
```

is performed during data preparation.

Therefore, when using a new checkpoint in the current implementation:

> `protein_emb.dat` must be generated using the same checkpoint from which
> `weights.bin` was exported.

This requirement is important even though `protein_emb.dat` itself is public.

After the planned compliance modification, the Gated-CNN will run inside the
timed inference path and the relationship between the original model checkpoint
and the public protein path will become explicit at runtime.

---

#### Weight binary format

The exported:

```text
weights.bin
```

uses the current C++/CUDA weight-loading contract:

```text
headerless
little-endian
signed int64 storage
fixed-point SCALE=12
```

The layers are serialized in model forward order:

```text
GCN layers
        |
        v
Drug FC layers
        |
        v
Fusion FC layers
```

Each layer stores:

```text
weight matrix
followed by
bias vector
```

PyTorch linear weights are normally represented as:

```text
(out_features, in_features)
```

The exporter transposes them to:

```text
(in_features, out_features)
```

before serialization to match the C++/CUDA matrix multiplication convention.

---

#### Weight manifest

Each exported model also contains:

```text
weights.bin.json
```

The manifest records metadata such as:

```text
scale

bitwidth

total number of int64 elements

layer names

weight shapes

bias shapes

weight offsets

bias offsets
```

For the current validated configuration the manifest should report:

```json
{
  "bitwidth": 64,
  "scale": 12
}
```

The manifest should be retained together with `weights.bin`.

Although the C++ inference binary primarily consumes the raw `weights.bin`,
the manifest provides an important consistency check when replacing models.

---

#### Selecting a model at runtime

The persistent runner does not require a hard-coded weight directory.

The converted weight file is supplied as the second positional argument:

```bash
python3 gpu_mpc/run_persistent_local.py \
  /work/hidden_test/inputs \
  /work/hidden_model/weights.bin \
  /work/mpc_keys \
  --num-samples 1000 \
  --micro-batch 8 \
  --bw 64 \
  --scale 12 \
  --allow-many-chunks
```

Therefore compatible model weights can be changed without modifying the runner.

Conceptually:

```text
Dataset A + Model A
    |
    v
run_persistent_local.py \
    inputs_A \
    model_A/weights.bin


Dataset B + Model B
    |
    v
run_persistent_local.py \
    inputs_B \
    model_B/weights.bin
```

The runtime model path is data-driven rather than compiled as a fixed repository
filename.

---

#### Weight and runtime arithmetic must match

The exported fixed-point scale must agree with the inference runtime.

For the current validated configuration:

```text
weights:
SCALE=12

input shares:
SCALE=12

runtime:
--scale 12
```

must be consistent.

Similarly:

```text
runtime BW=64
```

must use the BW64 input-share format.

A mismatch such as:

```text
SCALE12 input
+
SCALE24 model weights
```

or:

```text
BW32 shares
+
BW64 runtime
```

is not a valid configuration.

The model preparation process should therefore validate these values before
execution.

---

#### New dataset and new model may be changed independently

The input dataset and model checkpoint are separate runtime assets.

For example:

```text
new hidden dataset
        |
        v
processed input
        |
        +----------------+
                         |
new compatible .pth     |
        |                |
        v                |
weights.bin              |
        |                |
        +-------+--------+
                |
                v
             inference
```

A new dataset does not by itself require new model weights.

Likewise, a compatible new checkpoint does not require changes to the raw
dataset format.

Only the expected DeepDTAGen feature semantics and tensor dimensions must
remain compatible.

---

#### Planned submission-quality model adapter

The repository currently contains the low-level checkpoint loader and weight
exporter, but does not yet provide a dedicated submission-oriented command such
as:

```text
reference/prepare_model.py
```

The intended final interface is conceptually:

```bash
python3 reference/prepare_model.py \
  --input-pth /input/models/deepdtagen_model_hidden.pth \
  --output-dir /work/hidden_model \
  --bw 64 \
  --scale 12
```

The wrapper should perform:

```text
checkpoint existence check
        |
        v
architecture / parameter validation
        |
        v
load checkpoint
        |
        v
export MPC weights
        |
        v
validate weights.bin.json
        |
        v
retain / register public protein model
```

and produce:

```text
/work/hidden_model/
├── model.pth
├── weights.bin
└── weights.bin.json
```

The underlying weight serialization does not need to be redesigned for a
compatible same-architecture checkpoint.

The final `prepare_model.py` wrapper can therefore be implemented after the
timed public-protein path is finalized.

---

#### Recommended workflow for a new checkpoint

The intended workflow for an additional same-architecture DeepDTAGen model is:

```text
1. Receive new checkpoint
        |
        v
/input/models/new_model.pth


2. Validate architecture
        |
        v
same affinity topology?


3. Export MPC parameters
        |
        v
weights.bin
weights.bin.json


4. Retain public protein parameters
        |
        v
model.pth / cnn.*


5. Prepare evaluation dataset using the same model
        |
        v
current:
protein_emb.dat

future:
timed public target_sequence -> Gated-CNN


6. Run inference
        |
        v
continuous affinity outputs
```

For a compatible same-architecture model, this workflow should not require
changes to the persistent MPC execution logic.

---

## 10. Persistent Arbitrary-N Runner

The main current local runner is:

```text
gpu_mpc/run_persistent_local.py
```

It separates:

```text
logical batch N
```

from:

```text
internal fixed micro-batch B
```

The default configuration is:

```text
B = 8
```

but the value is configurable.

### Example

```bash
SRC=/path/to/sample_major_input
WEIGHTS=/path/to/weights.bin
KEY_PARENT=/large/local/path/mpc_keys

python3 gpu_mpc/run_persistent_local.py \
  "$SRC" \
  "$WEIGHTS" \
  "$KEY_PARENT" \
  --num-samples 128 \
  --micro-batch 8 \
  --bw 64 \
  --scale 12 \
  --allow-many-chunks
```

---

## 11. Fixed-B Chunking and Padding

For arbitrary logical N:

```text
number of chunks = ceil(N / B)
```

Every physical model invocation uses exactly B samples.

For example:

```text
N = 17
B = 8
```

becomes:

```text
chunk 0 : 8 real samples
chunk 1 : 8 real samples
chunk 2 : 1 real sample + 7 dummy samples
```

Therefore the model physically processes:

```text
24 samples
```

but the runner returns only:

```text
17 logical outputs
```

The padded outputs are explicitly discarded.

Conceptually:

```text
Logical N=17
      |
      v
8 + 8 + 1
      |
      v
8 + 8 + (1 + 7 dummy)
      |
      v
24 physical predictions
      |
      v
trim dummy outputs
      |
      v
17 logical predictions
```

Expected machine-readable output:

```text
AFFINITY_GLOBAL[0]=...
AFFINITY_GLOBAL[1]=...
...
AFFINITY_GLOBAL[N-1]=...
```

and a final completion line:

```text
PASS: N=N, fixed B=B, chunks=..., padded=..., returned=N
```

---

## 12. Persistent Process Architecture

The persistent implementation avoids rebuilding the entire MPC environment for every micro-batch.

The following state remains alive across chunks:

```text
Dealer process
Evaluator process
CUDA context
model
GPU allocations
MPC network connection
randomness state
```

Conceptually:

```text
Logical N
   |
   v
fixed-B chunking
   |
   +-------------------------------+
   |                               |
   v                               v
Persistent Dealer P0          Persistent Dealer P1
   |                               |
   v                               v
party0 key slot                party1 key slot
   |                               |
   v                               v
Persistent Eval P0  <------->  Persistent Eval P1
                 MPC network
```

Every chunk receives fresh FSS key material.

A key is **not intentionally reused for different chunks**.

---

## 13. Bounded FSS-Key Pipeline

### 13.1 Previous scalability problem

The earlier persistent implementation generated one sequential key file containing all chunks.

That gives:

```text
key storage = O(number of chunks)
```

which becomes impractical for very large N.

---

### 13.2 Current single-slot design

The current D2 implementation uses one reusable large key slot per party:

```text
Dealer
   |
   v
generate fresh chunk key
   |
   v
partyX.key.tmp
   |
complete write
   |
close
   |
atomic rename
   |
partyX.key
   |
ready marker
   |
   v
Evaluator
   |
copy key into evaluator RAM
   |
run MPC
   |
ACK marker
   |
   v
Dealer deletes / reuses slot
```

Therefore:

```text
key working storage = O(B)
```

rather than:

```text
O(N)
```

---

### 13.3 Measured key sizes

For BW64:

```text
B = 8
2,810,019,840 bytes / party / chunk
≈ 2.62 GiB
```

For B16:

```text
B = 16
5,607,149,568 bytes / party / chunk
≈ 5.22 GiB
```

The key size is approximately linear in internal batch size.

---

## 14. Buffered Ephemeral Key Transport

### 14.1 Legacy behavior

The generic EzPC key file helpers use:

```text
O_DIRECT
```

for key reads and writes.

This is appropriate for persistent large key files in some environments, but the bounded key slot has a different lifetime:

```text
generate
-> immediately consume
-> ACK
-> delete
```

---

### 14.2 Profiling result

For B8, each party produces approximately:

```text
2.81 GB
```

of FSS key material per chunk.

With the legacy direct-I/O slot path, local profiling observed approximately:

```text
~6.1 s / B8 chunk
```

in the key-slot write path.

FSS key generation itself required only approximately:

```text
~0.14 s / B8 chunk
```

so the dominant bottleneck was key transport rather than cryptographic key generation.

---

### 14.3 Current default

Bounded slots now use:

```text
normal buffered file I/O
```

and do **not** force `fsync()` for the temporary producer-consumer file.

The synchronization contract remains:

```text
complete write
-> close
-> atomic rename
-> ready marker
-> evaluator reads
-> evaluator ACK
-> dealer deletes slot
```

This permits the operating-system page cache to participate in the local Dealer-to-Evaluator transfer.

It does **not** mean physical disk I/O can never occur. Linux can still perform background writeback depending on memory pressure and kernel dirty-page policies.

---

### 14.4 Legacy regression mode

The old direct-I/O behavior remains available:

```bash
export DDG_LEGACY_SLOT_IO=1
```

which restores the bounded slot to the legacy:

```text
O_DIRECT + fsync
```

behavior.

For the current optimized default:

```bash
unset DDG_LEGACY_SLOT_IO
```

---

## 15. Micro-Batch Selection

The current engineering default remains:

```text
B = 8
```

although B is configurable.

Buffered-slot N64 experiments produced:

| B | Chunks | Wall time | End-to-end throughput |
|---:|---:|---:|---:|
| 8 | 8 | ~31 s | ~2.065 samples/s |
| 16 | 4 | ~31 s | ~2.065 samples/s |

B16 did not improve end-to-end throughput in the current local environment.

At the same time, B16 approximately doubled:

```text
FSS key size
key RAM
key-slot working set
communication/chunk
```

Therefore B8 is currently preferred because it achieves approximately the same end-to-end speed with lower memory and I/O pressure.

This is an **engineering default**, not a fixed competition assumption.

The optimal B on the official H100 environment may differ.

---

## 16. Local Correctness Validation

Current arithmetic:

```text
BW=64
SCALE=12
```

Selected persistent-pipeline validation:

| Logical N | Internal B | Physical chunks | Result |
|---:|---:|---:|---|
| 9 | 8 | 2 | PASS |
| 17 | 8 | 3 | PASS |
| 64 | 8 | 8 | PASS |
| 128 | 8 | 16 | PASS |

---

### 16.1 N17 default buffered path

The final default buffered N17 test produced:

```text
17 / 17 outputs

MAE
= 1.7555147045848516e-07

MAX abs error
= 5.000000005139782e-07

MAX Q12 LSB
= 0.0020480000021052547

<= 1 LSB
= True

same 6 decimal digits
= True
```

---

### 16.2 N128 sustained test

The N128 buffered persistent run produced:

```text
128 / 128 outputs

MAE
= 2.3144531237362376e-07

MAX abs error
= 5.000000005139782e-07

MAX Q12 LSB
= 0.0020480000021052547

<= 1 LSB
= True

same 6 decimal digits
= True
```

These values are comparisons against the project's strict BW64/S12 fixed-point reference.

They are engineering correctness tests.

They are **not** equivalent to the official hidden-test qualification result.

---

## 17. Local Performance Results

### 17.1 Validation environment

Current measurements were obtained using:

```text
2 x NVIDIA H800 PCIe
CUDA 12.8 build environment
local / localhost two-party execution
```

These are not official competition H100 measurements.

---

### 17.2 O_DIRECT versus buffered bounded slots

For:

```text
N = 128
B = 8
16 persistent chunks
```

the measured wall-clock results were:

| Key-slot mode | Wall time | End-to-end throughput |
|---|---:|---:|
| Legacy O_DIRECT | ~123 s | ~1.041 samples/s |
| Buffered ephemeral | ~54 s | ~2.370 samples/s |

Local end-to-end speedup:

```text
123 / 54
≈ 2.28x
```

Strict fixed-point outputs were unchanged.

---

### 17.3 Communication

For B8:

```text
communication/chunk
= 116,277,080 bytes
```

Since each chunk contains 8 samples:

```text
communication/sample
= 14,534,635 bytes
≈ 13.86 MiB
```

For B16:

```text
communication/chunk
= 232,554,160 bytes
```

which is approximately twice the B8 value.

Communication therefore scales approximately linearly with internal batch size in the current implementation.

---

## 18. Engineering Profiling

The current implementation contains phase-level profiling instrumentation.

### Dealer

Example:

```text
[DDG_PROFILE][DEALER]
```

with fields including:

```text
input_load_us
h2d_us
keygen_us
slot_write_us
ack_wait_us
```

### Evaluator

Example:

```text
[DDG_PROFILE][EVAL]
```

with fields including:

```text
input_load_us
key_wait_us
key_read_us
h2d_us
sync_us
compute_us
comm_bytes
```

---

### 18.1 Example B8 buffered profile

Representative observations:

```text
FSS key generation
~0.14 s / chunk

buffered slot write
~1.3 s / chunk

key read
~0.7 s / chunk

MPC compute
~0.6-0.8 s / chunk
```

These values can vary between chunks because of GPU scheduling, synchronization, filesystem behavior, and local system activity.

---

### 18.2 Wait timers must not be summed blindly

Fields such as:

```text
ACK wait
key wait
peer sync
```

can overlap computation performed by another process.

For example:

```text
Dealer ACK wait
```

can include time during which the Evaluator is:

```text
reading key
running MPC
communicating
```

Therefore:

> Phase-level instrumentation is useful for bottleneck diagnosis, but wait times must not simply be summed to estimate competition wall time.

---

## 19. Competition Timing Semantics

A critical distinction must be made between:

```text
MPC protocol offline phase
```

and:

```text
competition-untimed preprocessing
```

These are **not the same concept**.

The competition allows only limited preprocessing outside the measured inference runtime, such as:

```text
input-format conversion
encryption
secret splitting
```

Model-involving computation must be included in the timed path.

Therefore the planned timing framework will use categories conceptually similar to:

```text
COMPETITION_UNTIMED_PREPROCESS
    |
    +-- input conversion
    +-- encryption
    +-- secret splitting


COMPETITION_TIMED_INFERENCE
    |
    +-- public protein model
    |
    +-- secure graph preprocessing
    |
    +-- MPC/FSS preprocessing
    |
    +-- FSS key generation
    |
    +-- key transport
    |
    +-- MPC online computation
    |
    +-- communication
```

The most important competition performance metric should ultimately be measured using a single well-defined:

```text
timed wall-clock interval
```

rather than by summing potentially overlapping phase timers.

---

## 20. Known Limitations / Submission Compliance

The current branch is intentionally described as a:

```text
pre-compliance engineering baseline
```

rather than a final competition submission.

Three next-stage tasks are particularly important.

---

### 20.1 Online normalized adjacency — NOT YET COMPLIANT

The current offline drug preprocessing computes:

```text
A_hat = D^(-1/2) A D^(-1/2)
```

before secret sharing.

The resulting normalized adjacency is then stored as:

```text
adj_share0.dat
adj_share1.dat
```

The Track 3 requirements instead require the normalization components, including:

```text
D

and

D^(-1/2)
```

to be derived online from the secret-shared original adjacency.

Therefore the current pipeline is:

```text
raw private A
      |
      v
plaintext normalization
      |
      v
A_hat
      |
      v
secret splitting
      |
      v
MPC GCN
```

while the intended final path is:

```text
raw private A
      |
      v
secret splitting
      |
      v
TIMED SECURE COMPUTATION
      |
      +-- secure row sums
      |
      +-- degree D
      |
      +-- D^(-1/2)
      |
      +-- secure normalization
      |
      v
A_norm
      |
      v
GCN
```

This is currently the most explicit remaining preprocessing-compliance issue.

---

### 20.2 Protein Gated-CNN timing — NOT YET FINAL

The protein sequence is public.

Therefore the protein Gated-CNN does **not** need to be executed as confidential MPC.

However, the current engineering pipeline is:

```text
protein sequence
      |
      v
plaintext Gated-CNN
      |
      v
protein_emb.dat
      |
      v
C++ MPC inference
```

The plaintext Gated-CNN is therefore currently evaluated before the main C++ timed inference run.

Because this is model computation, the final competition timing path must include its runtime.

The intended design is:

```text
TIMED INFERENCE START
        |
        +-----------------------------+
        |                             |
        v                             v
public protein                  private drug
        |                             |
plaintext Gated-CNN                  MPC
        |                             |
128-d protein vector                 |
        |                             |
        +--------------+--------------+
                       |
                       v
                     Fusion
                       |
                       v
                    Affinity
```

The protein computation is independent and public, so future versions may potentially overlap part of this computation with other timed work.

---

### 20.3 Standardized competition timing / output framework — NEXT TASK

The current `[DDG_PROFILE]` output was built mainly for engineering bottleneck analysis.

The next implementation stage should define an explicit competition-aware timing contract.

A planned machine-readable format is conceptually:

```text
[DDG_TIME][PREPROCESS]
input_conversion_us=...
secret_split_us=...


[DDG_TIME][PUBLIC_MODEL]
protein_gatedcnn_us=...


[DDG_TIME][MPC_PREPROCESS]
secret_adj_norm_us=...


[DDG_TIME][MPC_OFFLINE]
keygen_us=...
key_transport_us=...


[DDG_TIME][MPC_ONLINE]
compute_us=...
comm_bytes=...


[DDG_TIME][TOTAL]
samples=...
micro_batch=...
timed_wall_us=...
throughput_samples_s=...
comm_bytes=...
```

A human-readable summary should also be emitted.

For example:

```text
========================================
DeepDTAGen MPC Timing Summary
========================================

Logical samples
128

Micro-batch
8

Chunks
16

BW / SCALE
64 / 12


Competition-untimed preprocessing
---------------------------------
input conversion             ...
secret splitting             ...


Competition-timed inference
---------------------------------
public protein encoder       ...
secret graph normalization   ...
FSS key generation           ...
key transport                ...
MPC online compute           ...


Communication
---------------------------------
total MPC bytes              ...


Final competition wall time
---------------------------------
timed inference              ...

Throughput
---------------------------------
samples / second             ...
```

The final competition wall time must be measured independently.

It must **not** be computed by simply summing all phase timers because multiple components can execute concurrently or wait on one another.

---

## 21. Remaining Input Scalability Limitation

The FSS-key pipeline is now bounded:

```text
key working storage = O(B)
```

However, the current Python validation runner still creates all fixed-B input chunk directories before starting the persistent execution.

Conceptually:

```text
logical N
   |
   v
pre-materialize:
chunk_00000
chunk_00001
chunk_00002
...
chunk_XXXXX
```

Therefore temporary input working storage still grows approximately as:

```text
O(N)
```

This is substantially smaller than the former O(N) multi-gigabyte key accumulation problem, but it is still not a fully streaming production input pipeline.

A future large-N launcher can reuse bounded input buffers or stream input chunks.

---

## 22. Local Runner versus Final Two-Server Deployment

`run_persistent_local.py` is currently a **local development and validation orchestrator**.

It launches:

```text
Dealer P0
Dealer P1
Evaluator P0
Evaluator P1
```

from the same host.

The intended two-server mapping is conceptually:

```text
Server 0
================================
Dealer P0
Evaluator P0
party0 local bounded key slot
GPU 0 / H100


             MPC network
                  |
                  |
                  v


Server 1
================================
Dealer P1
Evaluator P1
party1 local bounded key slot
GPU 1 / H100
```

More precisely, on official hardware each machine is expected to provide its own GPU:

```text
Server 0 -> H100
Server 1 -> H100
```

The Dealer is a **software/protocol role**.

It does not imply the existence of a third physical server.

The large key slot for each party can remain local to that party's server:

```text
Dealer P0 -> local party0 slot -> Evaluator P0

Dealer P1 -> local party1 slot -> Evaluator P1
```

Only the Evaluator-to-Evaluator MPC protocol traffic must cross the two-party network.

The current local Python runner is not yet the final official two-server launcher.

---

## 23. Official Evaluation Hardware Context

The Track 3 evaluation environment specifies approximately:

```text
Per server
--------------------------------
NVIDIA H100 PCIe
80 GB GPU memory


System RAM
--------------------------------
~376 GB total
~250 GB available


Storage
--------------------------------
~430-450 MB/s sequential
SATA-bus limited


Network
--------------------------------
>1 Gbps
RTT <1 ms


GPU DMA
--------------------------------
No functional GPU DMA
```

These constraints are relevant to system design.

For example, B8 currently needs approximately:

```text
2.81 GB key slot / party
```

which is manageable relative to the stated available host RAM.

However:

> The current H800/local measurements must not be reported as if they were measured on the official H100/two-server environment.

---

## 24. Current Development Checkpoints

Important `persistent-v2` checkpoints include:

```text
persistent-v2-bounded-n128-pass
```

which records the bounded arbitrary-N pipeline validated through N128, and:

```text
persistent-v2-buffered-default-pass
```

which records the buffered bounded-key transport path after the local performance improvement.

These tags are engineering checkpoints.

They are not claims of official hidden-test qualification.

---

## 25. Recommended Current Validation Command

For the current local dual-GPU engineering environment:

```bash
cd /path/to/gpu-mpc-track

unset DDG_LEGACY_SLOT_IO

SRC=/path/to/contiguous/input
WEIGHTS=/path/to/weights.bin
KEY_PARENT=/large/local/path/mpc_keys

python3 gpu_mpc/run_persistent_local.py \
  "$SRC" \
  "$WEIGHTS" \
  "$KEY_PARENT" \
  --num-samples 128 \
  --micro-batch 8 \
  --bw 64 \
  --scale 12 \
  --allow-many-chunks
```

A successful run should contain:

```text
[driver] bounded key slot cleanup: PASS
```

and:

```text
PASS: N=128, fixed B=8, chunks=16, padded=128, returned=128
```

---

## 26. Current Correctness Regression Strategy

After a substantial implementation change, the recommended local regression sequence is:

```text
N17
    |
    +-- arbitrary-N correctness
    +-- final partial chunk
    +-- padding / trim
    |
    v

N64
    |
    +-- persistent multi-chunk stability
    +-- phase profiling
    |
    v

N128
    |
    +-- longer sustained execution
    +-- strict correctness
    +-- end-to-end wall time
```

This avoids repeatedly running large tests when a smaller boundary test would expose the same correctness bug.

---

## 27. Planned Next Development Order

The next development phase is:

```text
Current persistent-v2 baseline
          |
          v
1. Phase-aware timing/output framework
          |
          v
2. Online secure A -> A_norm
          |
          v
3. Timed public protein Gated-CNN
          |
          v
4. Re-run strict correctness
          |
          v
5. Re-run timing / performance analysis
          |
          v
6. Final two-server launcher adaptation
          |
          v
7. Submission-oriented compliance audit
```

The timing framework is intentionally first.

Without a stable timing boundary, later compliance changes would make it difficult to determine whether additional cost comes from:

```text
public protein model
secure graph normalization
key generation
key transport
MPC online model
communication
synchronization
```

---

## 28. What Is Already Established

The current engineering branch has established:

```text
BW64 / SCALE12 fixed-point path          validated

arbitrary logical N                     implemented

fixed configurable micro-batch          implemented

remainder padding / trimming            implemented

persistent Dealer                       implemented

persistent Evaluator                    implemented

persistent model/CUDA/network state     implemented

fresh FSS key per chunk                 implemented

bounded FSS key working storage O(B)    implemented

single reusable key slot / party        implemented

buffered ephemeral key transport        implemented

N64 sustained execution                 validated

N128 / 16-chunk execution               validated

strict fixed-point agreement            validated

local O_DIRECT -> buffered speedup       measured
```

---

## 29. What Is NOT Yet Established

The current branch does **not yet establish**:

```text
fully compliant online A normalization

final competition-compliant protein-model timing

final standardized competition timing boundary

official hidden-test qualification

official H100 two-server throughput

fully streaming N~100,000 input pipeline

final submission launcher

final security/compliance audit
```

These are explicitly tracked as remaining work rather than being implied by the current engineering results.

---

## 30. Additional Documentation

Additional documentation is available under:

```text
docs/
├── COMPLETION_SUMMARY.md
├── FILE_GUIDE_CN.md
├── README.md
├── SETUP_GUIDE_CN.md
└── correctness_davis_v2.md
```

Some older documents may describe earlier development stages such as:

```text
BW32
single-sample execution
process-per-chunk execution
older correctness baselines
```

When older documentation conflicts with the current branch, the following should be considered authoritative for the current engineering implementation:

```text
current persistent-v2 source code
+
root README.md
```

---

## 31. Current Baseline Summary

```text
Arithmetic
--------------------------------
BW=64
SCALE=12


Default internal micro-batch
--------------------------------
B=8


Logical batch size
--------------------------------
arbitrary N


Persistent state
--------------------------------
Dealer
Evaluator
model
CUDA context
GPU allocations
MPC network


Key generation
--------------------------------
fresh key every chunk


Key working storage
--------------------------------
one reusable slot / party
O(B)


Default slot transport
--------------------------------
buffered ephemeral file
no forced fsync


Legacy slot transport
--------------------------------
DDG_LEGACY_SLOT_IO=1
O_DIRECT + fsync


Current sustained correctness
--------------------------------
N=128
B=8
16 chunks
PASS


Local dual-H800 performance
--------------------------------
legacy O_DIRECT N128
~123 s
~1.041 samples/s

buffered N128
~54 s
~2.370 samples/s

local E2E improvement
~2.28x


Current project status
--------------------------------
persistent engineering baseline
pre-compliance
```

---

## 32. Immediate Next Tasks

The immediate engineering roadmap is:

1. **Standardize phase-aware competition timing and output**
   - define timed and untimed boundaries;
   - emit machine-readable timing records;
   - emit human-readable summaries;
   - retain independent wall-clock timing.

2. **Move adjacency normalization into the timed secure path**
   - input raw secret-shared `A`;
   - securely derive degree information;
   - securely compute the required normalization factors;
   - construct `A_norm` online;
   - preserve BW64/S12 correctness.

3. **Move public protein Gated-CNN into the timed inference path**
   - keep the protein public;
   - avoid unnecessary MPC protection;
   - include model computation in competition wall time;
   - investigate safe overlap with MPC preparation.

4. **Revalidate**
   - N17 correctness;
   - N64 profiling;
   - N128 sustained correctness and performance.

5. **Prepare final two-server deployment**
   - one party per physical server;
   - one H100 per party;
   - local bounded key handoff;
   - network MPC between evaluators.

---

## 33. Status Disclaimer

The current implementation is intended to provide a reproducible and technically honest engineering baseline.

Performance numbers in this README should always be interpreted together with their environment.

In particular:

```text
2.370 samples/s
```

is a **local dual-H800 engineering measurement**.

It is not an official H100 Track 3 score.

Likewise, the current strict fixed-point correctness results demonstrate consistency with the project's BW64/S12 reference implementation, but do not by themselves establish official hidden-test qualification.

The project will only describe an implementation as submission-compliant after the remaining preprocessing, timing, deployment, and evaluation requirements have been addressed.