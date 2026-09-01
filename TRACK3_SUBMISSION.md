# iDASH Track 3 Submission Guide

## 1. Team Information

**Team Name:**  
[Your Team Name]

**Institution:**  
[Your Institution]

**Contact Email:**  
[Your Email Address]


---

# 2. Solution Overview

This submission provides a GPU-accelerated privacy-preserving inference framework for the DeepDTAGen drug-target affinity (DTA) prediction task.

The submitted solution performs secure two-party inference using GPU-based secure multi-party computation (MPC).

The main objectives are:

- Protect private drug molecular graph information.
- Enable efficient privacy-preserving affinity prediction.
- Support practical deployment with offline/online separation.
- Provide GPU acceleration for secure inference.


The system contains:

- Private drug molecular graph processing using MPC.
- Public protein sequence feature extraction.
- Secure affinity prediction.
- Offline FSS key generation.
- Online two-party GPU MPC inference.


For detailed algorithmic description, please refer to:

```
METHOD_DESCRIPTION.md
```


---

# 3. System Architecture

The submitted system follows an offline preprocessing phase and an online secure inference phase.


## 3.1 Offline Phase

A semi-honest offline dealer generates FSS key material for the online parties.

The generated keys are stored before online inference.


```
                 Offline Dealer

              FSS Key Generation

                    |
          -----------------------
          |                     |
          v                     v

        Party 0               Party 1

        Key 0                 Key 1
```


The offline phase is separated from online computation.


---

## 3.2 Online Phase

Two GPU-enabled parties execute secure inference.

```
Private Drug Share 0             Private Drug Share 1

        |                              |
        |                              |
        v                              v

             GPU MPC Party 0
                    |
                    |
              Secure MPC
                    |
                    |
             GPU MPC Party 1

                    |
                    v

          Affinity Prediction Output
```


---

# 4. Runtime Environment

The submitted solution is provided together with a Docker environment.

All required dependencies, CUDA libraries, GPU-MPC components, and Python environments are included in the submitted container.


The expected execution environment contains:

| Component | Requirement |
|---|---|
| GPU | NVIDIA GPU with CUDA support |
| CUDA | CUDA 12.x recommended |
| Memory | Sufficient RAM for FSS key loading |
| Network | Required for two-party deployment |


Detailed environment configuration and development setup are described in:

```
ENVIRONMENT_SETUP.md
```


---

# 5. Input Data Format

The evaluation system may provide a new private test dataset that is not shared with participating teams.

The submitted program expects the input data to follow the format described below.


## 5.1 Raw Dataset Format

The original dataset should be provided as a CSV file.

Required columns:

| Column | Description |
|---|---|
| compound_iso_smiles | SMILES representation of drug molecules |
| target_sequence | Protein target sequence |
| affinity | Optional affinity label for evaluation |


Example:

```csv
compound_iso_smiles,target_sequence,affinity
SMILES_EXAMPLE,PROTEIN_SEQUENCE_EXAMPLE,VALUE
```


The affinity label is only required for accuracy evaluation and is not consumed during private inference.


---

## 5.2 Dataset Preprocessing

Before MPC inference, the raw dataset should be converted into the MPC input format.


The preprocessing command is:

```bash
python3 reference/prepare_dataset.py \
    --csv <input_csv> \
    --output <prepared_output_directory>
```


The preprocessing stage generates sample-major MPC input files.


Example:

```
prepared_dataset/
├── x_share0.dat
├── x_share1.dat
├── adj_share0.dat
├── adj_share1.dat
├── mask_share0.dat
├── mask_share1.dat
├── target_ids.dat
└── metadata.json
```


---

## 5.3 Prepared MPC Input Format


### Drug Feature Shares

Files:

```
x_share0.dat
x_share1.dat
```

contain secret shares of drug molecular features.


Shape:

```
(number_of_samples, nmax, feature_dimension)
```


---

### Drug Graph Adjacency Shares

Files:

```
adj_share0.dat
adj_share1.dat
```

contain secret shares of molecular graph adjacency information.


Shape:

```
(number_of_samples, nmax, nmax)
```


---

### Mask Shares

Files:

```
mask_share0.dat
mask_share1.dat
```

contain secret shares of graph padding masks.


---

### Protein Target IDs

File:

```
target_ids.dat
```

contains identifiers used to retrieve public protein sequences.


The protein branch uses public protein information and does not require secret sharing.


---

## 5.4 Adding a New Dataset

To evaluate a new dataset, provide a CSV file following the required format.

The workflow is:

```
New CSV Dataset
        |
        v
prepare_dataset.py
        |
        v
Prepared MPC Input Directory
        |
        v
Offline Key Generation
        |
        v
Secure Online Inference
```


Example:

```bash
python3 reference/prepare_dataset.py \
    --csv new_dataset.csv \
    --output new_dataset_prepare
```


After preprocessing, the generated directory can be used as the input directory for inference.


---

# 6. Model Weight Format

The submitted framework uses MPC-compatible fixed-point model weights
for secure inference.

## 6.1 Existing Weight Format

The inference program uses:

```
weights.bin
weights.bin.json
```

The files contain the fixed-point representation of DeepDTAGen model
parameters required by the MPC inference backend.

During inference, the weight file is provided as an input argument:
```
<weights.bin>
```



Example:

```bash
python3 gpu_mpc/run_offline_dealer.py \
    <prepared_dataset> \
    <weights.bin> \
    <key_directory>
```

---

## 6.2 Adding New Model Weights

If a new DeepDTAGen model checkpoint is provided, it must first be
converted into the MPC-compatible weight format.

The conversion workflow is:

```
DeepDTAGen PyTorch checkpoint
            |
            v
    prepare_model.py
            |
            v
    MPC-compatible weights
            |
            v
    weights.bin
    weights.bin.json
```

The converted files should replace the existing model weight files used by inference.

Example structure:
```
model/
├── deepdtagen_model_xxx.pth
gpu_mpc/
├── weights.bin
└── weights.bin.json
```

After replacing the weight files, offline key generation and online
inference should be executed again because FSS keys are generated
according to the model parameters.

---

# 7. Execution Workflow


The complete execution contains the following stages:

```
Dataset Preparation
        |
        v
Offline FSS Key Generation
        |
        v
Two-party Online MPC Inference
        |
        v
Affinity Prediction Output
```


---

## 7.1 Offline Key Generation


The offline dealer generates party-specific FSS keys.

Example:

```bash
python3 gpu_mpc/run_offline_dealer.py \
    <prepared_dataset> \
    <weights.bin> \
    <key_directory> \
    --num-samples <N>
```


The offline phase is executed before online inference.


---

## 7.2 Local Two-Party Validation


For local validation with two GPUs:

```bash
python3 gpu_mpc/run_offline_online_local.py \
    <prepared_dataset> \
    <weights.bin> \
    <key_directory> \
    --num-samples <N> \
    --gpu0 0 \
    --gpu1 1 \
    --full-key-ram
```


This mode simulates two MPC parties on one machine.


---

## 7.3 Two-Machine Deployment

The submitted framework supports real two-party deployment on two
independent machines.

In the two-machine deployment mode, each machine runs one MPC party
independently.

The execution entry point is:

```gpu_mpc/run_offline_online_party.py```

The deployment model:

    Machine 0                              Machine 1

    Party 0                                Party 1

    Private share 0                        Private share 1

    FSS key K0                             FSS key K1

           |                                      |
           |                                      |
           +---------- Secure MPC Network --------+
                            |
                            v
                Affinity Prediction Output

Party 0 acts as the server side and listens for the MPC connection.

Party 1 connects to Party 0 using the provided peer IP address.

---

## 7.3.1 Party 0 Execution

Party 0 should be started first.

Example command:

```bash
python3 gpu_mpc/run_offline_online_party.py \
    <source_dir> \
    <weights_bin> \
    <offline_key_dir> \
    --party 0 \
    --peer-ip <party0_ip> \
    --num-samples <N> \
    --micro-batch 8 \
    --bw 64 \
    --scale 12 \
    --gpu 0 \
    --full-key-ram

```

Party 0 requires:

-  Party 0 private input shares.
-  Party 0 offline FSS key.
-  Network address for MPC communication.

---

## 7.3.2 Party 1 Execution

After Party 0 starts listening, Party 1 should be started.

Example command:
```bash
python3 gpu_mpc/run_offline_online_party.py \
    <source_dir> \
    <weights_bin> \
    <offline_key_dir> \
    --party 1 \
    --peer-ip <party0_ip> \
    --num-samples <N> \
    --micro-batch 8 \
    --bw 64 \
    --scale 12 \
    --gpu 0 \
    --protein-checkpoint <deepdtagen_checkpoint.pth> \
    --full-key-ram
```

Party 1 requires:
-   Party 1 private input shares.
-   Party 1 offline FSS key.
-   Public DeepDTAGen protein checkpoint containing the Gated-CNN
    component.
-   Party 0 network address.

------------------------------------------------------------------------

## 7.3.3 Important Parameters

| Parameter              | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| `--party`              | MPC party identifier (`0` or `1`)                              |
| `--peer-ip`            | Party 0 IP address used for MPC communication                  |
| `--num-samples`        | Number of inference samples                                    |
| `--micro-batch`        | Internal MPC batch size                                        |
| `--bw`                 | Fixed-point bit width                                          |
| `--scale`              | Fixed-point scaling factor                                     |
| `--gpu`                | GPU device used by the party                                   |
| `--protein-checkpoint` | Public DeepDTAGen checkpoint containing protein CNN parameters |
| `--full-key-ram`       | Load complete FSS key file into RAM before online inference    |

---

## 7.3.4 Local Two-Party Validation

For development and correctness validation, the repository also provides a local two-GPU execution mode.

The local validation entry point is:

```gpu_mpc/run_offline_online_local.py```

This mode simulates two MPC parties on a single machine using two GPUs.

Example:
```bash
    python3 gpu_mpc/run_offline_online_local.py \
        <source_dir> \
        <weights_bin> \
        <offline_key_dir> \
        --protein-checkpoint <deepdtagen_checkpoint.pth> \
        --num-samples <N> \
        --micro-batch 8 \
        --bw 64 \
        --scale 12 \
        --gpu0 0 \
        --gpu1 1 \
        --full-key-ram
```

The local mode is intended for validation and benchmarking.

The two-machine mode is the deployment mode used for official MPC
execution.

---

# 8. Output Format

The inference program outputs affinity prediction values.

Example:

```
AFFINITY_GLOBAL[0]=11.827393
AFFINITY_GLOBAL[1]=11.977295
AFFINITY_GLOBAL[2]=11.341064
```

Each output corresponds to one drug-target pair.

The output is a continuous affinity regression value.


---

# 9. Correctness Verification

The submission verifies:

- Secure inference execution.
- Offline/online separation.
- Correct reconstruction of affinity predictions.
- Removal of padded samples.

Example successful output:

```
PASS: N=3, fixed B=8, chunks=1, padded=8, returned=3

OFFLINE/ONLINE SEPARATION: PASS
```

---

# 10. Performance Evaluation

The submitted implementation provides GPU-accelerated MPC inference.

Performance benchmarks are reported in:

```
README.md
```


The benchmark includes:

- Offline key generation time.
- Online setup time.
- Online MPC computation time.
- Throughput.
- Communication overhead.

---

# 11. Contact Information

For questions regarding this submission, please contact:


Email:

```
[Your Email Address]
```
