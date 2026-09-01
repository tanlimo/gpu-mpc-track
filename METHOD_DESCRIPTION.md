# iDASH Track 3 Submission Description

## 1. Solution Overview

This submission implements a GPU-accelerated privacy-preserving
inference framework for the DeepDTAGen drug-target affinity (DTA)
prediction task.

The system performs secure two-party inference using GPU-based secure multi-party computation (MPC). The implementation protects private molecular graph information while enabling efficient affinity
prediction.

Main design goals:

-   Privacy-preserving drug-target affinity prediction.
-   GPU acceleration for MPC computation.
-   Efficient offline/online separation.
-   Practical deployment support.

---

## 2. System Architecture

The system follows an offline preprocessing phase and an online secure inference phase.

## Offline Phase

A semi-honest offline dealer generates FSS key material for online
parties.

                    Offline Dealer

                  FSS Key Generation

                      /       \
                     /         \
                  K0             K1
                   |             |
                   v             v

                Party 0       Party 1

The generated key material is prepared before online inference.

## Online Phase

Two GPU-enabled parties execute secure inference.

    Private Drug Share 0              Private Drug Share 1
            |                                |
            |                                |
            v                                v

         GPU MPC Party 0  <---------->  GPU MPC Party 1

                        |
                        v

              Predicted Binding Affinity

---

## 3. Privacy Model

The current implementation follows the Track 3 deployment setting.

| Component | Privacy Status |
|---|---|
| Drug molecular graph | Private |
| Drug feature representation | Private |
| Adjacency information | Private |
| Protein sequence | Public |
| Protein encoder model | Public |
| MPC model parameters | Public |
| Final affinity prediction | Released |

Private molecular graph computation is protected using MPC protocols.

The protein branch is evaluated using a public GPU-based Gated-CNN
component.

---

## 4. Implementation Highlights

### 4.1 GPU-Accelerated MPC Inference

The secure inference backend uses GPU-based MPC operators.

Accelerated components include:

-   Secure matrix operations.
-   GCN layers.
-   Masked pooling.
-   Affinity prediction.

### 4.2 Secure Graph Processing

Drug molecules are represented as dense graphs.

The system performs:

-   Secret-shared graph feature processing.
-   Secure graph convolution.
-   Secure adjacency normalization.

Adjacency normalization is executed inside the secure inference
pipeline.

---

### 4.3 Protein Gated-CNN Worker

The protein branch uses a public Gated-CNN inference path.

The current implementation introduces:

-   Persistent Protein worker.
-   Reduced model initialization overhead.
-   GPU execution inside the timed inference path.

---

### 4.4 Full-Key-RAM FSS Loading

The current implementation supports direct full FSS key loading.

    Offline Key File

           |
           v

    Direct Full RAM Buffer

           |
           v

    GPU MPC Evaluation

The full-key-RAM mode avoids per-chunk key file access during MPC execution.

---

## 5. Performance Evaluation

### 5.1 Validation Environment

| Component | Configuration |
|---|---|
| GPU | 2 × NVIDIA H800 PCIe |
| CPU | Intel Xeon Platinum 8458P |
| Memory | 200 GB RAM |
| CUDA | 12.8 |
| PyTorch | 2.8.0 + cu128 |

### 5.2 Validation Result

Configuration:

| Parameter | Value |
|---|---|
| Dataset | Davis |
| Samples | 3 |
| Micro-batch | 8 |
| Bitwidth | 64 |
| Scale | 12 |
| Key Mode | full-key-ram |
| Protein Mode | Persistent Protein Worker v1 |


Overall timing:

| Metric | Time |
|---|---:|
| Offline Dealer Key Generation | 9.435 s |
| Online Setup | 3.237 s |
| Online MPC Compute | 2.791 s |
| End-to-End Runtime | 6.028 s |
| Throughput | 1.075 samples/s |
| Online Time per Sample | 0.930 s/sample |


### 5.3 Detailed Profiling

| Module | Runtime |
|---|---:|
| Protein Gated-CNN | 211 ms |
| Party 0 MPC Compute | 347 ms |
| Party 1 MPC Compute | 347 ms |
| Key preload Party 0 | 694 ms |
| Key preload Party 1 | 686 ms |
| MPC Communication | 126 MB / party |

---

## 6. Correctness Verification

The implementation validates:

-   Secure inference execution.
-   Offline/online separation.
-   Correct output reconstruction.
-   Removal of padded outputs.

Example:

    PASS: N=3, fixed B=8, chunks=1, padded=8, returned=3

    OFFLINE/ONLINE SEPARATION: PASS

---

## 7. Deployment Notes

The repository provides:

-   GPU-MPC implementation.
-   Dataset preparation scripts.
-   Model preparation utilities.
-   Environment configuration instructions.
-   Local and multi-party execution examples.

Detailed environment configuration is provided in:

    ENVIRONMENT_SETUP.md

## 8. Current Limitations

The current implementation focuses on correctness validation,
reproducible deployment, and performance measurement.

The submitted version uses direct full-RAM FSS key loading during online inference.

Further engineering improvements may be considered in future versions, but are not part of the submitted implementation.