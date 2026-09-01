
# DeepDTAGen GPU-MPC Validation Results

This document records validation experiments for the current
the submitted implementation.

This file records validation results and benchmark measurements for the
submitted implementation.
It is not the main user documentation.

---

# 1. Validation Environment

All experiments were performed on:

| Item | Configuration |
|---|---|
| CPU | Intel Xeon Platinum 8458P |
| Logical CPU | 176 |
| GPU | 2 × NVIDIA H800 PCIe |
| GPU Memory | 81GB × 2 |
| RAM | 1 TB |
| CUDA | 12.8 |
| PyTorch | 2.8.0+cu128 |
| CUDA Runtime | cu128 |

---

# 2. Code Version

Current validation branch:

```text
Current submission
```

Current commit:

```text
67d8176 Add persistent Protein worker v1
```

Validated features:

- Direct full-RAM FSS key evaluation
- Persistent Protein worker
- Secure online adjacency normalization
- Two-party GPU MPC inference

---

# 3. Experiment Configuration

Unless otherwise specified:

| Parameter | Value |
|---|---|
| MPC parties | 2 |
| Ring bit width | 64 |
| Fixed-point scale | 12 |
| Dataset | Davis |
| Model checkpoint | KIBA |
| Key mode | full-key-ram |
| Protein mode | Persistent Protein Worker v1 |
| A_norm | Secure online |
| Internal batch B | 8 |

---

# 4. Performance Summary

## 4.1 Overall Runtime

Current validated performance:

| Version | N | B | Offline Dealer | Setup | Online Compute | End-to-End | Throughput | Online Time / Sample |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current submission | 3 | 8 | 9.435s | 3.237s | 2.791s | 6.028s | 1.075 sample/s | 0.930s |

Notes:

- Offline Dealer is measured separately.
- End-to-End includes setup and online computation.
- Offline Dealer is not included in online runtime.
- N=3 is a functional smoke test and not a representative large-scale throughput benchmark.

---

# 5. Detailed Profiling

## 5.1 N=3 B=8 Full-RAM Validation

Configuration:

```text
N = 3
B = 8
chunks = 1
BW = 64
SCALE = 12
```

## Key and Setup

| Component | Time |
|---|---:|
| Key preload P0 | 694 ms |
| Key preload P1 | 686 ms |
| Setup total | 3.237 s |

---

## Online Computation

| Component | Time |
|---|---:|
| Protein Worker | 211 ms |
| MPC compute P0 | 347 ms |
| MPC compute P1 | 347 ms |

Communication:

| Party | Bytes |
|---|---:|
| P0 | 126 MB |
| P1 | 126 MB |

---

# 6. FSS Key Size

Configuration:

```text
BW = 64
SCALE = 12
B = 8
Secure A_norm = enabled
```

Key size:

| Item | Size |
|---|---:|
| One chunk / party | 2,954,940,416 bytes |
| One chunk / party | approximately 2.95 GB |
| K0 + K1 | approximately 5.91 GB |

For multiple chunks:

```text
chunks = ceil(N / B)
```

Example:

| Logical N | B | Approximate Key / Party |
|---|---:|---:|
| 8 | 8 | 2.95 GB |
| 16 | 8 | 5.91 GB |
| 128 | 8 | 47.28 GB |
| 512 | 8 | 189.12 GB |

---

# 7. Correctness Validation

Validated cases:

| N | B | Result |
|---|---|---|
| 3 | 8 | PASS |
| 17 | 8 | PASS |

Validation includes:

- Output length equals logical N
- Padding outputs are removed
- Two-party inference completes
- Full-key RAM preload succeeds
- READY/START lifecycle succeeds

---

# 8. Scalability Experiments

Planned experiments:

## 8.1 Micro-batch Search

Search:

```text
B ∈ {8,16,32,...}
```

under fixed logical N.

Metrics:

- Online runtime
- Throughput
- Memory usage

---

## 8.2 Large-N Benchmark

Planned:

```text
N = 64
N = 128
N = 256
```

Measure:

- Key size
- Memory consumption
- Throughput
- End-to-end runtime

---

# 9. Accuracy Validation

Pending.

Planned validation:

- Compare plaintext reference prediction
- Compare MPC prediction
- Compute competition accuracy metric
- Verify accuracy degradation requirement

