# Davis Correctness Validation V2

## Version

Core correctness fix commit:

`40a6dd1 Fix maxpool add and subtraction to use full ring`

Configuration:

- Ring bitwidth: BW=32
- Fixed-point scale: SCALE=12
- Dataset: Davis
- Test samples: 5010
- MPC backend: GPU 2PC / DDGOrca

## Correctness Fix

The max-pool pairwise maximum is implemented as:

`max(a, b) = a + ReLU(b - a)`

Addition and subtraction between Q20.12 fixed-point values preserve the
fixed-point scale and therefore must operate in the full 32-bit ring.

The previous implementation used:

`tmpBw = bw - scale = 20`

for these add/sub operations.

This caused incorrect signed arithmetic before ReLU and produced large
prediction errors on some samples.

The corrected implementation uses:

`tmpBw = bw = 32`

for both evaluation and key generation.

## Float Reference vs C++-Order Q20.12 Fixed Point

Full Davis test set: 5010 samples.

Regression difference:

- MAE: 0.0014520681
- Median: 0.0005574226
- P90: 0.0039746284
- P95: 0.0060619116
- P99: 0.0114235926
- MAX: 0.0282564163

The C++-order fixed-point implementation follows the actual MPC GCN order:

`A @ H -> truncate -> @ W -> truncate`

## Balanced Accuracy Stress Test

Thresholds tested:

- 5.25
- 5.50
- 5.75
- 6.00
- 6.25
- 6.50
- 6.75
- 7.00
- 7.50

Results:

| Threshold | Float BA | C++ Fixed BA | BA Drop |
|---:|---:|---:|---:|
| 5.25 | 87.2691% | 87.1990% | 0.0701% |
| 5.50 | 86.4887% | 86.5682% | -0.0794% |
| 5.75 | 85.9546% | 85.9673% | -0.0127% |
| 6.00 | 85.0821% | 85.1066% | -0.0244% |
| 6.25 | 82.9648% | 82.8981% | 0.0667% |
| 6.50 | 81.8467% | 81.8467% | 0.0000% |
| 6.75 | 81.8306% | 81.8306% | 0.0000% |
| 7.00 | 79.4320% | 79.4320% | 0.0000% |
| 7.50 | 78.0680% | 77.3003% | 0.7677% |

Maximum observed BA drop in this internal threshold stress test:

`0.767690% at threshold 7.50`

## Real MPC vs C++ Fixed Point

67 unique stress inputs were selected from the Davis test set, including:

- threshold-adjacent predictions
- float/fixed classification flips
- large float/fixed regression-error cases
- deterministic control samples

Tested MPC batch sizes:

- BATCH=1
- BATCH=3
- BATCH=4
- BATCH=16

Results:

- 64 / 67 samples matched the C++ fixed-point result at the Q12 integer level.
- 3 / 67 samples differed by exactly one Q12 least-significant bit.
- 0 / 67 samples differed by more than one Q12 LSB.

One Q12 LSB:

`2^-12 = 0.000244140625`

Maximum observed MPC-vs-C++-fixed difference:

`0.000244140625` (one Q12 LSB)

The three one-LSB cases were:

- Davis index 873
- Davis index 1119
- Davis index 3182

## Conclusion

For the tested Davis workload:

1. The full-ring max-pool add/sub fix removes the previously observed large
   MPC prediction errors.
2. The real GPU 2PC implementation closely reproduces the C++ fixed-point
   computation across multiple batch sizes.
3. The C++ Q20.12 computation remains well within the competition's 2%
   balanced-accuracy degradation requirement on the tested thresholds.

This validation does not guarantee performance on the hidden competition
test set or undisclosed evaluation thresholds.
