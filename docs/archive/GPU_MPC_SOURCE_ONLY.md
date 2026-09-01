# GPU-MPC Source-Only Repository Structure

## Summary

This document records the cleanup performed to make `gpu_mpc/` source-only by gitignoring regeneratable offline artifacts (weights.bin, secret shares .dat) and adding pre-generation documentation.

## Changes Made

### 1. Updated `.gitignore`
**File**: `idash/mpc/.gitignore`

Added comprehensive ignore rules for generated MPC artifacts:
```gitignore
# Generated MPC artifacts (weights + secret shares)
gpu_mpc/weights.bin
gpu_mpc/weights.bin.json
gpu_mpc/**/*.dat
gpu_mpc/**/batch_manifest.json
gpu_mpc/**/golden_affinities.json

# Generated/intermediate test data
gpu_mpc/davis_mb_*/
gpu_mpc/test_batch*/
gpu_mpc/test_row*/
gpu_mpc/test_sample*/
gpu_mpc/test_dir/
gpu_mpc/shares_tmp/
```

**Rationale**: 
- `weights.bin` (13MB) is generated from the .pth model checkpoint via `export_weights.py`
- `.dat` files contain secret shares generated per-sample via `share_data.py`
- These are deterministic outputs from source-controlled inputs (CSV + .pth)
- Keeping them out of git reduces repo size and avoids binary conflicts

**Verification**: 
```bash
$ git check-ignore gpu_mpc/weights.bin gpu_mpc/davis_mb_0/x_share0.dat
gpu_mpc/weights.bin
gpu_mpc/davis_mb_0/x_share0.dat
```
✅ Files exist on disk but are properly ignored by git.

---

### 2. Added Pre-Generation Documentation to README
**File**: `idash/mpc/README.md`

Added step 5 "Generate offline artifacts (required before first run)":
```python
from reference.offline_prepare import prepare_sample
result = prepare_sample(
    dataset='davis',
    csv_path='data/davis_test.csv',
    row_idx=0,
    out_dir='gpu_mpc/davis_sample',
    scale=12,
    bw=32
)
# Generates:
#   gpu_mpc/davis_sample/sample_0/*.dat (secret shares)
#   gpu_mpc/davis_sample/weights.bin (~13MB)
```

Updated the single-sample test command to reference the generated paths:
```bash
./run_local_2pc.sh davis_sample/sample_0 /tmp/keys_test davis_sample/weights.bin
```

Updated project structure tree to clarify:
```
gpu_mpc/
├── deepdtagen_inference.cu  # Main 2PC entry point
├── run_local_2pc.sh          # Test runner
├── *.dat, weights.bin        # Generated offline artifacts (gitignored)
```

---

### 3. Added Preflight Checks to Offline Scripts

#### `real_gpu_2pc_benchmark.py`
**Lines**: Added `check_prerequisites()` function called in `main()`

Verifies before benchmarking:
- ✅ Compiled binary exists: `gpu_mpc/deepdtagen_inference`
- ✅ Model checkpoints exist: `model/deepdtagen_model_{davis,kiba}.pth`
- ✅ Dataset CSVs exist: `data/{davis_test,kiba_train}.csv`

**Note**: This script is self-contained — it calls `prepare_sample()` inline at line 144 to generate artifacts per-sample during the run, so it does NOT require pre-existing weights.bin/.dat files.

Exit message if prerequisites missing:
```
Prerequisites not met (offline artifacts are auto-generated, but these are required):
  ✗ 2PC binary not found: .../deepdtagen_inference
    Build it:  cd gpu_mpc && make GPU_MPC_ROOT=... BW=32 GPU_ARCH=89 deepdtagen_inference
  ✗ Model weights missing: .../deepdtagen_model_davis.pth
```

---

#### `gpu_mpc/run_local_2pc.sh`
**Lines**: 22-38 (after WEIGHTS_BIN check)

Added artifact existence checks with clear error messages:
```bash
# Offline artifacts (weights.bin + secret-share .dat) are regeneratable and
# gitignored — they must be produced by offline prep before running.
if [[ ! -f "$WEIGHTS_BIN" ]]; then
    echo "[run_local_2pc.sh] ERROR: weights.bin not found: $WEIGHTS_BIN"
    echo "  Generate it first with reference.offline_prepare.prepare_sample(...) — see README."
    exit 1
fi
if ! compgen -G "${1%/}/*.dat" > /dev/null; then
    echo "[run_local_2pc.sh] ERROR: no secret-share (.dat) files in sample dir: $1"
    echo "  Run offline prep first: reference.offline_prepare.prepare_sample(...) — see README."
    exit 1
fi
```

**Why**: Manual invocations of `run_local_2pc.sh` (outside of test/benchmark scripts) require pre-generated artifacts. The preflight check catches missing prep with a helpful message instead of an opaque binary segfault.

---

#### `run_davis_multibatch.sh`
**Lines**: 30-31 (prepare step comment)

Fixed script paths and added clarifying comment:
```bash
# Prepare this batch's samples (stratified offset by batch index)
# NOTE: Generates weights.bin + secret shares (.dat) into $BATCH_DIR
python3 "$SCRIPT_DIR/scripts/dev_tools/prepare_davis_multibatch_slice.py" \
    "$batch" "$BATCH_SIZE" "$BATCH_NAME" 2>&1 | grep -E "(row|Selected|error)" || true
```

Also fixed aggregate validation path reference (line 56):
```bash
echo "  python3 $SCRIPT_DIR/scripts/dev_tools/aggregate_davis_validation.py $NUM_BATCHES"
```

**Why**: The script is self-contained (calls `prepare_davis_multibatch_slice.py` which generates per-batch artifacts before each run), but the comment clarifies the generation step for maintainers.

---

### 4. Updated Framework Guide
**File**: `gpu_mpc/GPU_MPC_FRAMEWORK_GUIDE.md`

Added notice to "文件大小参考" section (line 495):
```markdown
> **注意**: `weights.bin` 和 `.dat` 文件都是离线准备生成的产物（被 gitignore），
> 不提交到代码仓库。使用前需要先运行 `reference.offline_prepare.prepare_sample()`
> 或相关测试脚本生成。参见主 README "Generate offline artifacts" 步骤。
```

Updated file size line:
```
weights.bin            : ~530 MB    (模型权重，离线生成)
```

---

## Workflow Impact

### Before First Run (New User Setup)
Users cloning the repo must now:
1. Compile the binary: `cd gpu_mpc && make ...`
2. **Generate offline artifacts**: Run `prepare_sample(...)` or let test scripts auto-generate
3. Run 2PC inference: `./run_local_2pc.sh ...`

The README step 5 and preflight checks guide users through this.

### Self-Contained Scripts (No Manual Prep)
These scripts auto-generate artifacts and need NO manual pre-generation:
- ✅ `real_gpu_2pc_benchmark.py` — calls `prepare_sample()` inline per sample
- ✅ `run_davis_multibatch.sh` — calls `prepare_davis_multibatch_slice.py` per batch
- ✅ `pytest tests/test_mpc_online_gate.py` — conftest fixtures call offline_prepare

### Manual Scripts (Require Pre-Generated Artifacts)
- ⚠️  `gpu_mpc/run_local_2pc.sh` — the raw shell wrapper expects sample_dir/*.dat + weights.bin to exist
  - Now has preflight checks that catch missing artifacts with helpful error messages

---

## Verification

### Git Status
```bash
$ cd idash/mpc
$ git status --short --untracked-files=all | grep -E "weights\.bin|davis_mb|\.dat"
(none — all ignored)

$ ls -la gpu_mpc/weights.bin gpu_mpc/davis_mb_0/
-rw-r--r-- 1 jiang jiang 12914936 Aug  8 15:42 gpu_mpc/weights.bin

gpu_mpc/davis_mb_0/:
-rw-r--r-- 1 jiang jiang  609408 Aug  9 01:34 adj_share0.dat
-rw-r--r-- 1 jiang jiang  609408 Aug  9 01:34 adj_share1.dat
...
```
✅ Files exist on disk but are gitignored.

### Script Validation
```bash
$ bash -n gpu_mpc/run_local_2pc.sh && echo OK
OK
$ bash -n run_davis_multibatch.sh && echo OK
OK
$ python3 -m py_compile real_gpu_2pc_benchmark.py && echo OK
OK
```
✅ All edited scripts compile without syntax errors.

---

## Related Documentation
- `idash/mpc/README.md` — main setup guide with step 5 "Generate offline artifacts"
- `idash/mpc/gpu_mpc/GPU_MPC_FRAMEWORK_GUIDE.md` — Chinese framework guide with file size reference updated
- `idash/mpc/.gitignore` — ignore rules for generated artifacts
- `idash/mpc/reference/offline_prepare.py` — `prepare_sample()` function that generates artifacts

---

## Rationale Summary

**Why gitignore weights.bin and .dat?**
1. **Deterministic regeneration**: These are pure functions of source-controlled inputs (CSV rows + .pth checkpoints)
2. **Size**: weights.bin alone is ~13-530MB depending on model; .dat files add ~1-2MB per sample
3. **Binary conflicts**: Merging binary files is error-prone; regenerating from source is cleaner
4. **Standard practice**: Similar to gitignoring build artifacts (*.o, *.pyc, node_modules/)

**Why add preflight checks?**
1. **New user onboarding**: Clear error messages guide first-time setup
2. **Debugging efficiency**: "weights.bin not found" is clearer than a segfault in the binary
3. **CI/CD readiness**: Scripts can detect missing setup steps early in the pipeline

**What about the reference sample davis_mb_0/?**
Previously had a contradictory .gitignore exception (`!gpu_mpc/davis_mb_0/`) but files were untracked. Resolved by removing the exception and treating it like any other generated sample — users regenerate it via README step 5 or test scripts.
