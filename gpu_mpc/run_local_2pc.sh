#!/usr/bin/env bash
# run_local_2pc.sh <sample_dir> <key_dir> <weights_bin>
#
# Runs a single-machine 2PC DeepDTAGen inference:
#   1. Dealer phase (role 0) — key generation for both parties
#   2. Online phase (role 1) — party 0 and party 1 in parallel on 127.0.0.1
#
# Prints "AFFINITY=<float>" to stdout when the online phase completes.
#
# Usage:
#   export DDG_WEIGHTS_BIN=/path/to/weights.bin   # OR pass as $3
#   ./run_local_2pc.sh <sample_dir> <key_dir> [weights_bin]
#
# All paths should be absolute.

set -euo pipefail

SAMPLE_DIR="${1:?Usage: $0 <sample_dir> <key_dir> [weights_bin]}"
KEY_DIR="${2:?Usage: $0 <sample_dir> <key_dir> [weights_bin]}"
WEIGHTS_BIN="${3:-${DDG_WEIGHTS_BIN:-}}"

if [[ -z "$WEIGHTS_BIN" ]]; then
    echo "[run_local_2pc.sh] ERROR: weights.bin path required as \$3 or DDG_WEIGHTS_BIN" >&2
    exit 1
fi

# Offline artifacts (weights.bin + secret-share .dat) are regeneratable and
# gitignored — they must be produced by offline prep before running.
# See: reference.offline_prepare.prepare_sample() / README step "Generate offline artifacts".
if [[ ! -f "$WEIGHTS_BIN" ]]; then
    echo "[run_local_2pc.sh] ERROR: weights.bin not found: $WEIGHTS_BIN" >&2
    echo "  Generate it first with reference.offline_prepare.prepare_sample(...) — see README." >&2
    exit 1
fi
if ! compgen -G "${1%/}/*.dat" > /dev/null; then
    echo "[run_local_2pc.sh] ERROR: no secret-share (.dat) files in sample dir: $1" >&2
    echo "  Run offline prep first: reference.offline_prepare.prepare_sample(...) — see README." >&2
    exit 1
fi

# Runtime env required by the binary (CUDA 12.1 on WSL)
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export DDG_WEIGHTS_BIN="$WEIGHTS_BIN"

# The binary lives alongside this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/deepdtagen_inference"

BW=${BW:-32}
SCALE=${SCALE:-12}
IP=${IP:-127.0.0.1}
BATCH=${BATCH:-1}   # internal MPC micro-batch

if (( BATCH < 1 || BATCH > 128 )); then
    echo "[run_local_2pc.sh] ERROR: BATCH must be in [1,128], got $BATCH" >&2
    exit 1
fi

# Dealer FSS key material grows approximately linearly with the internal
# micro-batch size.  Use conservative power-of-two capacities measured from
# BW64 DeepDTAGen experiments.  Explicit DDG_KEYBUF_CAP_GB still overrides
# this mapping for diagnostics/experiments.
if [[ -z "${DDG_KEYBUF_CAP_GB:-}" ]]; then
    if (( BATCH <= 4 )); then
        DDG_KEYBUF_CAP_GB=2
    elif (( BATCH <= 8 )); then
        DDG_KEYBUF_CAP_GB=4
    elif (( BATCH <= 16 )); then
        DDG_KEYBUF_CAP_GB=8
    elif (( BATCH <= 32 )); then
        DDG_KEYBUF_CAP_GB=16
    elif (( BATCH <= 64 )); then
        DDG_KEYBUF_CAP_GB=32
    else
        DDG_KEYBUF_CAP_GB=64
    fi
fi

export DDG_KEYBUF_CAP_GB

echo "[run_local_2pc.sh] BW=$BW SCALE=$SCALE BATCH=$BATCH KEYBUF_CAP=${DDG_KEYBUF_CAP_GB}GiB"

# Key dir must end with '/' (binary concatenates expName without separator)
mkdir -p "$KEY_DIR"
KEY_DIR_SLASH="${KEY_DIR%/}/"

echo "[run_local_2pc.sh] dealer keygen for party 0 (BATCH=$BATCH) ..."
"$BINARY" $BW $SCALE 0 0 "$KEY_DIR_SLASH" "$SAMPLE_DIR" "$BATCH" 2>&1

echo "[run_local_2pc.sh] dealer keygen for party 1 ..."
"$BINARY" $BW $SCALE 0 1 "$KEY_DIR_SLASH" "$SAMPLE_DIR" "$BATCH" 2>&1

echo "[run_local_2pc.sh] starting online parties ..."

# Party 1 goes first (it listens); party 0 connects.
# Use a temp file to capture party-0 stdout so we can grep AFFINITY= from it.
P0_LOG="$(mktemp)"
P1_LOG="$(mktemp)"
trap "rm -f '$P0_LOG' '$P1_LOG'" EXIT

"$BINARY" $BW $SCALE 1 1 "$KEY_DIR_SLASH" "$SAMPLE_DIR" "$BATCH" "$IP" >"$P1_LOG" 2>&1 &
P1_PID=$!

# Small sleep so party 1 is listening before party 0 connects
sleep 1

"$BINARY" $BW $SCALE 1 0 "$KEY_DIR_SLASH" "$SAMPLE_DIR" "$BATCH" "$IP" >"$P0_LOG" 2>&1 &
P0_PID=$!

# Wait for both — propagate non-zero exit
P1_RC=0; P0_RC=0
wait "$P1_PID" || P1_RC=$?
wait "$P0_PID" || P0_RC=$?

cat "$P1_LOG" >&2

# Party-0 AFFINITY lines are emitted exactly once below in a machine-readable
# form.  Keep the rest of party-0 diagnostics on stderr without duplicating
# predictions when stdout/stderr are merged by tee.
grep -v '^AFFINITY' "$P0_LOG" >&2 || true

if [[ $P1_RC -ne 0 || $P0_RC -ne 0 ]]; then
    echo "[run_local_2pc.sh] ERROR: online process exited non-zero (P0=$P0_RC P1=$P1_RC)" >&2
    exit 1
fi

# Emit predictions from party 0.
# BATCH=1 prints AFFINITY=<value>.
# Batched inference prints AFFINITY[i]=<value>.
if (( BATCH == 1 )); then
    AFFINITY_LINE="$(grep '^AFFINITY=' "$P0_LOG" | tail -1 || true)"

    if [[ -z "$AFFINITY_LINE" ]]; then
        echo "[run_local_2pc.sh] ERROR: AFFINITY= line not found in party-0 output" >&2
        exit 1
    fi

    echo "$AFFINITY_LINE"
else
    AFFINITY_COUNT="$(grep -c '^AFFINITY\[' "$P0_LOG" || true)"

    if (( AFFINITY_COUNT != BATCH )); then
        echo "[run_local_2pc.sh] ERROR: expected $BATCH affinity outputs, got $AFFINITY_COUNT" >&2
        exit 1
    fi

    grep '^AFFINITY\[' "$P0_LOG"
fi
