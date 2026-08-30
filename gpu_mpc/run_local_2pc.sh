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
SCALE=12
IP=127.0.0.1
BATCH=${BATCH:-1}   # samples per forward; batched shares must match (prepare_batch_samples.py)

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
cat "$P0_LOG" >&2

if [[ $P1_RC -ne 0 || $P0_RC -ne 0 ]]; then
    echo "[run_local_2pc.sh] ERROR: online process exited non-zero (P0=$P0_RC P1=$P1_RC)" >&2
    exit 1
fi

# Extract and echo the AFFINITY line from party-0 output
AFFINITY_LINE="$(grep '^AFFINITY=' "$P0_LOG" | tail -1)"
if [[ -z "$AFFINITY_LINE" ]]; then
    echo "[run_local_2pc.sh] ERROR: AFFINITY= line not found in party-0 output" >&2
    exit 1
fi

echo "$AFFINITY_LINE"
