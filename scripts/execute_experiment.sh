#!/usr/bin/env bash
# Run one configured experiment once.
#
# Usage:
#   ./scripts/auto_paralel_exps.sh <exp_num>
#
# The historical file name is kept for backwards compatibility, but this script
# does not launch parallel/background jobs and does not repeat experiments.

set -Eeuo pipefail

usage() {
    echo "Usage: $0 <exp_num>" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 1
fi

EXP_NUM="$1"

if ! [[ "$EXP_NUM" =~ ^[0-9]+$ ]]; then
    echo "Error: <exp_num> must be a number." >&2
    usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER="$PROJECT_ROOT/scripts/run_multiple_experiments.py"
CONSTANTS_LOCK_DIR="$PROJECT_ROOT/utils/constants/exp${EXP_NUM}"

cleanup_stale_constants() {
    # The constants folder is only a temporary hand-off for one_run_network.py.
    # Leftovers indicate stale state from an interrupted previous run.
    rm -rf "$CONSTANTS_LOCK_DIR"
}

cleanup_stale_constants

(
    cd "$PROJECT_ROOT"
    python3 "$RUNNER" "$EXP_NUM"
)

cleanup_stale_constants

echo "Experiment ${EXP_NUM} completed successfully."
