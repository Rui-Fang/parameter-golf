#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

mkdir -p logs

timestamp="$(date +%Y%m%d_%H%M%S)"
: "${RUN_ID:=planc_local_1gpu_${timestamp}}"
: "${NPROC_PER_NODE:=1}"
: "${TORCHRUN:=torchrun}"
: "${DATA_PATH:=./data/datasets/fineweb10B_sp1024}"
: "${TOKENIZER_PATH:=./data/tokenizers/fineweb_1024_bpe.model}"
: "${VOCAB_SIZE:=1024}"
: "${MAX_WALLCLOCK_SECONDS:=4800}"
: "${TRAIN_BATCH_TOKENS:=1048576}"
: "${GRAD_ACCUM_STEPS:=2}"

: "${NUM_FRONT_BLOCKS:=1}"
: "${NUM_CORE_BLOCKS:=3}"
: "${NUM_CORE_LOOPS:=3}"
: "${NUM_BACK_BLOCKS:=1}"
: "${ALIGN_ENABLED:=1}"
: "${ALIGN_MODE:=bias}"
: "${ALIGN_SCALE_CLAMP:=0.125}"
: "${ALIGN_MIX_INIT:=0.0}"
: "${DEPTH_BIAS_ENABLED:=1}"
: "${CARRY_ENABLED:=1}"
: "${CARRY_INIT:=0.05}"
: "${QAT_ENABLED:=1}"
: "${QAT_MODE:=shared_early}"
: "${QAT_CORE_START_FRAC:=0.05}"
: "${QAT_BOUNDARY_START_FRAC:=0.20}"
: "${QAT_AUX_START_FRAC:=1.00}"
: "${VE_LAST_N:=2}"
: "${XSA_LAST_N:=2}"
: "${COLLECT_LOOP_STATS:=1}"
: "${LOOP_STATS_EVERY:=200}"

: "${CONSOLE_LOG:=logs/${RUN_ID}.console.log}"

export RUN_ID
export DATA_PATH
export TOKENIZER_PATH
export VOCAB_SIZE
export MAX_WALLCLOCK_SECONDS
export TRAIN_BATCH_TOKENS
export GRAD_ACCUM_STEPS
export NUM_FRONT_BLOCKS
export NUM_CORE_BLOCKS
export NUM_CORE_LOOPS
export NUM_BACK_BLOCKS
export ALIGN_ENABLED
export ALIGN_MODE
export ALIGN_SCALE_CLAMP
export ALIGN_MIX_INIT
export DEPTH_BIAS_ENABLED
export CARRY_ENABLED
export CARRY_INIT
export QAT_ENABLED
export QAT_MODE
export QAT_CORE_START_FRAC
export QAT_BOUNDARY_START_FRAC
export QAT_AUX_START_FRAC
export VE_LAST_N
export XSA_LAST_N
export COLLECT_LOOP_STATS
export LOOP_STATS_EVERY
export PYTHONUNBUFFERED=1

echo "run_id=$RUN_ID"
echo "console_log=$CONSOLE_LOG"
echo "max_wallclock_seconds=$MAX_WALLCLOCK_SECONDS"
echo "train_batch_tokens=$TRAIN_BATCH_TOKENS"
echo "grad_accum_steps=$GRAD_ACCUM_STEPS"

"$TORCHRUN" --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py 2>&1 | tee "$CONSOLE_LOG"