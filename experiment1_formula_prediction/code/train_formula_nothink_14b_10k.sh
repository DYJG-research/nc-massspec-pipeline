#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Experiment 1: Qwen3-14B no-thinking listwise formula reranker
#
# Default:
#   - Dataset: locked v4 10k SFT data
#   - Model: Qwen3-14B
#   - GPUs: 6,7
#   - LoRA + BF16 + DeepSpeed ZeRO-2
#   - 3 epochs
#   - Effective batch size = 2 GPUs * 1 * grad_accum 12 = 24
#
# Usage:
#   bash train_formula_nothink_14b_10k.sh
#
# Override dataset/output/GPU:
#   DATASET=/path/to/train30k.jsonl \
#   OUTPUT_DIR=/path/to/output \
#   CUDA_DEVICES=4,5 \
#   bash train_formula_nothink_14b_10k.sh
#
# Resume:
#   RESUME_CKPT=/path/to/output/vx-xxx/checkpoint-xxx \
#   bash train_formula_nothink_14b_10k.sh
# =============================================================================

MODEL="${MODEL:?MODEL (Qwen3-14B base model path) is required}"

DATASET="${DATASET:-data/exp1/formula_nothink_sft_v4_train10k.jsonl}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_model/14b-nothink-listwise-10k}"

CUDA_DEVICES="${CUDA_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

# Two visible GPUs -> two DDP processes.
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MASTER_PORT="${MASTER_PORT:-29501}"

# Keep the NCCL settings from the previous working training script.
unset LD_LIBRARY_PATH || true
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

# Helps reduce CUDA allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Optional exact-resume checkpoint.
RESUME_CKPT="${RESUME_CKPT:-}"

echo "================================================================================"
echo "Qwen3-14B no-thinking listwise formula reranker"
echo "================================================================================"
echo "MODEL=${MODEL}"
echo "DATASET=${DATASET}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "RESUME_CKPT=${RESUME_CKPT:-<none>}"
echo "================================================================================"

if [[ ! -d "${MODEL}" ]]; then
    echo "[ERROR] model directory does not exist: ${MODEL}" >&2
    exit 1
fi

if [[ ! -f "${DATASET}" ]]; then
    echo "[ERROR] training dataset does not exist: ${DATASET}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# -----------------------------------------------------------------------------
# Preflight: GPU mapping
# -----------------------------------------------------------------------------
python - <<'PY'
import torch

print("device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"cuda:{i} name={p.name} uuid={p.uuid}")
PY

# -----------------------------------------------------------------------------
# Preflight: validate JSONL and listwise assistant targets.
# This scans the dataset once. For 10k/30k JSONL this is cheap relative to SFT.
# -----------------------------------------------------------------------------
DATASET_PATH="${DATASET}" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["DATASET_PATH"])

n = 0
bad = 0
rank_lengths = []

with path.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        n += 1
        try:
            obj = json.loads(line)
            messages = obj["messages"]
            assert isinstance(messages, list) and len(messages) >= 3
            assistant = messages[-1]
            assert assistant["role"] == "assistant"

            target = json.loads(assistant["content"])
            ranked = target["ranked_candidates"]
            assert isinstance(ranked, list) and len(ranked) >= 1

            ids = []
            for expected_rank, item in enumerate(ranked, 1):
                assert item["rank"] == expected_rank
                assert "candidate_id" in item
                assert "formula" in item
                assert "adduct" in item
                ids.append(item["candidate_id"])

            assert len(ids) == len(set(ids))
            rank_lengths.append(len(ranked))

        except Exception as exc:
            bad += 1
            print(f"[BAD] line={line_no}: {exc}")
            if bad >= 10:
                break

print(f"dataset_rows={n}")
print(f"bad_rows={bad}")
if rank_lengths:
    print(
        "ranked_candidates:",
        f"min={min(rank_lengths)}",
        f"max={max(rank_lengths)}",
        f"mean={sum(rank_lengths)/len(rank_lengths):.2f}",
    )

if n == 0 or bad > 0:
    raise SystemExit("Dataset preflight failed.")
PY

# -----------------------------------------------------------------------------
# Resume arguments
# -----------------------------------------------------------------------------
RESUME_ARGS=()

if [[ -n "${RESUME_CKPT}" ]]; then
    if [[ ! -d "${RESUME_CKPT}" ]]; then
        echo "[ERROR] resume checkpoint does not exist: ${RESUME_CKPT}" >&2
        exit 1
    fi

    RESUME_ARGS+=(
        --resume_from_checkpoint "${RESUME_CKPT}"
    )
fi

# =============================================================================
# Training
# =============================================================================
#
# Important no-thinking handling for Qwen3:
#   --add_non_thinking_prefix true
#       Let ms-swift's Qwen3 template add the empty non-thinking prefix.
#
#   --loss_scale ignore_empty_think
#       Do not optimize the empty <think></think> prefix itself.
#
# Do NOT manually add <think></think> to the JSONL.
#
# Validation:
#   2% of the training JSONL is held out internally.
#
# Checkpoints:
#   Save every epoch and keep optimizer/RNG state so exact resume remains
#   possible with --resume_from_checkpoint.
# =============================================================================

swift sft \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 24 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --split_dataset_ratio 0.02 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --logging_steps 1 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 3 \
    --save_only_model false \
    --output_dir "${OUTPUT_DIR}" \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    "${RESUME_ARGS[@]}"

echo "================================================================================"
echo "Training finished."
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "================================================================================"
