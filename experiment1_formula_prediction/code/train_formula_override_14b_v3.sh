#!/usr/bin/env bash
set -euo pipefail


# =============================================================================
# Experiment 1:
# Qwen3-14B XGBoost override verifier v2
#
# Goal:
#   Learn whether to override XGBoost Top1 candidate.
#
# Decision:
#   KEEP:
#       trust XGB Top1
#
#   SWAP:
#       replace XGB Top1 with Top2
#
#
# Data:
#   override_sft_v2_train_mixed.jsonl
#
#   override:
#       SWAP
#       KEEP_R1
#       KEEP_BOTH_WRONG
#
#   replay:
#       original listwise formula reasoning
#
#
# Init:
#   previous 10k listwise LoRA
#
# Training:
#   1 epoch
#   LoRA
#   BF16
#
# =============================================================================



MODEL="${MODEL:?MODEL (Qwen3-14B base model path) is required}"


INIT_ADAPTER="${INIT_ADAPTER:?INIT_ADAPTER (stage-1 listwise checkpoint-1227 path) is required}"


DATASET="${DATASET:-data/exp1/override_sft_v3_train_mixed.jsonl}"

VAL_DATASET="${VAL_DATASET:-data/exp1/override_sft_v3_dev_mixed.jsonl}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_model/14b-override-v3}"



CUDA_DEVICES="${CUDA_DEVICES:-7}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"


export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

export MASTER_PORT="${MASTER_PORT:-29521}"



NUM_EPOCHS="${NUM_EPOCHS:-1}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"

MAX_LENGTH="${MAX_LENGTH:-8192}"

GRAD_ACCUM="${GRAD_ACCUM:-24}"

SEED="${SEED:-20260821}"


RESUME_CKPT="${RESUME_CKPT:-}"



# -----------------------------------------------------------------------------
# CUDA
# -----------------------------------------------------------------------------

unset LD_LIBRARY_PATH || true


export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"


export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"



echo "======================================================"

echo "Qwen3-14B Override verifier v2"

echo "======================================================"

echo "MODEL=${MODEL}"

echo "INIT_ADAPTER=${INIT_ADAPTER}"

echo "DATASET=${DATASET}"

echo "VAL_DATASET=${VAL_DATASET}"

echo "OUTPUT_DIR=${OUTPUT_DIR}"

echo

echo "CUDA=${CUDA_VISIBLE_DEVICES}"

echo "EPOCH=${NUM_EPOCHS}"

echo "LR=${LEARNING_RATE}"

echo "======================================================"



mkdir -p "${OUTPUT_DIR}"



# -----------------------------------------------------------------------------
# Check paths
# -----------------------------------------------------------------------------

for p in \
"${MODEL}" \
"${INIT_ADAPTER}" \
"${DATASET}" \
"${VAL_DATASET}"
do

    if [[ ! -e "$p" ]]; then
        echo "[ERROR] missing: $p"
        exit 1
    fi

done



# -----------------------------------------------------------------------------
# Dataset check
# -----------------------------------------------------------------------------

TRAIN_PATH="${DATASET}" \
DEV_PATH="${VAL_DATASET}" \
python - <<'PY'

import json
import os


def check(path):

    total=0
    override=0
    replay=0


    with open(path,encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue


            total+=1

            obj=json.loads(line)


            msgs=obj["messages"]

            assert msgs[-1]["role"]=="assistant"


            target=json.loads(
                msgs[-1]["content"]
            )


            if "decision" in target:

                assert target["decision"] in [
                    "KEEP",
                    "SWAP"
                ]

                override+=1


            elif "ranked_candidates" in target:

                replay+=1


            else:

                raise ValueError(
                    target.keys()
                )


    print(
        path,
        "total=",total,
        "override=",override,
        "replay=",replay
    )


    assert total>0



check(os.environ["TRAIN_PATH"])

check(os.environ["DEV_PATH"])


print("dataset check passed")

PY



# -----------------------------------------------------------------------------
# Resume
# -----------------------------------------------------------------------------

RESUME_ARGS=()


if [[ -n "${RESUME_CKPT}" ]]; then

    RESUME_ARGS+=(
        --resume_from_checkpoint "${RESUME_CKPT}"
    )

fi



# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------

swift sft \
    --model "${MODEL}" \
    --adapters "${INIT_ADAPTER}" \
    --load_args false \
    \
    --dataset "${DATASET}" \
    --val_dataset "${VAL_DATASET}" \
    --split_dataset_ratio 0 \
    \
    --train_type lora \
    --torch_dtype bfloat16 \
    \
    --num_train_epochs "${NUM_EPOCHS}" \
    \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    \
    --learning_rate "${LEARNING_RATE}" \
    \
    --max_length "${MAX_LENGTH}" \
    \
    --warmup_ratio 0.03 \
    \
    --seed "${SEED}" \
    --data_seed "${SEED}" \
    \
    --eval_strategy steps \
    --eval_steps 50 \
    \
    --save_strategy steps \
    --save_steps 50 \
    \
    --logging_steps 1 \
    \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    \
    --save_total_limit 3 \
    \
    --save_only_model false \
    \
    --output_dir "${OUTPUT_DIR}" \
    \
    --report_to tensorboard \
    \
    --gradient_checkpointing true \
    \
    --add_non_thinking_prefix true \
    \
    --loss_scale ignore_empty_think \
    \
    "${RESUME_ARGS[@]}"


echo

echo "======================================================"

echo "Training finished"

echo "${OUTPUT_DIR}"

echo "======================================================"