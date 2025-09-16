#!/bin/bash
DEFAULT_LOCALHOST_CONFIG="localhost:0,1,3,4,5,6"
DEFAULT_MASTER_PORT=29500
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1
DEFAULT_NUM_EXPERTS=4
DEFAULT_TOP_K_EXPERTS=2
DEFAULT_MOE_MODE="sparse"
DEFAULT_NUM_TRAIN_EPOCHS=1

LOCALHOST_CONFIG=${1:-$DEFAULT_LOCALHOST_CONFIG}
MASTER_PORT=${2:-$DEFAULT_MASTER_PORT}
GRADIENT_ACCUMULATION_STEPS=${3:-$DEFAULT_GRADIENT_ACCUMULATION_STEPS}
NUM_EXPERTS=${4:-$DEFAULT_NUM_EXPERTS}
TOP_K_EXPERTS=${5:-$DEFAULT_TOP_K_EXPERTS}
MOE_MODE=${6:-$DEFAULT_MOE_MODE}
NUM_TRAIN_EPOCHS=${7:-$DEFAULT_NUM_TRAIN_EPOCHS}

OUTPUT_DIR="./output/llava-qwen-7b-moe-e${NUM_EXPERTS}k${TOP_K_EXPERTS}-${GRADIENT_ACCUMULATION_STEPS}-${MOE_MODE}"

export MASTER_PORT="$MASTER_PORT"

# 运行 deepspeed 命令
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed --include=$LOCALHOST_CONFIG --master_port "$MASTER_PORT" llava/train/train_mem.py \
    --moe_mode "$MOE_MODE" \
    --num_experts "$NUM_EXPERTS" \
    --top_k_experts "$TOP_K_EXPERTS" \
    --only_lora_ffn True \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path ./checkpoints/llava-qwen-7b-ft-lora-combine\
    --deepspeed ./scripts/zero2.json  \
    --lora_enable True \
    --version qwen_1_5 \
    --data_path ./outputs_prepare_rag/rag_llava_6144.json \
    --image_folder ./dataset/articles \
    --vision_tower ./checkpoints/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"\
    --ddp_find_unused_parameters=False \
    --moe_enable True \
    --freeze_mm_mlp_adapter True \
