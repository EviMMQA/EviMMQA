#!/bin/bash

DEFAULT_LOCALHOST_CONFIG="localhost:0,1,2,3,4,5"
DEFAULT_MASTER_PORT=29512
DEFAULT_NUM_TRAIN_EPOCHS=1
DEFAULT_GRADIENT_ACCUMULATION_STEPS=4

LOCALHOST_CONFIG=${1:-$DEFAULT_LOCALHOST_CONFIG}
MASTER_PORT=${2:-$DEFAULT_MASTER_PORT}
GRADIENT_ACCUMULATION_STEPS=${3:-$DEFAULT_GRADIENT_ACCUMULATION_STEPS}
NUM_TRAIN_EPOCHS=${4:-$DEFAULT_NUM_TRAIN_EPOCHS}


HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
deepspeed --include=$LOCALHOST_CONFIG --master_port "$MASTER_PORT" llava/train/train_mem.py \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --output_dir "./output/llava-qwen-7b-ft-lora"    \
    --lora_enable True \
    --deepspeed ./scripts/zero2.json  \
    --model_name_or_path ./checkpoints/llava-next-interleave-qwen-7b-dpo \
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
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"\
    --ddp_find_unused_parameters=False \
