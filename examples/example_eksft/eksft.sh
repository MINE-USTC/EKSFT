#!/bin/bash
cd /EKSFT

SWANLAB_API_KEY=xxxxxxxxxxxxx
swanlab login --api-key ${SWANLAB_API_KEY}


PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=1e-5
EPOCHS=8.0

MODEL_PATH=xxx/Qwen3-8B

# EKSFT 参数
top_k_ratio="0.2"
LAMBDA_ENTROPY="0.05"
LAMBDA_KL="0.05"

# 创建log目录
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"

      
# 动态输出目录和运行名称
OUTPUT_DIR="/xxxxx/Qwen3-8B-KL${LAMBDA_KL}-ENTROPY${LAMBDA_ENTROPY}-top_k${top_k_ratio}"
SWANLAB_RUN_NAME="Qwen3-8B-KL${LAMBDA_KL}-ENTROPY${LAMBDA_ENTROPY}-top_k${top_k_ratio}"

# 日志文件名（建议带时间戳）
LOG_FILE="${LOG_DIR}/${SWANLAB_RUN_NAME}_$(date +'%Y%m%d_%H%M%S').log"

# 开始训练并记录日志
# 注意: EKSFT 参数需要使用 eksft_ 前缀
deepspeed \
--include localhost:0,1,2,3,4,5,6,7 \
src/train.py \
    --model_name_or_path ${MODEL_PATH} \
    --eksft_output_dir ${OUTPUT_DIR} \
    --eksft_top_k_ratio ${top_k_ratio} \
    --eksft_lambda_entropy ${LAMBDA_ENTROPY} \
    --eksft_lambda_kl ${LAMBDA_KL} \
    --eksft_is_union_mask true \
    --trust_remote_code \
    --stage eksft \
    --do_train \
    --finetuning_type full \
    --dataset openr1_math_3k_sft \
    --template qwen3_nothink \
    --cutoff_len 20000 \
    --max_samples 100000 \
    --overwrite_cache \
    --preprocessing_num_workers 128 \
    --dataloader_num_workers 8 \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps 1 \
    --save_strategy epoch \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 0.1 \
    --warmup_ratio 0.01 \
    --save_total_limit 10 \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model true \
    --report_to none \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCHS} \
    --lr_scheduler_type cosine \
    --bf16 \
    --use_swanlab true \
    --swanlab_api_key ${SWANLAB_API_KEY} \
    --swanlab_run_name ${SWANLAB_RUN_NAME} \
    --deepspeed /xxx/examples/deepspeed/ds_z3_offload_config.json \
    --ddp_timeout 180000000
    2>&1 | tee "${LOG_FILE}"