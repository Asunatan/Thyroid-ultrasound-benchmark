#!/bin/bash
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=1 \
NODE_RANK=0 \
NPROC_PER_NODE=8 \
MAX_PIXELS=1048576 \
MASTER_PORT=29407 \
ROOT_IMAGE_DIR=/home/user02/SCY/thyroid_benchmark_desensitization \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model  /home/user02/SCY/Model/Qwen3.5-9B \
    --tuner_type full \
    --freeze_vit True \
    --freeze_aligner False \
    --freeze_llm True \
    --dataset  xxxxx \
    --attn_impl flash_attention_2  \
    --torch_dtype bfloat16 \
    --load_from_cache_file True \
    --add_non_thinking_prefix True \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.0 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --save_steps 50 \
    --eval_steps 50 \
    --logging_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 32 \
    --output_dir /home/user02/SCY/thyroid_benchmark/code/swift/checkpoint \
    --dataset_num_proc 128 \
    --deepspeed zero2 \
    --max_length 4096 \
    --packing True \



