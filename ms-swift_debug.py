from swift import sft_main, SftArguments,RLHFArguments,rlhf_main
import os
# os.environ["ROOT_IMAGE_DIR"]="/home/user02/LRF/VLM/Data"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
if __name__ == '__main__':
    os.environ['MAX_PIXELS'] = '262114'
    rlhf_main(RLHFArguments(
        rlhf_type='grpo',
        model='/home/user02/SCY/Model/Qwen2.5-VL-3B-Instruct',
        external_plugins=['/home/user02/SCY/thyroid_benchmark/code/swift/plugin.py'],
        reward_funcs=['thinkformat','async_ultrasound_genrm'],
        use_vllm=True,
        vllm_mode='colocate' ,
        vllm_server_host=['10.116.39.70'] ,
        vllm_server_port=[6000] ,
        max_completion_length=4096,
        num_generations=8,
        temperature=1.0,
        system='/home/user02/SCY/thyroid_benchmark/code/swift/prompt.txt',
        dataset=['/home/user02/SCY/thyroid_benchmark/code/swift/data_json/train_RL.json'],
        model_type='qwen2_5_vl',
        template='qwen2_5_vl',
        load_from_cache_file=False,
        split_dataset_ratio=0.0,
        tuner_type='lora',
        lora_rank=8,
        lora_alpha=16,
        target_modules=['all-linear'],
        freeze_vit=True,
        freeze_llm=True,
        freeze_aligner=False,
        torch_dtype='bfloat16',
        attn_impl='flash_attention_2',
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=1e-4,
        gradient_accumulation_steps=4,
        eval_steps=-1,
        save_steps=1000,
        save_total_limit=None,
        logging_steps=5,
        max_length=None,
        output_dir='/home/user02/SCY/thyroid_benchmark/code/swift/checkpoint/debug',
        warmup_ratio=0.05,
        dataloader_num_workers=16,
        dataset_num_proc=64,
        log_completions=True,# 是否记录训练中的模型生成内容
    ))