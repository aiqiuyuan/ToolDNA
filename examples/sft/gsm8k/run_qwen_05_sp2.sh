set -x
set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export HYDRA_FULL_ERROR=1
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_BASE_URL=https://api.bandw.top
wandb login edfcef4f9801daf389c28a3b7bc6b75073fdccfb

train_files="/mnt/bn/motor-nlp-team/users/aiqiuyuan/verl_debug/data/0515/data_human_baseline/train_sft.parquet"
val_files="/mnt/bn/motor-nlp-team/users/aiqiuyuan/verl_debug/data/0515/data_human_baseline/val_sft.parquet"
test_files="/mnt/bn/motor-nlp-team/users/aiqiuyuan/verl_debug/data/0515/data_human_baseline/test_sft.parquet"

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-6 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=1 \
    +trainer.precision=bf16 \
    +trainer.gradient_accumulation_steps=4 \
    +model.gradient_checkpointing=true \
    model.partial_pretrain=/mnt/bn/motor-nlp-team/models/LLM/base_models/Qwen2.5-7B-Instruct \
    trainer.default_local_dir=/mnt/bn/motor-nlp-team/users/aiqiuyuan/verl_debug/sft \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=ai_sale_sft \
    trainer.logger=['console'] \
    trainer.total_training_steps=null \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    data.max_length=12288 \
    use_remove_padding=true
