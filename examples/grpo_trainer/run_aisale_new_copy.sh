set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_BASE_URL=https://api.bandw.top
wandb login edfcef4f9801daf389c28a3b7bc6b75073fdccfb

train_files="/mnt/bn/motor-nlp-team/users/aiqiuyuan/verl_debug/data/0515/data_ai_baseline/train.parquet"
val_files="/mnt/bn/motor-nlp-team/users/aiqiuyuan/verl_debug/data/0515/data_ai_baseline/val.parquet"
test_files="/mnt/bn/motor-nlp-team/users/aiqiuyuan/verl_debug/data/0515/data_ai_baseline/test.parquet"
score_log_path="/mnt/bn/motor-nlp-team/users/aiqiuyuan/verl_debug/score_records/score_records_0708_ai_baseline_new_reward_4.jsonl"
debug_log_path="/mnt/bn/motor-nlp-team/users/aiqiuyuan/verl_debug/debug_records/debug_records_0708_ai_baseline_new_reward_4.jsonl"
tool_log_path="/mnt/bn/motor-nlp-team/users/aiqiuyuan/verl_debug/tool_records/tool_records_0708_ai_baseline_new_reward_4.jsonl"


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.test_files="$test_files" \
    data.score_log_path="$score_log_path" \
    data.debug_log_path="$debug_log_path" \
    data.tool_log_path="$tool_log_path" \
    data.train_batch_size=16 \
    data.max_prompt_length=12288 \
    data.max_response_length=1024 \
    ++actor_rollout_ref.rollout.max_num_batched_tokens=14000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/bn/motor-nlp-team/models/LLM/base_models/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path=/mnt/bn/motor-nlp-team/users/aiqiuyuan/verl_debug/verl/utils/reward_score/ai_sale_0722_baseline.py \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='aisale_0708' \
    trainer.experiment_name='ai_baseline_new_reward_4' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="/mnt/bn/motor-nlp-team/users/aiqiuyuan/verl_debug/sft/global_step_1101" \
    trainer.total_epochs=6 $@