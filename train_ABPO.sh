export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1

LENGTH=6000
RUN_NAME=ABPO_350step
MODEL=./DeepSeek-R1-Distill-Qwen-1.5B
WS=/data

N_GPUS=8 # GPU per node
TP=1     # tensor parallel for vLLM rollout


BATCH_SIZE=64 
SAVE_FREQ=50 #多少个step存一个（你需要用总数据量下整除data.train_batch_size得到，比如数据量1000，data.train_batch_size=128，则SAVE_FREQ=1000//128=7）
ROLLOUT_BS=128
ROLLOUT_N=16


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./train.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.shuffle=False  \
    data.max_prompt_length=768 \
    data.max_response_length=${LENGTH} \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$ROLLOUT_BS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$ROLLOUT_BS \
    reward_model.enable=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.default_local_dir=$WS/checkpoints/${RUN_NAME} \
    trainer.default_hdfs_dir=null \
    trainer.logger=['console'] \
    trainer.project_name='verl_math' \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${SAVE_FREQ} \
    trainer.total_training_steps=350 \
