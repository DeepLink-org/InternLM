JOB_NAME = "llama2_1.3b_train"

model_type = "LLAMA2"
DO_ALERT = False

# SEQ_LEN = 2048
SEQ_LEN = 512
HIDDEN_SIZE = 2048
NUM_ATTENTION_HEAD = 32
NUM_KV_ATTENTION_HEAD = 8
MLP_RATIO = 3.5
NUM_LAYER = 17
VOCAB_SIZE = 103168


# MODEL_ONLY_FOLDER = "local:/mnt/cachenew/wanglei9/llama/30B/"
MODEL_ONLY_FOLDER = "local:/mnt/cachenew/lilingjie/share/llama2/30B/"
# Ckpt folder format:
# fs: 'local:/mnt/nfs/XXX'
SAVE_CKPT_FOLDER = f"local:llm_ckpts-{JOB_NAME}"
# LOAD_CKPT_FOLDER = "local:/mnt/cachenew/wanglei9/llama/30B/"
LOAD_CKPT_FOLDER = "local:/mnt/cachenew/lilingjie/share/llama2/30B/"

# boto3 Ckpt folder format:
# import os
# BOTO3_IP = os.environ["BOTO3_IP"] # boto3 bucket endpoint
# SAVE_CKPT_FOLDER = f"boto3:s3://model_weights.{BOTO3_IP}/internlm"
# LOAD_CKPT_FOLDER = f"boto3:s3://model_weights.{BOTO3_IP}/internlm/snapshot/1/"
CHECKPOINT_EVERY = 50
# CHECKPOINT_EVERY = 1
ckpt = dict(
    enable_save_ckpt=False,  # enable ckpt save.
    save_ckpt_folder=SAVE_CKPT_FOLDER,  # Path to save training ckpt.
    # load_ckpt_folder= dict(path=MODEL_ONLY_FOLDER, content=["model"], ckpt_type="normal"),
    # load_ckpt_folder="local:llm_ckpts/",
    # 'load_ckpt_info' setting guide:
    # 1. the 'path' indicate ckpt path,
    # 2. the 'content‘ means what states will be loaded, support: "model", "sampler", "optimizer", "scheduler", "all"
    # 3. the ’ckpt_type‘ means the type of checkpoint to be loaded, support: "internlm", "llama", "hf_llama".
    # load_ckpt_info=dict(
    #     path=MODEL_ONLY_FOLDER, content=("model",), ckpt_type="hf_llama"
    # ),
    # 'auto_resume' is designed to automatically load the latest checkpoint from 'save_ckpt_folder' when encountering
    # training interruptions/hangs caused by hardware failures, using a scheduling system (such as k8s/slurm)
    # with an automatic restart mechanism upon training reboot.
    # Please be aware that if `auto_resume` is not set (its default value is True), it will not load the checkpoint
    # path specified in `load_ckpt_info` by default.
    # If you want to initialize your model weights from another model, you must set `auto_resume` to False.
    # If you want to train from scratch, please set `auto_resume` to False and 'load_ckpt_info' to None.
    auto_resume=False,
    checkpoint_every=CHECKPOINT_EVERY,
    async_upload=True,  # async ckpt upload. (only work for boto3 ckpt)
    async_upload_tmp_folder="/dev/shm/internlm_tmp_ckpt/",  # path for temporarily files during asynchronous upload.
    oss_snapshot_freq=int(CHECKPOINT_EVERY / 2),  # snapshot ckpt save frequency.
)

# TRAIN_FOLDER = "/mnt/lustre/share_data/PAT/datasets/dolly_tokenizer_llama/train"
# VALID_FOLDER = "/mnt/lustre/share_data/PAT/datasets/dolly_tokenizer_llama/valid"
TRAIN_FOLDER = "/mnt/cache/share_data/PAT/datasets/lm_data/data/alpaca.bin/train"
VALID_FOLDER = "/mnt/cache/share_data/PAT/datasets/lm_data/data/alpaca.bin/valid"


# TRAIN_FOLDER = "/mnt/lustre/share_data/wanglei/data/split/"  # "/path/to/dataset"
# VALID_FOLDER = "/mnt/lustre/share_data/wanglei/data/split/"  # "/path/to/dataset"
data = dict(
    seq_len=SEQ_LEN,
    # micro_num means the number of micro_batch contained in one gradient update
    # micro_num=4,
    micro_num=1,
    # packed_length = micro_bsz * SEQ_LEN
    # micro_bsz=2,
    micro_bsz=1,
    # defaults to the value of micro_num
    valid_micro_num=4,
    # valid_micro_num=1,
    # defaults to 0, means disable evaluate
    # valid_every=50,
    valid_every=0,
    pack_sample_into_one=False,
    total_steps=50000,
    skip_batches="",
    # rampup_batch_size (str): A string with three space-separated integers representing the
    #       starting batch size, the increment, and the number of steps between
    #       each increment. For example, "192 24 8" means that the batch size (micro_num)
    #       starts at 192 and increases by 24 every 8 steps. Defaults to None.
    #       (IMPORTANT): The interval step size is 'micro_bsz'.
    rampup_batch_size="",
    # Datasets with less than 50 rows will be discarded
    min_length=50,
    train_folder=TRAIN_FOLDER,
    valid_folder=VALID_FOLDER,
    empty_cache_and_diag_interval=200,
    diag_outlier_ratio=1.1,
)

grad_scaler = dict(
    fp16=dict(
        # the initial loss scale, defaults to 2**16
        initial_scale=2**16,
        # the minimum loss scale, defaults to None
        min_scale=1,
        # the number of steps to increase loss scale when no overflow occurs
        growth_interval=1000,
    ),
    # the multiplication factor for increasing loss scale, defaults to 2
    growth_factor=2,
    # the multiplication factor for decreasing loss scale, defaults to 0.5
    backoff_factor=0.5,
    # the maximum loss scale, defaults to None
    max_scale=2**24,
    # the number of overflows before decreasing loss scale, defaults to 2
    hysteresis=2,
)

hybrid_zero_optimizer = dict(
    # Enable low_level_optimzer overlap_communication
    overlap_sync_grad=True,
    overlap_sync_param=False,
    # bucket size for nccl communication params
    reduce_bucket_size=512 * 1024 * 1024,
    # grad clipping
    clip_grad_norm=1.0,
)

loss = dict(
    label_smoothing=0,
)

adam = dict(
    lr=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_beta2_c=0,
    adam_eps=1e-8,
    weight_decay=0.01,
)

lr_scheduler = dict(
    total_steps=data["total_steps"],
    init_steps=0,  # optimizer_warmup_step
    warmup_ratio=0.01,
    eta_min=1e-5,
    last_epoch=-1,
)

beta2_scheduler = dict(
    init_beta2=adam["adam_beta2"],
    c=adam["adam_beta2_c"],
    cur_iter=-1,
)

model = dict(
    checkpoint=True,  # The proportion of layers for activation aheckpointing, the optional value are True/False/[0-1]
    num_attention_heads=NUM_ATTENTION_HEAD,
    embed_split_hidden=True,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    # parallel_output=True,
    parallel_output=False,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    no_bias=True,
    mlp_ratio=MLP_RATIO,
    apply_post_layer_norm=False,
    # dtype="torch.bfloat16",  # Support: "torch.float16", "torch.half", "torch.bfloat16", "torch.float32", "torch.tf32"
    dtype="torch.float16",
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-5,
    use_flash_attn=False,
    num_chunks=1,  # if num_chunks > 1, interleaved pipeline scheduler is used.
    num_kv_attention_heads=NUM_KV_ATTENTION_HEAD,
    adapt_hf=True,  # if use hf_llama ckpt, please set true
)
"""
zero1 parallel:
    1. if zero1 <= 0, The size of the zero process group is equal to the size of the dp process group,
        so parameters will be divided within the range of dp.
    2. if zero1 == 1, zero is not used, and all dp groups retain the full amount of model parameters.
    3. zero1 > 1 and zero1 <= dp world size, the world size of zero is a subset of dp world size.
        For smaller models, it is usually a better choice to split the parameters within nodes with a setting <= 8.
pipeline parallel (dict):
    1. size: int, the size of pipeline parallel.
    2. interleaved_overlap: bool, enable/disable communication overlap when using interleaved pipeline scheduler.
tensor parallel: tensor parallel size, usually the number of GPUs per node.
"""
parallel = dict(
    # zero1=dict(size=8, fsdp=False),
    # tensor=8,
    zero1=dict(size=1, fsdp=False),
    tensor=1,
    pipeline=dict(size=1, interleaved_overlap=True),
    sequence_parallel=False,
)

cudnn_deterministic = False
cudnn_benchmark = False

monitor = dict(
    # feishu alert configs
    alert=dict(
        enable_feishu_alert=DO_ALERT,
        feishu_alert_address=None,  # feishu webhook to send alert message
        light_monitor_address=None,  # light_monitor address to send heartbeat
        alert_file_path=f"llm_alter/{JOB_NAME}_alert.log",
    ),
)
