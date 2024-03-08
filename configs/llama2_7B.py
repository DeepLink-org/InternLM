model_type = "LLAMA2"

# TRAIN_FOLDER = "/data/lm_data/llama2/alpaca.bin/train"  # "/path/to/dataset"
# VALID_FOLDER = "/data/lm_data/llama2/alpaca.bin/valid"  # "/path/to/dataset"
VOCAB_FILE = "/data/lm_data/llama2/llama2_tokenizer.model"
VOCAB_SIZE = 32000

HIDDEN_SIZE = 4096
NUM_ATTENTION_HEAD = 32
NUM_KV_ATTENTION_HEAD = 8
MLP_RATIO = 2.675
NUM_LAYER = 32
SEQ_LEN = 512

ckpt = dict(
    enable_save_ckpt=False,  # enable ckpt save.
    save_ckpt_folder="local:llm_ckpts",  # Path to save training ckpt.
    auto_resume=False,
    checkpoint_every=50,
    oss_snapshot_freq=int(50 / 2),  # snapshot ckpt save frequency.
)

data = dict(
    seq_len=SEQ_LEN,
    # 一次模型参数更新中会处理的 micro_batch 的数目
    micro_num=2,
    # packed_length = micro_bsz * SEQ_LEN为一次处理的 micro_batch 的数据大小
    micro_bsz=2,
    valid_micro_num=4,
    # defaults to 0, means disable evaluate
    valid_every=0,
    
    pack_sample_into_one=False,
    total_steps=50000,
    skip_batches="",
    rampup_batch_size="",
    # Datasets with less than 50 rows will be discarded
    min_length=50,
    # train_folder=TRAIN_FOLDER,    # 为空则生成随机数据测试
    # valid_folder=VALID_FOLDER,
    empty_cache_and_diag_interval=10,
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
    checkpoint=False,
    num_chunks=1,
    num_attention_heads=NUM_ATTENTION_HEAD,
    embed_split_hidden=True,
    vocab_size=VOCAB_SIZE,
    embed_grad_scale=1,
    parallel_output=True,  # 最后的输出是否需要gather起来，如果不gather的话，每个tensor parallel获取的就是自己对应的结果。注意华为现在强制改为了false（但分支不全，可能存在精度问题）
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYER,
    no_bias=True,
    mlp_ratio=MLP_RATIO,
    apply_post_layer_norm=False,
    # dtype="torch.bfloat16",       华为使用float16
    dtype="torch.float16",
    norm_type="rmsnorm",
    layer_norm_epsilon=1e-5,
    use_flash_attn=False,           # 华为目前为false
    num_kv_attention_heads=NUM_KV_ATTENTION_HEAD,
    adapt_hf=True,
)
"""
zero1 parallel (dict):
    1. size: int
        * if size <= 0, the size of the zero process group is equal to the size of the dp process group,
            so parameters will be divided within the range of dp.
        * if size == 1, zero is not used, and all dp groups retain the full amount of model parameters.
        * if size > 1 and size <= dp world size, the world size of zero is a subset of dp world size.
        For smaller models, it is usually a better choice to split the parameters within nodes with a setting <= 8.
    2. fsdp: bool, enable/disable torch's fully sharded data parallel, defaults to False.
tensor parallel (dict):
    1. size: int, the size of tensor parallel.
    2. mode: str, the tensor parallel mode, should be in ['mtp', 'msp', 'fsp', 'isp'],
        defaults to 'mtp', means the pure megatron tensor parallel without sequence parallel.
        msp: megatron tensor parallel with sequence parallel, sequence parallel size = tensor parallel size.
        fsp: tensor parallel by flash-attn with sequence parallel, sequence parallel size = tensor parallel size.
        isp: customed intern sequence parallel without tensor parallel, can be used with weight parallel.
pipeline parallel (dict):
    1. size: int, the size of pipeline parallel.
    2. interleaved_overlap: bool, enable/disable communication overlap when using interleaved pipeline scheduler,
        defaults to False.
weight parallel (dict):
    1. size: int, the size of weight parallel.
    2. overlap: bool, enable/disable all_gather/reduce_scatter communication overlap, defaults to False.
    3. memory_pool: bool, enable/disable memory pool, defaults to False.
"""
parallel = dict(
    zero1=dict(size=8, fsdp=False),
    tensor=dict(size=1, mode="mtp"),
    pipeline=dict(size=1, interleaved_overlap=False),
    # pipeline=dict(size=1, interleaved_overlap=True),
    # weight=dict(size=1, overlap=False, memory_pool=False), 新版本功能，目前没有用
)
