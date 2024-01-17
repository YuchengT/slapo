 # Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.utils import RepeatingLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import slapo
from slapo import set_random_seed
from slapo.logger import get_logger
from slapo.utils.report import report_memory

from slapo.model_schedule import apply_schedule
from examples.utils import (
    get_ds_inference_config,
    create_dist_group_for_pipeline,
    generate_pipeline_cuts,
)

SINGLE_DEVICE_FOR_DEBUG = False

logger = get_logger()


def reconfig_model(args, model_config):
    if args.hidden_size > 0:
        model_config.hidden_size = args.hidden_size
        model_config.num_hidden_layers = args.nlayers
        model_config.num_attention_heads = args.num_attn_heads

    model_config.attn_pdrop = args.dropout
    model_config.resid_pdrop = args.dropout
    model_config.embd_pdrop = args.dropout

    model_config.activation_function = args.activation_function
    model_config.max_position_embeddings = args.seq_len

    return model_config


def inference(args):
    #batch_size = args.batch_size
    #micro_batch_size = args.micro_batch_size

    num_pp, num_mp = 1, 1
    rank = args.local_rank
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    # Configurations.
    enable_pipeline = not SINGLE_DEVICE_FOR_DEBUG and not args.disable_pipeline
    #if args.disable_schedule and args.checkpoint not in [0.0, 1.0]:
    #    raise ValueError("checkpoint must be 0.0 or 1.0 with disable_schedule")
    #use_default_ckpt = args.checkpoint == 1.0 and args.disable_schedule

    topology, group = None, None
    if not SINGLE_DEVICE_FOR_DEBUG:
        deepspeed.init_distributed(dist_backend="nccl")
        logger.info("Use deepspeed to initialize", ranks=0)
        if enable_pipeline:
            # num_pp, num_mp = 4, 2 # For single node testing.
            num_pp = args.pmp
            num_mp = args.tmp
        else:
            logger.info("Pipeline disabled", ranks=0)
            num_pp = 1
            num_mp = args.tmp

        topology, group = create_dist_group_for_pipeline(num_pp, num_mp)

        # FIXME: Pytorch _coalescing_manager requires all the ranks to join
        # if that is the first collective call in the given group.
        # We use the following broadcast as the first call for workaround,
        # and it will be removed once we implement the features to synchonrize
        # the model parameters during initialization.
        x = torch.tensor(0, device=torch.cuda.current_device())
        dist.broadcast(x, src=0)

    logger.info(f"TMP {num_mp}, PMP {num_pp}", ranks=[0])
    # https://huggingface.co/EleutherAI/gpt-neo-2.7B/blob/main/config.json
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    # FIXME: This model has vocab size 5025 7 that cannot be sharded by 2,
    # so we pad it to 50258 in this example. In practice, the tokenizer
    # should be used to pad the vocab size to a multiple of 2.
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - config.vocab_size % 8
    config.use_cache = False
    config.gradient_checkpointing = False
    config = reconfig_model(args, config)
    logger.info(config, ranks=[0])

    report_memory(msg="Before creating model")
    with slapo.init_empty_weights(enable=enable_pipeline):
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    report_memory(msg="After creating model")

    # Evenly partition layers for pipelining.
    if enable_pipeline:
        pipeline_cuts = generate_pipeline_cuts(config.num_hidden_layers, num_pp)
    elif SINGLE_DEVICE_FOR_DEBUG:
        pipeline_cuts = generate_pipeline_cuts(config.num_hidden_layers, 4)
    else:
        pipeline_cuts = []
    logger.info(f"Pipeline cuts: {pipeline_cuts}", ranks=0)

    if args.disable_schedule:
        assert not enable_pipeline
        sch = slapo.create_schedule(model, group=group)
    else:
        sch = apply_schedule(
            model,
            "falcon40b",
            model_config=config,
            prefix="transformer",
            attn_op_name=args.attn_op_name,
            ckpt_ratio=args.checkpoint,
            bcast_input=True,
            fp16=args.fp16,
            bf16=args.bf16,
            group=group,
            pipeline_cuts=pipeline_cuts,
            delay_init=enable_pipeline,
            sequence_parallel=args.sequence_parallel,
            checkpoint_method=args.checkpoint_method,
        )
    tp_rank = sch.rank
    print(sch.world_size)
    print(sch.rank)

    # After scheduling, we check again whether the pipeline is really enabled.
    # If users specified to enable pipeline but the number of pipeline stage is 1,
    # then we set enable_pipeline=False for the rest process to propertly setup
    # DeepSpeed config and runtime engine.
    enable_pipeline = enable_pipeline and pipeline_cuts

    if enable_pipeline:
        ds_config_dict = get_ds_inference_config(
            use_triton=args.use_triton,
            dtype=torch.float16 if args.fp16 else torch.float,
        )
        model = slapo.build(
            sch,
            topology=topology,
            config=ds_config_dict,
            target="deepspeed",
        )
    else:
        ds_config_dict = get_ds_inference_config(
            use_triton=args.use_triton,
            dtype=torch.float16 if args.fp16 else torch.float,
        )
        model = slapo.build(
            sch,
            topology=topology,
            config=ds_config_dict,
            target="deepspeed",
        )
        model = model.to(device)
    report_memory(msg="After building model")

    pp_rank = None if args.disable_pipeline else model.mpu.get_pipe_parallel_rank()
    set_random_seed(
        2013,
        model.mpu.get_data_parallel_rank(),
        pp_rank,
        tp_rank,
        always_enable_tp_seed=args.sequence_parallel,
    )

    input_prompt = [
    "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:"
    ]
    input_tokens = tokenizer.batch_encode_plus(input_prompt, return_tensors="pt",)
    token_num = input_tokens['input_ids'].size(-1)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(get_accelerator().current_device_name())
    input_tokens.pop('token_type_ids')
    sequences = model.generate(**input_tokens, min_length=200, max_length=300, do_sample=True)

    if torch.distributed.get_rank() == 0:
        print(f"Result: {tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]}")


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    parser = argparse.ArgumentParser()
    # This is passed in via cmd
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=n_gpus)
    parser.add_argument(
        "--model_name",
        type=str,
        default="tiiuae/falcon-40b",
        help="Model name",
    )
    parser.add_argument(
        "--checkpoint",
        type=float,
        default=0.0,
        help="Activation checkpointing ratio. 1.0 means all",
    )
    parser.add_argument(
        "--checkpoint_method",
        type=str,
        default="head",
        help="Activation checkpointing method {'head', 'uniform'}",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=1024,
        help="Sequence length",
    )
    parser.add_argument(
        "--activation_function",
        type=str,
        default="gelu_new",
        help="Activation function",
    )
    parser.add_argument(
        "--attn_op_name",
        type=str,
        default="cuda",
        help="Attention op name {'native_xformers', 'cutlass', 'triton', 'cuda'}. "
        "'cuda' and 'triton' only support sm_80+, and other archs will "
        "fallback to 'cutlas'",
    )
    parser.add_argument(
        "--disable_pipeline",
        action="store_true",
        help="Disable pipeline and only use ZeRO-3",
    )
    parser.add_argument(
        "--disable_schedule",
        action="store_true",
        help="Disable Slapo schedule (only applicable with --disable-pipeline)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=-1,
        help="Config hidden size of the model, if it is negative value,"
        " it uses default value associated with the model name",
    )
    parser.add_argument(
        "--nlayers", type=int, default=-1, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num-attn-heads", type=int, default=-1, help="Number of attention heads"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )
    parser.add_argument(
        "--pmp", type=int, default=2, help="Pipeline model parallel size"
    )
    parser.add_argument("--tmp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument(
        "--sequence_parallel",
        action="store_true",
        help="Sequence parallelism is enabled",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="fp16 is enabled. fp16 is enabled by default",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="bf16 is enabled",
    )
    parser.add_argument(
        "--use_triton",
        type=bool,
        default=False,
        help="whether to use triton kernels for inference ops",
    )
    args = parser.parse_args()
    if os.environ.get("LOCAL_RANK"):
        args.local_rank = int(os.environ["LOCAL_RANK"])

    if args.fp16 and args.bf16:
        raise ValueError(
            f"fp16={args.fp16} and bf16={args.bf16} cannot be enabled at the same time"
        )
    elif not args.fp16 and not args.bf16:
        args.fp16 = True
        logger.info("fp16 is enabled by default", ranks=0)

    if args.hidden_size > 0:
        assert args.nlayers > 0, "must have nlayers > 0"
        assert args.num_attn_heads > 0, "must have num_attn_heads > 0"

    # The main entry point is called directly without using subprocess
    inference(args)
