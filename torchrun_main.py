import os
import time
import json
import random
import argparse
import numpy as np
import warnings

from modules import SmallLinear

warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed
import wandb

from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM
from peft_pretraining.grad_clipping import clip_grad_norm_

from optimizers import SmallAdamW
from random_utils import Manager, Projector
from memory import MemoryTracker, track

transformers.logging.set_verbosity_error()


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts",
                                                                            "exponential"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000, help="Number of update steps")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--track_memory", default=False, action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--group", type=str, default="no_group")
    parser.add_argument("--grad_clipping", type=float, default=0.0)

    # low rank arguments
    parser.add_argument("--rank", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--update_proj_gap", type=int, default=None)
    parser.add_argument("--proj_type", type=str, default="gaussian")
    parser.add_argument("--memory_efficient", default=False, action="store_true")
    parser.add_argument("--full_matrices", default=False, action="store_true")
    parser.add_argument("--scale_o_proj", type=float, default=1.0)
    parser.add_argument("--layer_filter", type=str, default="all")
    parser.add_argument("--depth_filter", type=str, default="0-100")

    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")

    args = parser.parse_args(args)
    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    _time = time.time()
    val_data = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True)  # DGX
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size
    ppl = torch.exp(torch.tensor(total_loss))

    return total_loss, evaluated_on_tokens, ppl


def get_depth_from_name(full_name):
    # returns the first number in the layer name assuming they are split by '.'
    name_parts = full_name.split('.')
    depth = None
    for part in name_parts:
        if part.isdigit():
            depth = int(part)
            break
    return depth


def parse_llama_layer_filter(layer_str):
    # also supports "mlp","attn", "all" (same as using "mlp+attn") CompAct is never applied to 'x'.
    convert_dict = {"all": "qkvugdx", "mlp": "ugd", "attn": "qkvx"}
    for c, alt in convert_dict.items():
        layer_str = layer_str.replace(c, alt)
    layer_map = {'q': 'q_proj', 'k': 'k_proj', 'v': 'v_proj', 'x': 'o_proj',
                 'g': 'gate_proj', 'u': 'up_proj', 'd': 'down_proj'}
    return list(set([layer_map[char] for char in layer_str if char in layer_map]))


def parse_roberta_layer_filter(layer_str):
    # also supports "mlp","attn", "all" (same as using "mlp+attn"). CompAct is never applied to 'x'.
    convert_dict = {"all": "qkvxio", "mlp": "io", "attn": "qkvx"}
    for c, alt in convert_dict.items():
        layer_str = layer_str.replace(c, alt)
    layer_map = {'q': 'query', 'k': 'key', 'v': 'value', 'i': 'intermediate.dense',
                 'o': 'output.dense', 'x': 'attention.output.dense'}
    return list(set([layer_map[char] for char in layer_str if char in layer_map]))


def parse_depth_string(depth_str):
    # Parses a string like '0-3,5-6' into a list of integers: [0, 1, 2, 3, 5, 6].
    depth_list = []
    parts = depth_str.split(',')
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            depth_list.extend(range(start, end + 1))  # include the end value
        else:
            depth_list.append(int(part))
    return depth_list


def find_parent_module(model, target_module):
    """
    Recursively find and return the parent of a given module within the model.
    """
    for name, child in model.named_children():
        if child is target_module:
            return model  # Return the parent module
        else:
            parent = find_parent_module(child, target_module)
            if parent is not None:
                return parent
    return None  # No parent found if target_module is not part of model


def _replace_linear_with_custom(full_model, args, module, small_params,
                                name_prefix, layer_name_filters, depth_filters):
    """
    Recursively find nn.Linears and replace with SmallLinear according to filters
    """

    for name, child in module.named_children():
        full_name = f"{name_prefix}.{name}" if name_prefix else name  # full name of current module

        is_llama_o_proj = 'o_proj' in full_name
        is_roberta_o_proj = 'attention.output.dense' in full_name

        if isinstance(child, nn.Linear):
            depth = get_depth_from_name(full_name)

            # If the name contains any of the filters and the depth is in the allowed list, replace the layer
            if any(layer_name in full_name for layer_name in layer_name_filters) and (
                    depth is not None and depth in depth_filters):
                small_like_child = SmallLinear.like(child, single_gpu=args.single_gpu)
                setattr(module, name, small_like_child)
                small_params.append(small_like_child.weight)

                if is_llama_o_proj:  # identity, no projection, just scaling by (scale_o_proj) * args.scale
                    print('scaling grads for weights in module: ', full_name)
                    small_like_child.weight.projector = Projector(rank=1, proj_type='id',
                                                                  scale=args.scale_o_proj * args.scale,
                                                                  gap=args.update_proj_gap,
                                                                  full_matrices=args.full_matrices)
                elif is_roberta_o_proj:  # identity, no projection, no scaling
                    print('regular grads for weights in module: ', full_name)
                    small_like_child.weight.projector = Projector(rank=1, proj_type='id',
                                                                  scale=1, gap=args.update_proj_gap,
                                                                  full_matrices=args.full_matrices)
                else:  # CompAct projector
                    print('enabling small grads for weights in module: ', full_name)
                    small_like_child.weight.projector = Projector(rank=args.rank, proj_type=args.proj_type,
                                                                  scale=args.scale, gap=args.update_proj_gap,
                                                                  full_matrices=args.full_matrices,
                                                                  single_gpu=args.single_gpu)
        # Recursively process the submodules
        _replace_linear_with_custom(full_model, args, child, small_params, full_name, layer_name_filters, depth_filters)


def replace_parameters(model, args, is_roberta=False):
    small_params = []
    """
    This function replaces nn.Linear layers with SmallLinear layers (CompAct), currently supporting LLaMA and RoBERTa. 
    For a custom model simply define your own parser, defining which nn.Linears to compress the activations of.

    is_roberta a boolean flag whether the model is LLaMA or RoBERTa.

    - args.layer_name_filter is the layer types to apply CompAct to;
       it is a string of letters, some subset of "qkvogud" representing query,key,value,output,gate,up,down.
       in roberta, o refers to the output of the mlp, not the attention
       this is because as mentioned in our paper we currently never compress the output of attention 
       as flashattn saves its full rank tensor anyway.
    - args.depth_filter is a string of comma separated int of the depths to apply CompAct to. unused. (all is default). 
    """
    if is_roberta:
        parsed_layer_filter = parse_roberta_layer_filter(args.layer_filter)
    else:
        parsed_layer_filter = parse_llama_layer_filter(args.layer_filter)

    _replace_linear_with_custom(model, args, model, small_params, "", parsed_layer_filter,
                                parse_depth_string(args.depth_filter))

    print(len(small_params), 'len(small_params)')
    id_small_params = [id(p) for p in small_params]
    print(len(id_small_params), 'len(id_small_params)')
    regular_params = [p for p in model.parameters() if id(p) not in id_small_params]
    param_groups = [{'params': regular_params},
                    {'params': small_params}]
    return small_params, param_groups


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    sketch_manager = Manager()
    sketch_manager.init_iter()

    tracker = None
    if args.track_memory:
        if global_rank == 0:
            tracker = MemoryTracker("MB")

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()

    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        wandb.init(entity="gnn-depth", project="memory-eff-llms", name=args.name, group=args.group)

    data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

    seed_for_shuffle = 42

    # logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)
    pad_idx = tokenizer.pad_token_id

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    model_config = AutoConfig.from_pretrained(args.model_config)

    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)
    track(tracker, 0, "model_init")
    # print("post-init")
    # print_tensors()

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(
                f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'allenai/c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now")  # save current script

    if args.memory_efficient:
        small_params, param_groups = replace_parameters(model, args)

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")
    if args.memory_efficient:
        logger.info(f"Total params with SmallLinear enabled: {sum(p.numel() for p in small_params) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    if args.optimizer.lower() == "adam":
        if args.memory_efficient:
            optimizer = SmallAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    scheduler = training_utils.get_scheduler(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        num_training_steps=args.num_training_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading optimizer from {args.continue_from}")
        opt_chkpnt = os.path.join(args.continue_from, "optimizer.pt")
        opt_chkpnt = torch.load(opt_chkpnt)
        optimizer.load_state_dict(opt_chkpnt["optimizer"])
        scheduler.load_state_dict(opt_chkpnt["scheduler"])
        logger.info("*" * 40)

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                                            output_device=local_rank,
                                                                            broadcast_buffers=False, )

    # global steps and others are defined above
    # pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step
    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################
    track(tracker, global_step, "init-params")

    for batch_idx, batch in enumerate(dataloader):
        global_step += 1
        local_step += 1

        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        track(tracker, global_step, f"pre-forward")

        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / args.gradient_accumulation
        track(tracker, global_step, f"post-forward")

        scaled_loss.backward()
        track(tracker, global_step, f"post-backward")

        if global_step % args.gradient_accumulation != 0:
            continue

        # The below code is only executed during the update step
        # add grad clipping
        # if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
        if args.grad_clipping != 0.0: clip_grad_norm_(trainable_params, args.grad_clipping)

        if args.memory_efficient:
            optimizer_metrics = optimizer.step(iter=global_step, global_rank=global_rank)
        else:
            optimizer_metrics = optimizer.step()
        track(tracker, global_step, f"post-optimizer")

        scheduler.step()
        optimizer.zero_grad()
        track(tracker, global_step, f"post-zero_grads")
        if args.track_memory:
            wandb.finish()
            exit()

        update_step += 1
        update_time = time.time() - update_time

        # track(tracker, global_step, "post-update")

        sketch_manager.update_iter()

        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(current_model_directory, exist_ok=True)
            if args.single_gpu:
                model.save_pretrained(current_model_directory, max_shard_size='100GB', safe_serialization=False)
            else:
                model.module.save_pretrained(current_model_directory, max_shard_size='100GB', safe_serialization=False)

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

            # save wandb related info
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=4)

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens, ppl = evaluate_model(
                model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
            )
            if global_rank == 0:
                wandb.log({
                    "final_eval_loss": total_loss,
                    "final_eval_tokens": evaluated_on_tokens,
                    "perplexity": ppl
                },
                    step=global_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")

        lr = optimizer.param_groups[0]["lr"]

        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            output_dict = {
                "loss": loss.item(),
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
            }
            if optimizer_metrics is not None:
                output_dict.update(optimizer_metrics)
            wandb.log(output_dict, step=global_step)
        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(current_model_directory, exist_ok=True)
        if args.single_gpu:
            model.save_pretrained(current_model_directory, safe_serialization=False)
        else:
            model.module.save_pretrained(current_model_directory, safe_serialization=False)

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    total_loss, evaluated_on_tokens, ppl = evaluate_model(
        model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
    )

    if global_rank == 0:
        wandb.log({
            "final_eval_loss": total_loss,
            "final_eval_tokens": evaluated_on_tokens,
            "perplexity": ppl
        },
            step=global_step,
        )
        logger.info(f"Final eval loss: {total_loss}")
    wandb.finish()

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    # print("Starting script")
    args = parse_args(None)
    seeds = list(range(args.n_seeds))
    for seed in seeds:
        args.seed = seed
        main(args)
