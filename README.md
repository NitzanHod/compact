# CompAct: Compressed Activations for Memory-Efficient LLM Training


### Install experiment dependencies

```bash
pip install -e .
pip install -r requirements.txt
```

## Benchmark 1: Pre-Training LLaMA on C4 dataset

```bash
# LLaMA-60M
torchrun torchrun_main.py
    --model_config configs/llama_60m.json\
    --lr 1e-2 \  
    --group llama60 \  
    --batch_size 256 \  
    --total_batch_size 512 \  
    --num_training_steps 10000 \  
    --warmup_steps 1000 \  
    --weight_decay 0.0 \  
    --dtype bfloat16 \  
    --eval_every 1000 \  
    --optimizer adam \  
    --n_seeds 1 \  
    --memory_efficient \  
    --rank 0.25 \  
    --proj_type gaussian \  
    --update_proj_gap 50 \ 
    --name gap-50-rank-0.25 \  
    --scale 0.25 \  
    --scale_o_proj 0.5 \ 
    --single_gpu 

```

```bash
# LLaMA-1B
torchrun --nprocs_per_node=2 torchrun_main.py \
    --model_config configs/llama_1b.json \
    --lr 3e-3 \
    --group llama1b \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 100000 \
    --warmup_steps 10000 \
    --weight_decay 0.0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer adam \
    --n_seeds 1 \
    --memory_efficient \
    --rank 0.5 \
    --proj_type gaussian \
    --update_proj_gap 200 \
    --name flora_setting_0.5 \
    --scale 0.25
```

Currently per-layer weight updates technique is only supported for single GPU training (`--single_gpu`) without using `nn.parallel.DistributedDataParallel`. We are working on supporting multi-GPU training with per-layer weight updates.

## Benchmark 2: Fine-Tuning RoBERTa on GLUE tasks
```bash
python run_glue.py \
    --model_name_or_path roberta-base \
    --n_seeds 3 \
    --group finetuning_roberta-base \
    --max_length 512 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 30 \
    --report_to wandb \
    --with_tracking \ 
    --checkpointing_steps 10000 \
    --output_dir results/ft/m_roberta_base \
    --proj_type gaussian \
    --memory_efficient \
    --update_proj_gap 500 \
    --rank 8 \
    --scale 2 \
    --name r8s2g500_rte \
    --task_name rte \
    --lr 3e-5
```