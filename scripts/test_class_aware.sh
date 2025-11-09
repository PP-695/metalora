#!/bin/bash
# Test script for Class-Aware Meta-Learning on CIFAR100-LT

# Test on CIFAR100 with IR=100 (long-tailed)
python main.py \
  --dataset CIFAR100_IR100 \
  --model clip_vit_b16 \
  --tuner class_aware_lora \
  --opts \
    use_meta=True \
    use_class_aware=True \
    meta_lr=0.001 \
    meta_objective=balanced_accuracy \
    meta_data_ratio=0.2 \
    meta_update_freq=1 \
    meta_inner_steps=3 \
    focus_on_tail=True \
    tail_loss_weight=2.0 \
    rank_divergence_penalty=0.01 \
    alpha_smoothness_penalty=0.005 \
    num_epochs=50 \
    batch_size=128 \
    lr=0.01 \
    seed=42
