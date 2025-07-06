#!/bin/bash

# 设置参数
TOTAL=2501                # 总数据条数
GPUS=(0 1 2 3 4 5 6 7)             # 手动指定的GPU号列表

MODEL_NAME="VideoHallu"
DATASET_PATH="/mnt/world_foundational_model/fck/LizhangChen/Video-R1/data/PhysBench"
SPLIT="test"

# 创建日志目录
mkdir -p evaluation_physbench

# 计算总任务数和每个GPU的分配数量
num_gpus=${#GPUS[@]}
base_step=$((TOTAL / num_gpus))  # 整数部分
remainder=$((TOTAL % num_gpus))  # 余数，用于前几个GPU分配多一个

start=0

for ((i=0; i<num_gpus; i++))
do
  gpu_id=${GPUS[$i]}

  # 每个GPU的 upper = base_step，前 remainder 个GPU多分一个
  step=$base_step
  if [ $i -lt $remainder ]; then
    step=$((step + 1))
  fi

  lower=$start
  upper=$((start + step))
  start=$upper

  LOG_FILE="evaluation_physbench/test_${lower}-${upper}.log"

  echo "Assigning $lower to $upper to GPU $gpu_id"
  nohup env CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH='./' \
    python eval/test_benchmark.py \
    --model_name $MODEL_NAME \
    --dataset_path $DATASET_PATH \
    --split $SPLIT \
    --lower $lower --upper $upper \
    > $LOG_FILE 2>&1 &
done
