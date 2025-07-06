#!/bin/bash

# ======= 配置区域 =======
START=405              # 起始编号（包含）
END=626                # 结束编号（不包含）
GPUS=(0 1 2 3 4 5 6 7)            # 指定可用 GPU（手动指定）
MODEL_NAME="VideoHallu"
DATASET_PATH="/mnt/world_foundational_model/fck/LizhangChen/Video-R1/data/PhysBench"
SPLIT="test"
# ========================

# 创建日志目录
#mkdir -p evaluation

# 总数据数
TOTAL=$((END - START))

# GPU数量
NUM_GPUS=${#GPUS[@]}

# 每张 GPU 应处理的基本步长
BASE_STEP=$((TOTAL / NUM_GPUS))
REMAINDER=$((TOTAL % NUM_GPUS))

CURRENT=$START

# 分配任务
for ((i=0; i<NUM_GPUS; i++))
do
  GPU_ID=${GPUS[$i]}
  
  STEP=$BASE_STEP
  if [ $i -lt $REMAINDER ]; then
    STEP=$((STEP + 1))
  fi

  LOWER=$CURRENT
  UPPER=$((CURRENT + STEP))
  CURRENT=$UPPER

  LOG_FILE="evaluation_qwen/test_${LOWER}-${UPPER}.log"

  echo "Launching range $LOWER-$UPPER on GPU $GPU_ID"
  nohup env CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH='./' \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python eval/test_benchmark.py \
    --model_name $MODEL_NAME \
    --dataset_path $DATASET_PATH \
    --split $SPLIT \
    --lower $LOWER --upper $UPPER \
    > $LOG_FILE 2>&1 &
done
