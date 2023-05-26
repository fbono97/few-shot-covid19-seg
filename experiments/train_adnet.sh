#!/bin/bash

cd /content/drive/MyDrive/few-shot-covid19-seg


# Args
SEED=123
FOLD=0                # Change this to train on every fold (0 to 4)
RUN_ID="adnet_fold${FOLD}"

CHECKPOINT_EPOCH=""   # If not null, training will pause at indicated epoch

RESUME_FROM_EPOCH=""  # If not null, training will resume from indicated
                      # epoch if available or from latest checkpoint


# Paths
CONFIG_FILE=/content/drive/MyDrive/few-shot-covid19-seg/configs/config_adnet.yaml
RESULTS_PATH=/content/drive/MyDrive/few-shot-covid19-seg/results
LOGS_PATH=${RESULTS_PATH}/logs
CHECKPOINTS_PATH=${RESULTS_PATH}/checkpoints


# Create dirs for storing results (if not already created)
[ ! -d ${RESULTS_PATH} ] && mkdir ${RESULTS_PATH}
[ ! -d ${LOGS_PATH} ] && mkdir ${LOGS_PATH}
[ ! -d ${CHECKPOINTS_PATH} ] && mkdir ${CHECKPOINTS_PATH}


# Run
python3 train.py \
  --seed ${SEED} \
  --fold ${FOLD} \
  --run_id ${RUN_ID} \
  --checkpoint_epoch ${CHECKPOINT_EPOCH} \
  --resume_from_epoch ${RESUME_FROM_EPOCH} \
  --config_file ${CONFIG_FILE} \
  --logs_path ${LOGS_PATH} \
  --checkpoints_path ${CHECKPOINTS_PATH}
 
# If using multiple GPUS/nodes replace "python3 train.py \" with line below
#torchrun --nproc_per_node=${N_GPUS} distributed_train.py \