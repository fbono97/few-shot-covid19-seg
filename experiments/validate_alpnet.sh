#!/bin/bash

cd /content/drive/MyDrive/few-shot-covid19-seg


# Args
FOLD=0                # Change this to validate on every fold (0 to 4)
RUN_ID="alpnet_fold${FOLD}"


# Paths
CONFIG_FILE=/content/drive/MyDrive/few-shot-covid19-seg/configs/config_alpnet.yaml
RESULTS_PATH=/content/drive/MyDrive/few-shot-covid19-seg/results
LOGS_PATH=${RESULTS_PATH}/logs
CHECKPOINTS_PATH=${RESULTS_PATH}/checkpoints
PREDICTIONS_PATH=${RESULTS_PATH}/predictions
IMG_SAVE_PATH=${PREDICTIONS_PATH}/${RUN_ID}


# Create dirs for storing results (if not already created)
[ ! -d ${RESULTS_PATH} ] && mkdir ${RESULTS_PATH}
[ ! -d ${LOGS_PATH} ] && mkdir ${LOGS_PATH}
[ ! -d ${PREDICTIONS_PATH} ] && mkdir ${PREDICTIONS_PATH}
[ ! -d ${IMG_SAVE_PATH} ] && mkdir ${IMG_SAVE_PATH}


# Run
python3 validate.py \
  --fold ${FOLD} \
  --run_id ${RUN_ID} \
  --config_file ${CONFIG_FILE} \
  --logs_path ${LOGS_PATH} \
  --checkpoints_path ${CHECKPOINTS_PATH} \
  --img_save_path ${IMG_SAVE_PATH}
 