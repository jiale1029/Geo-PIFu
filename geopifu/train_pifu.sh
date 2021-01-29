#!/bin/bash

DATASET_DIR="/mnt/tanjiale/geopifu_dataset/humanRender_no_config"
MESH_DIR="/mnt/tanjiale/geopifu_dataset/deephuman_dataset"
RESULT_DIR="/mnt/tanjiale/geopifu_dataset/geopifu_results"

EPOCH=45

PIFU_NAME="PIFu_baseline_adjusted_8"
BATCH_SIZE=4
DATASET_TYPE="adjusted"

netG_checkpoint="netG_epoch_44_191"

python -m apps.train_shape_iccv \
      --gpu_id 1 \
      --name ${PIFU_NAME} \
      --sigma 3.5 \
      --meshDirSearch ${MESH_DIR} \
      --datasetDir ${DATASET_DIR} \
      --random_multiview \
      --num_views 1 \
      --batch_size ${BATCH_SIZE} \
      --num_epoch ${EPOCH} \
      --schedule 8 23 40 \
      --num_sample_inout 5000 \
      --num_sample_color 0 \
      --sampleType occu_sigma3.5_pts5k \
      --freq_plot 1 \
      --freq_save 888 \
      --freq_save_ply 888 \
      --z_size 200. \
      --num_threads 8 \
      --datasetType ${DATASET_TYPE} && \
python -m apps.test_shape_iccv \
      --datasetDir ${DATASET_DIR} \
      --resultsDir ${RESULT_DIR}/${PIFU_NAME} \
      --splitNum 1 \
      --splitIdx 0 \
      --gpu_id 1 \
      --load_netG_checkpoint_path ./checkpoints/${PIFU_NAME}/${netG_checkpoint} \
      --datasetType ${DATASET_TYPE} && \
source /home/tanjiale/miniconda3/bin/activate opendrEnv && \
  cd .. && \
  python main_eval_prepare_iccv.py \
        --compute_vn \
        --datasetDir ${DATASET_DIR} \
        --resultsDir ${RESULT_DIR}/${PIFU_NAME} \
        --splitNum 1 \
        --splitIdx 0 \
        --datasetType ${DATASET_TYPE} && \
  python main_eval_metrics_iccv.py \
        --datasetDir ${DATASET_DIR} \
        --resultsDir ${RESULT_DIR}/${PIFU_NAME} \
        --datasetType ${DATASET_TYPE}
