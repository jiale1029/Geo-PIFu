#!/bin/bash

DATASET_DIR="/mnt/tanjiale/geopifu_dataset/humanRender_no_config"
MESH_DIR="/mnt/tanjiale/geopifu_dataset/deephuman_dataset"

NAME="GeoPIFu_coarse_minidataset"
NAME_QUERY="GeoPIFU_query_minidataset"
BATCH_SIZE=4

python -m apps.train_shape_coarse --gpu_ids 0,1 --name ${NAME} --meshDirSearch ${MESH_DIR} --datasetDir ${DATASET_DIR} --random_multiview --num_views 1 --batch_size ${BATCH_SIZE} --num_epoch 30 --schedule 8 23 --freq_plot 1 --freq_save 100 --freq_save_ply 100 --num_threads 8 --num_sample_inout 0 --num_sample_color 0 --load_single_view_meshVoxels --vrn_occupancy_loss_type ce --weight_occu 1000.0 --mini_dataset && \
python ../multiprocess.py -pv --name ${NAME} && \
python -m apps.train_query --gpu_ids 0,1 --name ${NAME_QUERY} --sigma 3.5 --meshDirSearch ${MESH_DIR} --datasetDir ${DATASET_DIR} --deepVoxelsDir ${DATASET_DIR}/geogeopifuResults/${NAME}/train --random_multiview --num_views 1 --batch_size ${BATCH_SIZE} --num_epoch 45 --schedule 8 23 40 --num_sample_inout 5000 --num_sample_color 0 --sampleType occu_sigma3.5_pts5k --freq_plot 1 --freq_save 100 --freq_save_ply 100 --z_size 200. --num_threads 8 --deepVoxels_fusion early --deepVoxels_c_len 56 --multiRanges_deepVoxels --minidataset
