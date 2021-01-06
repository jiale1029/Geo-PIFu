#!/bin/bash

DATASET_DIR="/mnt/tanjiale/geopifu_dataset/humanRender_no_config"
MESH_DIR="/mnt/tanjiale/geopifu_dataset/deephuman_dataset"
RESULT_DIR="/mnt/tanjiale/geopifu_dataset"

NAME="GeoPIFu_coarse_1024_minidataset"
NAME_QUERY="GeoPIFu_query_1024_minidataset"
NAME_COLOR="GeoPIFu_color_1024_minidataset"
PIFU_NAME="PIFu_baseline"
BATCH_SIZE=4

# python -m apps.train_shape_coarse --gpu_ids 0,1 --name ${NAME} --meshDirSearch ${MESH_DIR} --datasetDir ${DATASET_DIR} --random_multiview --num_views 1 --batch_size ${BATCH_SIZE} --num_epoch 30 --schedule 8 23 --freq_plot 1 --freq_save 100 --freq_save_ply 100 --num_threads 8 --num_sample_inout 0 --num_sample_color 0 --load_single_view_meshVoxels --vrn_occupancy_loss_type ce --weight_occu 1000.0 --mini_dataset --num_threads 8 && \
# python ../multiprocess.py -pv --name ${NAME}
# python -m apps.train_query --gpu_id 1 --name ${NAME_QUERY} --sigma 3.5 --meshDirSearch ${MESH_DIR} --datasetDir ${DATASET_DIR} --deepVoxelsDir ${RESULT_DIR}/geopifu_results/${NAME}/train --random_multiview --num_views 1 --batch_size ${BATCH_SIZE} --num_epoch 45 --schedule 8 23 40 --num_sample_inout 5000 --num_sample_color 0 --sampleType occu_sigma3.5_pts5k --freq_plot 1 --freq_save 100 --freq_save_ply 100 --z_size 200. --num_threads 8 --deepVoxels_fusion early --deepVoxels_c_len 56 --multiRanges_deepVoxels --mini_dataset
# python -m apps.train_color --gpu_id 1 --name ${NAME_COLOR} --sigma 0.005 --meshDirSearch ${MESH_DIR} --datasetDir ${DATASET_DIR} --batch_size ${BATCH_SIZE} --num_epoch 45 --schedule 8 23 40 --num_sample_inout 0 --num_sample_color 8000 --freq_plot 1 --freq_save 888 --freq_save_ply 888 --num_threads 8 --mini_dataset --load_netG_checkpoint_path ./checkpoints/${NAME_QUERY}/netG_epoch_44_191 --deepVoxels_fusion early --deepVoxels_c_len 56 --multiRanges_deepVoxels --random_multiview --deepVoxelsDir ${RESULT_DIR}/geopifu_results/${NAME}/train && \
# python -m apps.test_shape_iccv --datasetDir ${DATASET_DIR} --resultsDir ${RESULT_DIR}/geopifu_results/${NAME_QUERY} --deepVoxelsDir ${RESULT_DIR}/geopifu_results/${NAME}/train --deepVoxels_fusion early --deepVoxels_c_len 56 --multiRanges_deepVoxels --splitNum 1 --splitIdx 0 --gpu_id 1 --load_netG_checkpoint_path ./checkpoints/${NAME_QUERY}/netG_epoch_44_191 --mini_dataset && \
# python -m apps.test_shape_iccv --datasetDir ${DATASET_DIR} --resultsDir ${RESULT_DIR}/geopifu_results/${NAME_COLOR} --deepVoxelsDir ${RESULT_DIR}/geopifu_results/${NAME}/train --deepVoxels_fusion early --deepVoxels_c_len 56 --multiRanges_deepVoxels --splitNum 1 --splitIdx 0 --gpu_id 1 --load_netG_checkpoint_path ./checkpoints/${NAME_QUERY}/netG_epoch_44_191 --mini_dataset --load_netC_checkpoint_path ./checkpoints/${NAME_COLOR}/netC_epoch_44_191
source /home/tanjiale/miniconda3/bin/activate opendrEnv && cd .. && python main_eval_prepare_iccv.py --compute_vn --datasetDir ${DATASET_DIR} --resultsDir ${RESULT_DIR}/geopifu_results/${NAME_COLOR} --splitNum 1 --splitIdx 0 --mini_dataset && \
python main_eval_metrics_iccv.py --datasetDir ${DATASET_DIR} --resultsDir ${RESULT_DIR}/geopifu_results/${NAME_QUERY} --mini_dataset

python -m apps.train_shape_iccv --gpu_id 1 --name PIFu_baseline --sigma 3.5 --meshDirSearch ${MESH_DIR} --datasetDir ${DATASET_DIR} --random_multiview --num_views 1 --batch_size 4 --num_epoch 45 --schedule 8 23 40 --num_sample_inout 5000 --num_sample_color 0 --sampleType occu_sigma3.5_pts5k --freq_plot 1 --freq_save 888 --freq_save_ply 888 --z_size 200. --num_threads 8 --mini_dataset && \
python -m apps.test_shape_iccv --datasetDir ${DATASET_DIR} --resultsDir ${RESULT_DIR}/geopifu_results/${PIFU_NAME} --splitNum 1 --splitIdx 0 --gpu_id 1 --load_netG_checkpoint_path ./checkpoints/${PIFU_NAME}/netG_epoch_44_95 --mini_dataset && \
source /home/tanjiale/miniconda3/bin/activate opendrEnv && cd .. && python main_eval_prepare_iccv.py --compute_vn --datasetDir ${DATASET_DIR} --resultsDir ${RESULT_DIR}/geopifu_results/${PIFU_NAME} --splitNum 1 --splitIdx 0 --mini_dataset && \
python main_eval_metrics_iccv.py --datasetDir ${DATASET_DIR} --resultsDir ${RESULT_DIR}/geopifu_results/${PIFU_NAME} --mini_dataset