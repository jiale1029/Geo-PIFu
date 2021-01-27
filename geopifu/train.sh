#!/bin/bash

# Directory Settings
DATASET_DIR="/mnt/tanjiale/geopifu_dataset/humanRender_no_config"
MESH_DIR="/mnt/tanjiale/geopifu_dataset/deephuman_dataset"
RESULT_DIR="/mnt/tanjiale/geopifu_dataset/geopifu_results"

DATASET_TYPE="all"

# Network Training Settings
# NAME="GeoPIFu_coarse_2048_${DATASET_TYPE}_8"
# NAME_QUERY="GeoPIFu_query_2048_${DATASET_TYPE}_8"
# NAME_COLOR="GeoPIFu_color_2048_${DATASET_TYPE}_8"
NAME="ICCV_GeoPIFu_coarse"
NAME_QUERY="ICCV_GeoPIFu_query"
NAME_COLOR="ICCV_GeoPIFu_color_embedder_correct_10"

BATCH_SIZE=3
COARSE_EPOCH=30
QUERY_EPOCH=45
COLOR_EPOCH=25

SUBTRACTED_COARSE=$((COARSE_EPOCH-1))
SUBTRACTED_QUERY=$((QUERY_EPOCH-1))
SUBTRACTED_COLOR=$((COLOR_EPOCH-1))

netV_checkpoint="./checkpoints/${NAME}/netV_epoch_${SUBTRACTED_COARSE}_2899"
netG_checkpoint="./checkpoints/${NAME_QUERY}/netG_epoch_${SUBTRACTED_QUERY}_2415"
netC_checkpoint="./checkpoints/${NAME_COLOR}/netC_epoch_${SUBTRACTED_COLOR}_14495"
netC_checkpoint="./checkpoints/${NAME_COLOR}/netC_epoch_8_14495"

GPU_ID=1

# 8 scripts to execute in total
# 1. Train the coarse shape
# 2. Generate the coarse prediction using the trained model
# 3. Train the query model (2D Pixel Aligned + 3D Aligned Latent Voxel)
# 4. Train the color model which is based on the query model
# 5. Generate the shape using the query model.
# 6. Generate the shape using the color model. ( With predicted color )
# 7. Prepare for metrics evaluation (change result_dir to the result you want to evaluate on)
# 8. Output the average metrics

      # --gpu_id ${GPU_ID} \
      # --gpu_ids 0,1 \
# python -m apps.train_shape_coarse \
#       --gpu_id 1 \
#       --name ${NAME} \
#       --meshDirSearch ${MESH_DIR} \
#       --datasetDir ${DATASET_DIR} \
#       --random_multiview \
#       --num_views 1 \
#       --batch_size ${BATCH_SIZE} \
#       --num_epoch ${COARSE_EPOCH} \
#       --schedule 8 23 \
#       --freq_plot 1 \
#       --freq_save 100 \
#       --freq_save_ply 100 \
#       --num_threads 8 \
#       --num_sample_inout 0 \
#       --num_sample_color 0 \
#       --load_single_view_meshVoxels \
#       --vrn_occupancy_loss_type ce \
#       --continue_train \
#       --checkpoints_path ./checkpoints \
#       --resume_name ${NAME} \
#       --resume_epoch 1 \
#       --resume_iter 200 \
#       --load_from_multi_GPU_shape \
#       --weight_occu 1000.0 \
#       --num_threads 8 \
#       --datasetType ${DATASET_TYPE}
# python ../multiprocess.py \
#       --prepare_voxel \
#       --gpu_id ${GPU_ID} \
#       --resultsDir ${RESULT_DIR} \
#       --datasetDir ${DATASET_DIR} \
#       --meshDirSearch ${MESH_DIR} \
#       --netV_checkpoint ${netV_checkpoint} \
#       --name ${NAME} \
#       --dataType "train" \
#       --datasetType ${DATASET_TYPE}
# python -m apps.train_query \
#       --gpu_id 1 \
#       --name ${NAME_QUERY} \
#       --sigma 3.5 \
#       --meshDirSearch ${MESH_DIR} \
#       --datasetDir ${DATASET_DIR} \
#       --deepVoxelsDir ${RESULT_DIR}/${NAME}/train \
#       --random_multiview \
#       --num_views 1 \
#       --batch_size 4 \
#       --num_epoch ${QUERY_EPOCH} \
#       --schedule 8 23 40 \
#       --num_sample_inout 5000 \
#       --num_sample_color 0 \
#       --sampleType occu_sigma3.5_pts5k \
#       --freq_plot 1 \
#       --freq_save 100 \
#       --freq_save_ply 100 \
#       --z_size 200. \
#       --num_threads 8 \
#       --deepVoxels_fusion early \
#       --deepVoxels_c_len 56 \
#       --multiRanges_deepVoxels \
#       --datasetType ${DATASET_TYPE}
python -m apps.train_color \
      --gpu_id ${GPU_ID} \
      --name ${NAME_COLOR} \
      --sigma 0.005 \
      --meshDirSearch ${MESH_DIR} \
      --datasetDir ${DATASET_DIR} \
      --batch_size 6 \
      --learning_rate 1e-4 \
      --num_epoch ${COLOR_EPOCH} \
      --schedule 8 23 40 \
      --num_sample_inout 0 \
      --num_sample_color 8000 \
      --freq_plot 1 \
      --freq_save 888 \
      --freq_save_ply 888 \
      --num_threads 8 \
      --load_netG_checkpoint_path ${netG_checkpoint} \
      --load_from_multi_GPU_shape \
      --deepVoxels_fusion early \
      --deepVoxels_c_len 56 \
      --multiRanges_deepVoxels \
      --random_multiview \
      --num_views 1 \
      --use_embedder \
      --deepVoxelsDir ${RESULT_DIR}/${NAME}/train \
      --datasetType ${DATASET_TYPE}
# python -m apps.test_shape_iccv \
#       --datasetDir ${DATASET_DIR} \
#       --resultsDir ${RESULT_DIR}/${NAME_QUERY} \
#       --deepVoxelsDir ${RESULT_DIR}/${NAME}/train \
#       --deepVoxels_fusion early \
#       --deepVoxels_c_len 56 \
#       --multiRanges_deepVoxels \
#       --splitNum 1 \
#       --splitIdx 0 \
#       --gpu_id ${GPU_ID} \
#       --load_from_multi_GPU_shape \
#       --load_netG_checkpoint_path ${netG_checkpoint} \
#       --datasetType "mini" && \
# python -m apps.test_shape_iccv \
#       --datasetDir ${DATASET_DIR} \
#       --resultsDir ${RESULT_DIR}/${NAME_COLOR} \
#       --deepVoxelsDir ${RESULT_DIR}/${NAME}/train \
#       --deepVoxels_fusion early \
#       --deepVoxels_c_len 56 \
#       --multiRanges_deepVoxels \
#       --splitNum 1 \
#       --splitIdx 0 \
#       --gpu_id ${GPU_ID} \
#       --load_netG_checkpoint_path ${netG_checkpoint} \
#       --load_netC_checkpoint_path ${netC_checkpoint} \
#       --load_from_multi_GPU_shape \
#       --datasetType ${DATASET_TYPE}
# source /home/tanjiale/miniconda3/bin/activate opendrEnv && \
#   cd .. && \
#   python main_eval_prepare_iccv.py \
#       --compute_vn \
#       --datasetDir ${DATASET_DIR} \
#       --resultsDir ${RESULT_DIR}/${NAME_COLOR} \
#       --splitNum 12 \
#       --splitIdx 0 \
#       --gpu_id ${GPU_ID} \
#       --datasetType ${DATASET_TYPE}
#   python main_eval_metrics_iccv.py \
#       --datasetDir ${DATASET_DIR} \
#       --resultsDir ${RESULT_DIR}/${NAME_COLOR} \
#       --datasetType ${DATASET_TYPE}

      # --load_from_multi_GPU_shape \
