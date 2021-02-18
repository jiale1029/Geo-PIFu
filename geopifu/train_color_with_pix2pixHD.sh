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
NAME_COLOR="ICCV_GeoPIFu_color_with_embedder_pix2pix_global"

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
netC_checkpoint="./checkpoints/${NAME_COLOR}/netC_epoch_23_14495"

GPU_ID=0
GPU_IDS='1'

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
export CUDA_VISIBLE_DEVICES=1 && python -m apps.train_color \
      --gpu_id ${GPU_ID} \
      --name ${NAME_COLOR} \
      --sigma 0.005 \
      --meshDirSearch ${MESH_DIR} \
      --datasetDir ${DATASET_DIR} \
      --batch_size 2 \
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
      --embedder_input_dim 3 \
      --deepVoxelsDir ${RESULT_DIR}/${NAME}/train \
      --datasetType ${DATASET_TYPE} \
      --use_pix2pix \
      --load_pretrain '/home/tanjiale/pix2pixHD/checkpoints/geopifu_back_v1_global_512p_fixed_lightspot_masked' \
      --which_epoch 10 \
      --ngf 32 \
      --n_blocks_global 12 \
      --fineSize 512 \
      --no_instance \
      --label_nc 0 \
      --verbose
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
#       --use_embedder \
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
