MESH_NAME="nezuko"
IMG_PATH="/home/tanjiale/test_image/${MESH_NAME}.png"
MASK_PATH="/home/tanjiale/test_image/${MESH_NAME}_binarized.png"
DEEP_VOXEL_PATH="./singleImage/${MESH_NAME}/${MESH_NAME}_deepVoxels.npy"
RESULT_DIR="./singleImage/${MESH_NAME}"

NAME="ICCV_GeoPIFu_coarse"
NAME_QUERY="ICCV_GeoPIFu_query"
NAME_COLOR="ICCV_GeoPIFu_color_simulated_multiview_fixed_with_pix2pix_global"
netV_checkpoint="./checkpoints/${NAME}/netV_epoch_44_2899"
netG_checkpoint="./checkpoints/${NAME_QUERY}/netG_epoch_44_2415"
netC_checkpoint="./checkpoints/${NAME_COLOR}/netC_epoch_4_43487"

export CUDA_VISIBLE_DEVICES="1" && python -m apps.test_shape_coarse \
      --gpu_id 0 \
      --load_netV_checkpoint_path ./checkpoints/ICCV_GeoPIFu_coarse/netV_epoch_29_2899 \
      --load_from_multi_GPU_shape \
      --batch_size 1 \
      --img_path ${IMG_PATH} \
      --mask_path ${MASK_PATH} \
      --mesh_name ${MESH_NAME} \
      --resultsDir ${RESULT_DIR} && \
export CUDA_VISIBLE_DEVICES="1" && python -m apps.test_shape_iccv \
      --resultsDir ${RESULT_DIR} \
      --deepVoxels_fusion early \
      --deepVoxels_c_len 56 \
      --multiRanges_deepVoxels \
      --gpu_id 0 \
      --phase train \
      --load_netG_checkpoint_path ${netG_checkpoint} \
      --load_netC_checkpoint_path ${netC_checkpoint} \
      --load_from_multi_GPU_shape \
      --load_pretrain '/home/tanjiale/pix2pixHD/checkpoints/geopifu_back_v1_global_512p_fixed_lightspot_masked' \
      --which_epoch 7 \
      --use_pix2pix \
      --ngf 32 \
      --n_blocks_global 12 \
      --fineSize 512 \
      --no_instance \
      --label_nc 0 \
      --img_path ${IMG_PATH} \
      --mask_path ${MASK_PATH} \
      --mesh_name ${MESH_NAME} \
      --deepVoxelPath ${DEEP_VOXEL_PATH}
