### Using labels only
python train.py \
    --name geopifu_back_v1_global_512p_fixed_lightspot_masked \
    --label_nc 0 \
    --no_instance \
    --dataroot /mnt/tanjiale/geopifu_dataset/humanRender_no_config \
    --nThreads 8 \
    --gpu_ids 1 \
    --ngf 32 \
    --ndf 32 \
    --n_blocks_global 12 \
    --batchSize 4 \
    --loadSize 512 \
    --fineSize 512 \
    --continue_train
