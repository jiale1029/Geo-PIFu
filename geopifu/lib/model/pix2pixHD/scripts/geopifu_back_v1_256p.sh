### Using labels only
python train.py \
    --name geopifu_test_pix2pix \
    --label_nc 0 \
    --no_instance \
    --dataroot /mnt/tanjiale/geopifu_dataset/humanRender_no_config \
    --serial_batches \
    --nThreads 8 \
    --gpu_ids 1 \
    --ngf 32 \
    --ndf 32 \
    --n_blocks_global 12 \
    --batchSize 4 \
    --loadSize 512 \
    --fineSize 512
