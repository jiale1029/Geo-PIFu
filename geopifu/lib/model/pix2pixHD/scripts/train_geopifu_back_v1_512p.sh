#!/bin/bash

python train.py \
    --name geopifu_back_v1_512p \
    --load_pretrain ./checkpoints/geopifu_back_v1_local_256p \
    --label_nc 0 \
    --no_instance \
    --dataroot /mnt/tanjiale/geopifu_dataset/humanRender_no_config \
    --serial_batches \
    --nThreads 8 \
    --gpu_ids 1 \
    --netG local \
    --ngf 16 \
    --ndf 16 \
    --num_D 3 \
    --n_blocks_local 6 \
    --n_local_enhancers 1 \
    --n_blocks_global 12  \
    --batchSize 4 \
    --loadSize 512 \
    --fineSize 512 \
    --niter_fix_global 5
