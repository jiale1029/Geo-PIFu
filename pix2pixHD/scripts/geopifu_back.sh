### Using labels only
# python train.py --name geopifu_back --label_nc 0 --no_instance --dataroot /mnt/tanjiale/geopifu_dataset/humanRender_no_config --serial_batches --nThreads 8 --gpu_ids 1 --ngf 32 --ndf 32 --continue_train --which_epoch latest

### Using labels only
python test.py \
    --name geopifu_back_v1 \
    --results_dir /mnt/tanjiale/pix2pix \
    --label_nc 0 \
    --no_instance \
    --dataroot /mnt/tanjiale/geopifu_dataset/humanRender_no_config \
    --phase test \
    --gpu_ids 0 \
    --netG global \
    --ngf 32 \
    --n_blocks_global 12 \
    --loadSize 512 \
    --fineSize 512 \
    --resize_or_crop resize_and_crop \
    --which_epoch 10 \
    --how_many 99999

