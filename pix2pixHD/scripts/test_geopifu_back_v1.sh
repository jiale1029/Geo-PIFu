### Using labels only
python test.py \
    --name geopifu_back_v1_global_512p_fixed_lightspot_masked \
    --results_dir /mnt/tanjiale/pix2pix \
    --label_nc 0 \
    --no_instance \
    --dataroot /mnt/tanjiale/geopifu_dataset/humanRender_no_config \
    --phase test \
    --gpu_ids 0 \
    --ngf 32 \
    --loadSize 512 \
    --fineSize 512 \
    --n_blocks_global 12 \
    --which_epoch 10 \
    --how_many 99999
