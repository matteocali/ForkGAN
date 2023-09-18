### test
python main.py \
    --phase test \
    --dataset_dir fog_mix \
    --test_dataset_dir uavid \
    --train_fine_size 256 \
    --fine_size 512 \
    --single_img 'std' \
    --which_direction BtoA \
    --gpu 0 \
    --n_d 2 \
    --n_scale 2 \
    --checkpoint_dir ./check/fog_mix \
    --sample_dir ./check/fog_mix/sample \
    --test_dir ./check/fog_mix/testB2A_uavid
    