### test
python main.py \
    --phase test \
    --dataset_dir multi_src \
    --test_dataset_dir uavid_night \
    --fine_size 256 \
    --img_out_shape 2160,3840 \
    --single_img 'refine' \
    --which_direction BtoA \
    --gpu 0 \
    --n_d 2 \
    --n_scale 2 \
    --checkpoint_dir ./check/multi_src_40e \
    --sample_dir ./check/multi_src_40e/sample \
    --test_dir ./check/multi_src_40e/uavid_night_refine
    