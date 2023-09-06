### test
python main.py \
    --phase test \
    --dataset_dir multi_src \
    --gpu 0 \
    --n_d 2 \
    --n_scale 2 \
    --which_direction BtoA \
    --single_img False \
    --checkpoint_dir ./check/multi_src \
    --sample_dir ./check/multi_src/sample \
    --test_dir ./check/multi_src/testB2A
    