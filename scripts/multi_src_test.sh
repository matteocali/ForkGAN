### test
python main.py \
    --phase test \
    --dataset_dir multi_src \
    --test_dataset_dir multi_src \
    --train_fine_size 256 \
    --fine_size 256 \
    --single_img 'none' \
    --which_direction BtoA \
    --gpu 0 \
    --n_d 2 \
    --n_scale 2 \
    --checkpoint_dir ./check/multi_src_40e \
    --sample_dir ./check/multi_src_40e/sample \
    --test_dir ./check/multi_src_40e/testB2A
    