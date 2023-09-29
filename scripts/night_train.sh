### test
python main.py \
    --phase train \
    --dataset_dir multi_src \
    --epoch 40 \
    --gpu 0 \
    --n_d 2 \
    --n_scale 2 \
    --L1_lambda 10 \
    --batch_size 1 \
    --which_direction BtoA \
    --checkpoint_dir ./check/multi_src_40e \
    --sample_dir ./check/multi_src_40e/sample \
    --test_dir ./check/multi_src_40e/uavid_night_refine
    