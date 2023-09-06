### train
python main.py \
    --phase train \
    --dataset_dir multi_src \
    --epoch 20 \
    --gpu 0 \
    --n_d 2 \
    --n_scale 2 \
    --L1_lambda 10 \
    --which_direction BtoA \
    --checkpoint_dir ./check/multi_src \
    --sample_dir ./check/multi_src/sample \
    --test_dir ./check/multi_src/testB2A
