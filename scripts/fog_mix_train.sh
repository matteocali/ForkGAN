### train
python main.py \
    --phase train \
    --dataset_dir fog_mix \
    --epoch 40 \
    --gpu 0 \
    --n_d 2 \
    --n_scale 2 \
    --L1_lambda 10 \
    --batch_size 1 \
    --which_direction BtoA \
    --checkpoint_dir ./check/fog_mix \
    --sample_dir ./check/fog_mix/sample \
    --test_dir ./check/fog_mix/testB2A_uavid
