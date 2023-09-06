### train
python main.py \
    --phase train \
    --dataset_dir visdrone \
    --epoch 20 \
    --gpu 0 \
    --n_d 2 \
    --n_scale 2 \
    --L1_lambda 10 \
    --which_direction BtoA \
    --checkpoint_dir ./check/visdrone \
    --sample_dir ./check/visdrone/sample \
    --test_dir ./check/visdrone/testB2A
