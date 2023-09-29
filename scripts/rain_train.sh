### test
python main.py \
    --phase train \
    --dataset_dir bdd100k_acdc_synth \
    --epoch 40 \
    --gpu 0 \
    --n_d 2 \
    --n_scale 2 \
    --L1_lambda 10 \
    --batch_size 1 \
    --which_direction BtoA \
    --checkpoint_dir ./check/bdd100k_acdc_synth \
    --sample_dir ./check/bdd100k_acdc_synth/sample \
    --test_dir ./check/bdd100k_acdc_synth/uavid_night_refine
    