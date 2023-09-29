### test
python main.py \
    --phase test \
    --dataset_dir bdd100k_acdc_synth \
    --test_dataset_dir uavid_rain \
    --fine_size 256 \
    --img_out_shape 2160,3840 \
    --single_img 'refine' \
    --which_direction BtoA \
    --gpu 0 \
    --n_d 2 \
    --n_scale 2 \
    --checkpoint_dir ./check/bdd100k_acdc_synth \
    --sample_dir ./check/bdd100k_acdc_synth/sample \
    --test_dir ./check/bdd100k_acdc_synth/uavid_rain_refine
    