### test
python main.py \
    --phase test \
    --dataset_dir visdrone \
    --gpu 0 \
    --n_d 2 \
    --n_scale 2 \
    --which_direction BtoA \
    --checkpoint_dir ./check/visdrone \
    --sample_dir ./check/visdrone/sample \
    --test_dir ./check/visdrone/testB2A
    