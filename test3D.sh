CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --num_machines 1 --mixed_precision 'no' --dynamo_backend 'no' main.py \
 --config /data/jzk/DDPM_Framework/conf/config-jct_test.yaml
#   --config /mnt/sda1/lhw/project/CT/new_framework/conf/config-jct_noise_test.yaml