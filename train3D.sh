CUDA_VISIBLE_DEVICES=0 accelerate launch \
--num_processes 1 --num_machines 1 --mixed_precision "no" --dynamo_backend "no" main.py \
--config /data/jzk/DDPM_Framework/conf/config-jct.yaml