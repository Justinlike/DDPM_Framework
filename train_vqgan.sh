CUDA_VISIBLE_DEVICES=0 \
python -u train_vqgan.py \
  model.gpus=1 \
  model.auto_resume=false \
  model.resume_from_checkpoint=null \
  model.default_root_dir_postfix=single_gan_train_E8_N32