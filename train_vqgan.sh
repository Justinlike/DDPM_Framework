CUDA_VISIBLE_DEVICES=0,1,2 \
VQGAN_MODEL_PARALLEL=1 \
VQGAN_DECODER_GPUS=3 \
python train_vqgan.py \
  model.auto_resume=false \
  model.resume_from_checkpoint=null \
  model.default_root_dir_postfix=parallel3_gan_smoke \
  model.discriminator_iter_start=0 \
  model.max_epochs=1 \
  +model.limit_train_batches=3 \
  +model.limit_val_batches=1 \
  +model.media_logging=false \
  2>&1 | tee vqgan_parallel3_gan_smoke.log