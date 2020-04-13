python main.py \
--model_type='BLSTM' \
--exp_path='./exp' \
--cache_dir='./data_cache' \
--data_dir='../../nas_data/CHiME4' \
--gpu_device='1' \
--num_workers=25 \
--num_epochs=25 \
--dropout=0.5 \
--test_ckpt='./exp/debugging_545/ck_24_141.8975.pth.tar' \
--test_dir='./mask_figures' \
--test_render=True
#--write_cache=False \
# --load_ckpt='./ckpt' \
# --test_ckpt='.'\
