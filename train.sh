export CUDA_VISIBLE_DEVICES=1
python train.py -data_pkl .data/m30k_deen_shr.pkl -log logs/m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model checkpoints/trained -b 256 -warmup 128000 -epoch 400