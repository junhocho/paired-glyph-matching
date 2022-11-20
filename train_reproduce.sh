###### O'Donovan dataset

## Paired-glyph matching + Attr
## epoch 3900, 89.91
CUDA_VISIBLE_DEVICES=0 python main.py --phase train-representation \
	--experiment_name rep-paired_glyph_matching+attr --dataset_name donovan_embedding \
	--n_epochs 50000 --check_freq 100 --lr 0.0002 \
	--backbone ResNet18  --feat_dim 512 \
	--heads 70 --simclr --temperature 0.2 --train_attr  

# ## Paired-glyph matching
# ## epoch 7000, 89.60
# CUDA_VISIBLE_DEVICES=0 python main.py --phase train-representation \
# 	--experiment_name rep-paired_glyph_matching --dataset_name donovan_embedding \
# 	--n_epochs 50000 --check_freq 100 --lr 0.0002 \
# 	--backbone ResNet18  --feat_dim 512 \
# 	--heads 70 --simclr --temperature 0.2 

# ## Classifier
# ## epoch 800,  83.90
# CUDA_VISIBLE_DEVICES=0 python main.py --phase train-representation --experiment_name rep-font-cls \
# 	--dataset_name donovan_embedding --data_type 1glyph \
# 	--n_epochs 50000 --check_freq 100  --lr 0.0002 --backbone ResNet18 --train_fontcls 

# ## Style transfer
# ## epoch 4100, 71.84
# CUDA_VISIBLE_DEVICES=0 python main.py --phase train-representation --experiment_name rep-styletransfer \
# 	--dataset_name donovan_embedding --data_type 2glyphs \
# 	--n_epochs 50000 --check_freq 100  --check_L1_gen_freq 500 --lr 0.0002 \
# 	--backbone ResNet18 --feat_dim 512 \
# 	--init_epoch 10900 --train_cae  --no_augmentation 

# ## Autoencoder
# ## epoch 28300, 27.125
# CUDA_VISIBLE_DEVICES=0 python main.py --phase train-representation --experiment_name rep-glyph-autoencoder \
# 	--dataset_name donovan_embedding --data_type 1glyph \
# 	--n_epochs 50000 --check_freq 100  --lr 0.0002 --backbone ResNet18 --train_ae 

####### OFL dataset

# ## Paired-glyph matching
# ## epoch 19400, 91.822
# CUDA_VISIBLE_DEVICES=0 python main.py --phase train-representation \
# 	--experiment_name rep-paired_glyph_matching-ofl --dataset_name ofl \
# 	--n_epochs 50000 --check_freq 100 --lr 0.0002 \
# 	--backbone ResNet18  --feat_dim 1024 \
# 	--heads 70 --simclr --temperature 0.1 

# ## Classifier
# ## epoch 1600, 83.67
# CUDA_VISIBLE_DEVICES=0 python main.py --phase train-representation --experiment_name font-cls-ofl-NoAug-lr1e-5 \
# 	--dataset_name ofl --data_type 1glyph \
# 	--n_epochs 50000 --check_freq 100  --lr 0.00001 --backbone ResNet34 \
# 	--backbone ResNet18  --feat_dim 512 \
# 	--train_fontcls --no_augmentation 

# ## Style transfer
# ## epoch 23300, 82.239
# CUDA_VISIBLE_DEVICES=0 python main.py --phase train-representation --experiment_name cae-NoAug-ofl-dim1024 \
# 	--dataset_name ofl --data_type 2glyphs \
# 	--n_epochs 50000 --check_freq 100  --lr 0.0002 \
# 	--backbone ResNet18 --feat_dim 1024 \
# 	--train_cae --no_augmentation 

# ## Autoencoder
# ## epoch 7500, 15.551
# CUDA_VISIBLE_DEVICES=0 python main.py --phase train-representation --experiment_name glyph-autoencoder-NoAug-ofl \
# 	--dataset_name ofl --data_type 1glyph \
# 	--n_epochs 50000 --check_freq 100  --lr 0.0002 \
# 	--backbone ResNet18 --train_ae  --no_augmentation 
