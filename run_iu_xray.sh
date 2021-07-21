python main.py \
--image_dir data/iu_xray/images/ \
--ann_path data/iu_xray/annotation.json \
--dataset_name iu_xray \
--max_seq_length 60 \
--threshold 3 \
--visual_extractor hrnet \
--batch_size 8 \
--epochs 20 \
--save_dir results/iu_xray-densenet \
--step_size 50 \
--gamma 0.1 \
--seed 9223 \
--d_vf 1664 --lr_ve 1e-4
# --lr_ve 25e-7 \
# --lr_ed 5e-6 \
# --resume /home/teamc/dwij/R2Gen/results/iu_xray/model_best.pth
