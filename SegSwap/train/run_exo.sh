cd ..

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29501 eval_segswap.py \
    --ckpt_path train/output/1102_dinov3cnlarge_dinov3large_dice5_bs16as16ep200_mxlr1e5_lp20lr1e4/best_test_miou.pth \
    --data_path true_data \
    --splits_path data/split.json \
    --split test \
    --out_path output/1110_dinov3cnlarge_dinov3large_dice5_bs16as16ep200_mxlr1e5_lp20lr1e4_tttlayers11_iter6 \
    --setting exo-ego \
    --image_size 512 \
    --backbone_type dinov3 \
    --extractor_type dinov3_cn_large \
    --distributed \
    --ttt_enable \
    --ttt_layers 11 \
    --ttt_iter 6 \
    --use_amp