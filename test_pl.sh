#CUDA_VISIBLE_DEVICES=0 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/nuscenes_lidarseg/day_night/baseline_dg.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/day_night/dg/UniDSeg_ViT-L-14/model_2d_100000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/day_night/dg/UniDSeg_ViT-L-14/model_3d_100000.pth \
#    --pselab \
#    DATASET_TARGET.TEST "('train_night',)"
#CUDA_VISIBLE_DEVICES=0 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/nuscenes_lidarseg/day_night/baseline_da.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/day_night/da/UniDSeg_ViT-L-14/model_2d_100000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/day_night/da/UniDSeg_ViT-L-14/model_3d_100000.pth \
#    --pselab \
#    DATASET_TARGET.TEST "('train_night',)"

#CUDA_VISIBLE_DEVICES=0 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/nuscenes_lidarseg/usa_singapore/baseline_dg.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/usa_singapore/dg/UniDSeg_ViT-L-14/model_2d_060000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/usa_singapore/dg/UniDSeg_ViT-L-14/model_3d_060000.pth \
#    --pselab \
#    DATASET_TARGET.TEST "('train_singapore',)"
#CUDA_VISIBLE_DEVICES=0 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/nuscenes_lidarseg/usa_singapore/baseline_da.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/usa_singapore/da/UniDSeg_ViT-L-14/model_2d_060000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/usa_singapore/da/UniDSeg_ViT-L-14/model_3d_060000.pth \
#    --pselab \
#    DATASET_TARGET.TEST "('train_singapore',)"

#CUDA_VISIBLE_DEVICES=4 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/vkitti_skitti/baseline_dg.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/vkitti_skitti/dg/UniDSeg_ViT-L-14/model_2d_060000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/vkitti_skitti/dg/UniDSeg_ViT-L-14/model_3d_060000.pth \
#    --pselab \
#    DATASET_TARGET.TEST "('train',)"
#CUDA_VISIBLE_DEVICES=4 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/vkitti_skitti/baseline_da.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/vkitti_skitti/da/UniDSeg_ViT-L-14/model_2d_060000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/vkitti_skitti/da/UniDSeg_ViT-L-14/model_3d_060000.pth \
#    --pselab \
#    DATASET_TARGET.TEST "('train',)"

#CUDA_VISIBLE_DEVICES=0 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/a2d2_skitti/baseline_dg.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/a2d2_skitti/dg/UniDSeg_ViT-L-14/model_2d_150000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/a2d2_skitti/dg/UniDSeg_ViT-L-14/model_3d_150000.pth \
#    --pselab \
#    DATASET_TARGET.TEST "('train',)"
#CUDA_VISIBLE_DEVICES=4 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/a2d2_skitti/baseline_da.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/a2d2_skitti/da/UniDSeg_ViT-L-14/model_2d_150000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/a2d2_skitti/da/UniDSeg_ViT-L-14/model_3d_150000.pth \
#    --pselab \
#    DATASET_TARGET.TEST "('train',)"
