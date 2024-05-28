### DG
#CUDA_VISIBLE_DEVICES=0 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/nuscenes_lidarseg/day_night/uda/baseline_dg.yaml \
#    --ckpt2d=./output/day_night/UniDSeg_ViT-L-14/model_2d_025000.pth \
#    --ckpt3d=./output/day_night/UniDSeg_ViT-L-14/model_3d_080000.pth \
#    --viz

#CUDA_VISIBLE_DEVICES=0 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/nuscenes_lidarseg/usa_singapore/uda/baseline_dg.yaml \
#    --ckpt2d=./output/usa_singapore/UniDSeg_ViT-L-14/model_2d_060000.pth \
#    --ckpt3d=./output/usa_singapore/UniDSeg_ViT-L-14/model_3d_055000.pth \
#    --viz

#CUDA_VISIBLE_DEVICES=0 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/vkitti_skitti/baseline_dg.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/vkitti_skitti/UniDSeg_ViT-L-14/model_2d_012000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/vkitti_skitti/UniDSeg_ViT-L-14/model_3d_024000.pth \
#    --viz

#CUDA_VISIBLE_DEVICES=0 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/a2d2_skitti/baseline_dg.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/a2d2_skitti/UniDSeg_ViT-L-14/model_2d_105000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/a2d2_skitti/ft_ViT-L-14/model_3d_150000.pth \
#    --viz

### DA (2D/3D/Avg)
#CUDA_VISIBLE_DEVICES=3 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/nuscenes_lidarseg/day_night/baseline_da.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/day_night/da/UniDSeg_ViT-L-14/model_2d_085000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/day_night/da/UniDSeg_ViT-L-14/model_3d_085000.pth \
#    --viz

#CUDA_VISIBLE_DEVICES=3 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/nuscenes_lidarseg/usa_singapore/baseline_da.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/usa_singapore/da/UniDSeg_ViT-L-14/model_2d_015000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/usa_singapore/da/UniDSeg_ViT-L-14/model_3d_050000.pth \
#    --viz

#CUDA_VISIBLE_DEVICES=3 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/vkitti_skitti/baseline_da_color.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/vkitti_skitti/da/UniDSeg_ViT-L-14/model_2d_036000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/vkitti_skitti/da/UniDSeg_ViT-L-14_color/model_3d_024000.pth \
#    --viz

CUDA_VISIBLE_DEVICES=3 python -W ignore xmuda/test_ft_clip.py \
    --cfg=configs/a2d2_skitti/baseline_da.yaml \
    --ckpt2d=/data1/user1/code/UniDSeg/output/a2d2_skitti/da/UniDSeg_ViT-L-14/model_2d_082500.pth \
    --ckpt3d=/data1/user1/code/UniDSeg/output/a2d2_skitti/da/UniDSeg_ViT-L-14/model_3d_135000.pth \
    --viz
