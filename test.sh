### Day/Night
#CUDA_VISIBLE_DEVICES=5 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/nuscenes_lidarseg/day_night/baseline_dg.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/day_night/dg/UniDSeg_ViT-L-14/model_2d_035000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/day_night/dg/UniDSeg_ViT-L-14/model_3d_080000.pth \
#CUDA_VISIBLE_DEVICES=1 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/nuscenes_lidarseg/day_night/baseline_da.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/day_night/da/UniDSeg_ViT-L-14/model_2d_085000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/day_night/da/UniDSeg_ViT-L-14/model_3d_085000.pth \
#CUDA_VISIBLE_DEVICES=3 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/nuscenes_lidarseg/day_night/baseline_da_pl.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/day_night/sfda/UniDSeg_ViT-L-14/model_2d_015000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/day_night/sfda/UniDSeg_ViT-L-14/model_3d_060000.pth \


### USA/Singapore
#CUDA_VISIBLE_DEVICES=0 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/nuscenes_lidarseg/usa_singapore/baseline_dg.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/usa_singapore/dg/UniDSeg_ViT-L-14/model_2d_055000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/usa_singapore/dg/UniDSeg_ViT-L-14/model_3d_055000.pth \
#CUDA_VISIBLE_DEVICES=3 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/nuscenes_lidarseg/usa_singapore/baseline_da.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/usa_singapore/da/UniDSeg_ViT-L-14/model_2d_015000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/usa_singapore/da/UniDSeg_ViT-L-14/model_3d_050000.pth \
#CUDA_VISIBLE_DEVICES=0 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/nuscenes_lidarseg/usa_singapore/baseline_da_pl.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/usa_singapore/sfda/UniDSeg_ViT-L-14/model_2d_020000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/usa_singapore/sfda/UniDSeg_ViT-L-14/model_3d_045000.pth \


### Singapore/USA
#CUDA_VISIBLE_DEVICES=0 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/nuscenes_lidarseg/singapore_usa/baseline_dg.yaml \
#    --ckpt2d=/data1/user1/code/UniDSeg/output/singapore_usa/dg/UniDSeg_ViT-L-14/model_2d_025000.pth \
#    --ckpt3d=/data1/user1/code/UniDSeg/output/singapore_usa/dg/UniDSeg_ViT-L-14/model_3d_060000.pth \


### vkitti/skitti
#CUDA_VISIBLE_DEVICES=0 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/vkitti_skitti/baseline_dg.yaml \
#    --ckpt2d=/data1/wuyao/code/UniDSeg/output/vkitti_skitti/dg/UniDSeg_ViT-L-14/model_2d_027000.pth \
#    --ckpt3d=/data1/wuyao/code/UniDSeg/output/vkitti_skitti/dg/UniDSeg_ViT-L-14/model_3d_024000.pth \
#CUDA_VISIBLE_DEVICES=3 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/vkitti_skitti/baseline_da.yaml \
#    --ckpt2d=/data1/wuyao/code/UniDSeg/output/vkitti_skitti/da/UniDSeg_ViT-L-14/model_2d_036000.pth \
#    --ckpt3d=/data1/wuyao/code/UniDSeg/output/vkitti_skitti/da/UniDSeg_ViT-L-14/model_3d_060000.pth \


### a2d2/skitti
#CUDA_VISIBLE_DEVICES=0 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/a2d2_skitti/baseline_dg.yaml \
#    --ckpt2d=/data1/wuyao/code/UniDSeg/output/a2d2_skitti/dg/UniDSeg_ViT-L-14/model_2d_105000.pth \
#    --ckpt3d=/data1/wuyao/code/UniDSeg/output/a2d2_skitti/dg/UniDSeg_ViT-L-14/model_3d_150000.pth \
#CUDA_VISIBLE_DEVICES=0 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/a2d2_skitti/baseline_da.yaml \
#    --ckpt2d=/data1/wuyao/code/UniDSeg/output/a2d2_skitti/da/UniDSeg_ViT-L-14/model_2d_082500.pth \
#    --ckpt3d=/data1/wuyao/code/UniDSeg/output/a2d2_skitti/da/UniDSeg_ViT-L-14/model_3d_135000.pth \
#CUDA_VISIBLE_DEVICES=3 python -W ignore xmuda/test_ft_clip.py \
#    --cfg=configs/a2d2_skitti/baseline_da_pl.yaml \
#    --ckpt2d=/data1/wuyao/code/UniDSeg/output/a2d2_skitti/sfda/UniDSeg_ViT-L-14/model_2d_022500.pth \
#    --ckpt3d=/data1/wuyao/code/UniDSeg/output/a2d2_skitti/sfda/UniDSeg_ViT-L-14/model_3d_060000.pth \

