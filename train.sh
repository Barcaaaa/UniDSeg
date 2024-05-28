#CUDA_LAUNCH_BLOCKING=1

# fine-tuning
#CUDA_VISIBLE_DEVICES=0 nohup python -W ignore xmuda/train_dg_clip_ft.py --cfg=configs/nuscenes_lidarseg/day_night/baseline_dg.yaml >out_01.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python -W ignore xmuda/train_dg_clip_ft.py --cfg=configs/nuscenes_lidarseg/usa_singapore/baseline_dg.yaml >out_01.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python -W ignore xmuda/train_dg_clip_ft.py --cfg=configs/nuscenes_lidarseg/singapore_usa/baseline_dg.yaml >out_01.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python -W ignore xmuda/train_dg_clip_ft.py --cfg=configs/vkitti_skitti/baseline_dg.yaml >out_03.log 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -W ignore xmuda/train_dg_clip_ft.py --cfg=configs/a2d2_skitti/baseline_dg.yaml >out_01.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python -W ignore xmuda/train_dg_clip_ft.py --cfg=configs/a2d2_nuscenes/baseline_dg.yaml >out_03.log 2>&1 &

# frozen
#CUDA_VISIBLE_DEVICES=1 nohup python -W ignore xmuda/train_dg_clip_fz.py --cfg=configs/nuscenes_lidarseg/day_night/baseline_dg.yaml >out_02.log 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -W ignore xmuda/train_dg_clip_fz.py --cfg=configs/nuscenes_lidarseg/usa_singapore/baseline_dg.yaml >out_02.log 2>&1 &
#CUDA_VISIBLE_DEVICES=2 nohup python -W ignore xmuda/train_dg_clip_fz.py --cfg=configs/nuscenes_lidarseg/singapore_usa/baseline_dg.yaml >out_02.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python -W ignore xmuda/train_dg_clip_fz.py --cfg=configs/vkitti_skitti/baseline_dg.yaml >out_04.log 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -W ignore xmuda/train_dg_clip_fz.py --cfg=configs/a2d2_skitti/baseline_dg.yaml >out_01.log 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -W ignore xmuda/train_dg_clip_fz.py --cfg=configs/a2d2_nuscenes/baseline_dg.yaml >out_04.log 2>&1 &

# UniDSeg
# domain generalization
#CUDA_VISIBLE_DEVICES=0 nohup python -W ignore xmuda/train_dg_clip_UniDSeg.py --cfg=configs/nuscenes_lidarseg/day_night/baseline_dg.yaml >out_01.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python -W ignore xmuda/train_dg_clip_UniDSeg.py --cfg=configs/nuscenes_lidarseg/usa_singapore/baseline_dg.yaml >out_03.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python -W ignore xmuda/train_dg_clip_UniDSeg.py --cfg=configs/nuscenes_lidarseg/singapore_usa/baseline_dg.yaml >out_01.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python -W ignore xmuda/train_dg_clip_UniDSeg.py --cfg=configs/vkitti_skitti/baseline_dg.yaml >out_03.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python -W ignore xmuda/train_dg_clip_UniDSeg.py --cfg=configs/a2d2_skitti/baseline_dg.yaml >out_01.log 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python -W ignore xmuda/train_dg_clip_UniDSeg.py --cfg=configs/a2d2_nuscenes/baseline_dg.yaml >out_01.log 2>&1 &
# domain adaptation
#CUDA_VISIBLE_DEVICES=0 nohup python -W ignore xmuda/train_da_clip_UniDSeg.py --cfg=configs/nuscenes_lidarseg/usa_singapore/baseline_da.yaml >out_01.log 2>&1 &
#CUDA_VISIBLE_DEVICES=1 nohup python -W ignore xmuda/train_da_clip_UniDSeg.py --cfg=configs/nuscenes_lidarseg/day_night/baseline_da.yaml >out_02.log 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python -W ignore xmuda/train_da_clip_UniDSeg.py --cfg=configs/vkitti_skitti/baseline_da.yaml >out_03.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python -W ignore xmuda/train_da_clip_UniDSeg.py --cfg=configs/a2d2_skitti/baseline_da.yaml >out_01.log 2>&1 &
# source-free domain adaptation
#CUDA_VISIBLE_DEVICES=0 nohup python -W ignore xmuda/train_sfda_clip_UniDSeg.py --cfg=configs/nuscenes_lidarseg/usa_singapore/baseline_da_pl.yaml >out_03.log 2>&1 &
#CUDA_VISIBLE_DEVICES=3 nohup python -W ignore xmuda/train_sfda_clip_UniDSeg.py --cfg=configs/nuscenes_lidarseg/day_night/baseline_da_pl.yaml >out_04.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0 nohup python -W ignore xmuda/train_sfda_clip_UniDSeg.py --cfg=configs/a2d2_skitti/baseline_da_pl.yaml >out_05.log 2>&1 &

