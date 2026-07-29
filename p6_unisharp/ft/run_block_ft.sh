set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
cd /raid/git/gaussian-splatting
COARSE=p6_unisharp/ft/runs/gg_F_027/point_cloud/iteration_30000/point_cloud.ply
for B in 0 1; do
  echo "=== block $B: FINE-TUNE from the coarse model (their method) ==="
  INIT_PLY=$COARSE BLOCK_JSON=p3_pano/blk027_blocks.json BLOCK_ID=$B \
  SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 GROW_GRAD2D=0.0002 REFINE_STOP_FRAC=0.5 \
  TEST_IDX_FILE=p3_pano/fair027_test_idx.txt \
    python p3_pano/train_pano_gsplat_sph.py p3_pano/blk027_b$B.json p6_unisharp/ft/runs/gg_ft027_b$B 30000 4096 2>&1 | grep -Ev "^\s*$" | tail -8
done
python p6_unisharp/ft/block_merge.py p3_pano/blk027_blocks.json p6_unisharp/ft/runs/gg_ft027_merged.ply \
  p6_unisharp/ft/runs/gg_ft027_b0/point_cloud/iteration_30000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_ft027_b1/point_cloud/iteration_30000/point_cloud.ply
echo "=== coarse (global) vs coarse+block-finetune (their two-stage) ==="
python p6_unisharp/ft/eval_res.py p3_pano/fair027_d2.json p3_pano/fair027_test_idx.txt 4096 1024 \
  $COARSE p6_unisharp/ft/runs/gg_ft027_merged.ply
echo FT_BLOCK_DONE
