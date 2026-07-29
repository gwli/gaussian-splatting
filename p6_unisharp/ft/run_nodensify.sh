set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
cd /raid/git/gaussian-splatting
COARSE=p6_unisharp/ft/runs/gg_F2_027/point_cloud/iteration_30000/point_cloud.ply
FILT='grep -E "block |resume|DONE|EVAL|Traceback|Error"'
# capacity test: fine-tune with densification OFF (REFINE_STOP_FRAC=0) and no opacity
# reset, so the gaussian count stays at the coarse model's. If PSNR now holds while
# LPIPS/sharpness still improve, the drop was capacity under pose error, not the method.
FT="INIT_PLY=$COARSE SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 REFINE_STOP_FRAC=0 RESET_EVERY=999999 TEST_IDX_FILE=p3_pano/fair027_test_idx.txt"
echo "=== global fine-tune, densification OFF ==="
env $FT python p3_pano/train_pano_gsplat_sph.py p3_pano/fair027_d2.json p6_unisharp/ft/runs/gg_ng027 7000 4096 2>&1 | eval $FILT
for B in 0 1; do
  echo "=== block $B fine-tune, densification OFF ==="
  env $FT BLOCK_JSON=p3_pano/blk027_blocks.json BLOCK_ID=$B \
    python p3_pano/train_pano_gsplat_sph.py p3_pano/blk027_b$B.json p6_unisharp/ft/runs/gg_nb027_b$B 7000 4096 2>&1 | eval $FILT
done
python p6_unisharp/ft/block_merge.py p3_pano/blk027_blocks.json p6_unisharp/ft/runs/gg_nb027_merged.ply \
  p6_unisharp/ft/runs/gg_nb027_b0/point_cloud/iteration_7000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_nb027_b1/point_cloud/iteration_7000/point_cloud.ply
echo "=== capacity held fixed: coarse vs global-ft vs block-ft ==="
python p6_unisharp/ft/eval_res.py p3_pano/fair027_d2.json p3_pano/fair027_test_idx.txt 4096 1024 \
  $COARSE p6_unisharp/ft/runs/gg_ng027/point_cloud/iteration_7000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_nb027_merged.ply
echo ND_DONE
