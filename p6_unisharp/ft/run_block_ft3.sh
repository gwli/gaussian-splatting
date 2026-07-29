set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
cd /raid/git/gaussian-splatting
set -o pipefail
FILT='grep -E "block |resume|CKPT|sky-lock|skybox|DONE|EVAL|Traceback|Error"'

# stage 1: retrain the coarse model, this time checkpointing the optimiser state.
# (gg_F_027 predates the checkpoint, so every fine-tune from it started cold.)
echo "=== stage 1: coarse (F config) with optimiser checkpoint ==="
env ABSGRAD=1 GROW_GRAD2D=0.0004 SKYBOX_NUM=100000 TEST_IDX_FILE=p3_pano/fair027_test_idx.txt \
  python p3_pano/train_pano_gsplat_sph.py p3_pano/fair027_d2.json p6_unisharp/ft/runs/gg_F2_027 30000 4096 2>&1 | eval $FILT
COARSE=p6_unisharp/ft/runs/gg_F2_027/point_cloud/iteration_30000/point_cloud.ply

# their block stage: 30k per block, lr x0.4, grad_abs 2e-4, sky inherited + locked
FT="SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 GROW_GRAD2D=0.0002 TEST_IDX_FILE=p3_pano/fair027_test_idx.txt"
echo "=== stage 2: GLOBAL fine-tune 30k (control, warm optimiser) ==="
env INIT_PLY=$COARSE $FT python p3_pano/train_pano_gsplat_sph.py p3_pano/fair027_d2.json p6_unisharp/ft/runs/gg_wg027 30000 4096 2>&1 | eval $FILT
for B in 0 1; do
  echo "=== stage 3: block $B fine-tune 30k (warm optimiser) ==="
  env INIT_PLY=$COARSE BLOCK_JSON=p3_pano/blk027_blocks.json BLOCK_ID=$B $FT \
    python p3_pano/train_pano_gsplat_sph.py p3_pano/blk027_b$B.json p6_unisharp/ft/runs/gg_wb027_b$B 30000 4096 2>&1 | eval $FILT
done
python p6_unisharp/ft/block_merge.py p3_pano/blk027_blocks.json p6_unisharp/ft/runs/gg_wb027_merged.ply \
  p6_unisharp/ft/runs/gg_wb027_b0/point_cloud/iteration_30000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_wb027_b1/point_cloud/iteration_30000/point_cloud.ply
echo "=== coarse vs global-ft vs block-ft, all with warm optimiser state ==="
python p6_unisharp/ft/eval_res.py p3_pano/fair027_d2.json p3_pano/fair027_test_idx.txt 4096 1024 \
  $COARSE p6_unisharp/ft/runs/gg_wg027/point_cloud/iteration_30000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_wb027_merged.ply
echo FT3_DONE
