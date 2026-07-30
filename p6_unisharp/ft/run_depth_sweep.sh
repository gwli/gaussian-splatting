set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
export HF_HOME=/raid/git/gaussian-splatting/p6_unisharp/.hf
export PYTHONPATH=$PYTHONPATH:/raid/git/gaussian-splatting/p6_unisharp/UniSHARP/UniK3D
cd /raid/git/gaussian-splatting
BASE="SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 REFINE_STOP_FRAC=0 RESET_EVERY=999999"
# 0.15 > 0.05 > 0.01 on 027, so the optimum is still not bracketed. Note the log
# residual differs per scene (0.81 / 1.29 / 1.51), so a fixed weight is not a fixed
# strength -- the sweep tells us whether the optimum tracks the residual.
run () {  # scene, coarse ply, weight
  env INIT_PLY=$2 $BASE DEPTH_DIR=p6_unisharp/ft/depth$1 DEPTH_W=$3 TEST_IDX_FILE=p3_pano/fair$1_test_idx.txt \
    python p3_pano/train_pano_gsplat_sph.py p3_pano/fair$1_d2.json p6_unisharp/ft/runs/gg_dw$3_$1 7000 4096 2>&1 | grep -E "^\[depth\]|EVAL"
}
C27=p6_unisharp/ft/runs/gg_F2_027/point_cloud/iteration_30000/point_cloud.ply
C21=p6_unisharp/ft/runs/gg_F_021/point_cloud/iteration_30000/point_cloud.ply
C23=p6_unisharp/ft/runs/gg_F/point_cloud/iteration_30000/point_cloud.ply
echo "=== 027: W=0.4, 1.0 ==="; run 027 $C27 0.4; run 027 $C27 1.0
echo "=== 021: W=0.15, 0.4 ==="; run 021 $C21 0.15; run 021 $C21 0.4
echo "=== 023: W=0.15, 0.4 ==="; run 023 $C23 0.15; run 023 $C23 0.4
for S in 027 021 023; do
  echo "=== $S sweep ==="
  python p6_unisharp/ft/eval_res.py p3_pano/fair${S}_d2.json p3_pano/fair${S}_test_idx.txt 4096 1024 \
    p6_unisharp/ft/runs/gg_ng$S/point_cloud/iteration_7000/point_cloud.ply \
    p6_unisharp/ft/runs/gg_dw*_$S/point_cloud/iteration_7000/point_cloud.ply 2>/dev/null
done
echo SWEEP_DONE
