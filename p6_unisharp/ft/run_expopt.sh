set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
cd /raid/git/gaussian-splatting
cd /raid/git/gaussian-splatting
# Train-time exposure compensation. Held-out eval uses identity exposure, so any gain
# comes from a cleaner map rather than from fitting the test image (unlike the diagnostic).
for S in 021 027; do
  C=p6_unisharp/ft/runs/gg_F_021/point_cloud/iteration_30000/point_cloud.ply
  [ "$S" = "027" ] && C=p6_unisharp/ft/runs/gg_F2_027/point_cloud/iteration_30000/point_cloud.ply
  echo "=== $S: refine 15k + EXP_OPT ==="
  env INIT_PLY=$C SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 REFINE_STOP_FRAC=0 RESET_EVERY=999999 \
      DEPTH_DIR=p6_unisharp/ft/depth$S DEPTH_NORM=0.25 TEST_IDX_FILE=p3_pano/fair${S}_test_idx.txt EXP_OPT=1 \
    python p3_pano/train_pano_gsplat_sph.py p3_pano/fair${S}_d2.json p6_unisharp/ft/runs/gg_e1_$S 15000 4096 2>&1 | grep -E "exp-opt|^\[depth\]|EVAL|Traceback|Error"
done
# 027 has no 15k no-exposure control yet (021 does: gg_p0_021)
echo "=== 027: refine 15k, no exposure (control) ==="
env INIT_PLY=p6_unisharp/ft/runs/gg_F2_027/point_cloud/iteration_30000/point_cloud.ply \
    SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 REFINE_STOP_FRAC=0 RESET_EVERY=999999 \
    DEPTH_DIR=p6_unisharp/ft/depth027 DEPTH_NORM=0.25 TEST_IDX_FILE=p3_pano/fair027_test_idx.txt \
  python p3_pano/train_pano_gsplat_sph.py p3_pano/fair027_d2.json p6_unisharp/ft/runs/gg_p0_027 15000 4096 2>&1 | grep -E "^\[depth\]|EVAL"
echo "=== 021: 15k without vs with exposure compensation ==="
python p6_unisharp/ft/eval_res.py p3_pano/fair021_d2.json p3_pano/fair021_test_idx.txt 4096 1024 \
  p6_unisharp/ft/runs/gg_p0_021/point_cloud/iteration_15000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_e1_021/point_cloud/iteration_15000/point_cloud.ply
echo "=== 027: 15k without vs with exposure compensation ==="
python p6_unisharp/ft/eval_res.py p3_pano/fair027_d2.json p3_pano/fair027_test_idx.txt 4096 1024 \
  p6_unisharp/ft/runs/gg_p0_027/point_cloud/iteration_15000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_e1_027/point_cloud/iteration_15000/point_cloud.ply
echo EXPOPT_DONE
