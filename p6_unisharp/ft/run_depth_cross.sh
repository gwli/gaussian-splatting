set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
export HF_HOME=/raid/git/gaussian-splatting/p6_unisharp/.hf
export PYTHONPATH=$PYTHONPATH:/raid/git/gaussian-splatting/p6_unisharp/UniSHARP/UniK3D
cd /raid/git/gaussian-splatting
C=p6_unisharp/ft/runs/gg_F2_027/point_cloud/iteration_30000/point_cloud.ply
BASE="SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 REFINE_STOP_FRAC=0 RESET_EVERY=999999"
# 0.05 beat 0.01, so the optimum is not bracketed yet -- probe higher.
echo "=== 027 probe: DEPTH_W=0.15 ==="
env INIT_PLY=$C $BASE DEPTH_DIR=p6_unisharp/ft/depth027 DEPTH_W=0.15 TEST_IDX_FILE=p3_pano/fair027_test_idx.txt \
  python p3_pano/train_pano_gsplat_sph.py p3_pano/fair027_d2.json p6_unisharp/ft/runs/gg_dp0.15027 7000 4096 2>&1 | grep -E "^\[depth\]|EVAL"
python p6_unisharp/ft/eval_res.py p3_pano/fair027_d2.json p3_pano/fair027_test_idx.txt 4096 1024 \
  p6_unisharp/ft/runs/gg_ng027/point_cloud/iteration_7000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_dp0.05027/point_cloud/iteration_7000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_dp0.15027/point_cloud/iteration_7000/point_cloud.ply
# cross-scene: 38b's lesson is that one scene proves nothing
for S in 021 023; do
  [ "$S" = "021" ] && CS=p6_unisharp/ft/runs/gg_F_021/point_cloud/iteration_30000/point_cloud.ply \
                   || CS=p6_unisharp/ft/runs/gg_F/point_cloud/iteration_30000/point_cloud.ply
  echo "=== $S: depth priors ==="
  python p6_unisharp/ft/gen_depth.py p3_pano/fair${S}_d2.json p6_unisharp/ft/depth$S 1024 2>&1 | grep -E "^\[depth\]"
  echo "=== $S: refine + depth (W=0.05) vs refine without ==="
  env INIT_PLY=$CS $BASE DEPTH_DIR=p6_unisharp/ft/depth$S DEPTH_W=0.05 TEST_IDX_FILE=p3_pano/fair${S}_test_idx.txt \
    python p3_pano/train_pano_gsplat_sph.py p3_pano/fair${S}_d2.json p6_unisharp/ft/runs/gg_dp$S 7000 4096 2>&1 | grep -E "^\[depth\]|EVAL"
  python p6_unisharp/ft/eval_res.py p3_pano/fair${S}_d2.json p3_pano/fair${S}_test_idx.txt 4096 1024 \
    $CS p6_unisharp/ft/runs/gg_ng$S/point_cloud/iteration_7000/point_cloud.ply \
    p6_unisharp/ft/runs/gg_dp$S/point_cloud/iteration_7000/point_cloud.ply
done
echo DEPTH_CROSS_DONE
