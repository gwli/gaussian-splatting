set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
export HF_HOME=/raid/git/gaussian-splatting/p6_unisharp/.hf
export PYTHONPATH=$PYTHONPATH:/raid/git/gaussian-splatting/p6_unisharp/UniSHARP/UniK3D
cd /raid/git/gaussian-splatting
BASE="SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 REFINE_STOP_FRAC=0 RESET_EVERY=999999"
# per-scene optima had effective strengths 0.32 / 0.23 / 0.19; 0.25 is the compromise.
# The question is whether one auto-normalised constant lands near each hand-tuned peak,
# because the 7-scene rollout cannot sweep every scene.
C27=p6_unisharp/ft/runs/gg_F2_027/point_cloud/iteration_30000/point_cloud.ply
C21=p6_unisharp/ft/runs/gg_F_021/point_cloud/iteration_30000/point_cloud.ply
C23=p6_unisharp/ft/runs/gg_F/point_cloud/iteration_30000/point_cloud.ply
for SC in "027 $C27" "021 $C21" "023 $C23"; do
  set -- $SC
  echo "=== $1: DEPTH_NORM=0.25 (auto weight) ==="
  env INIT_PLY=$2 $BASE DEPTH_DIR=p6_unisharp/ft/depth$1 DEPTH_NORM=0.25 TEST_IDX_FILE=p3_pano/fair$1_test_idx.txt \
    python p3_pano/train_pano_gsplat_sph.py p3_pano/fair$1_d2.json p6_unisharp/ft/runs/gg_dn$1 7000 4096 2>&1 | grep -E "^\[depth\]|EVAL"
done
echo "=== auto-normalised vs hand-tuned peak ==="
python p6_unisharp/ft/eval_res.py p3_pano/fair027_d2.json p3_pano/fair027_test_idx.txt 4096 1024 \
  p6_unisharp/ft/runs/gg_dw0.4_027/point_cloud/iteration_7000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_dn027/point_cloud/iteration_7000/point_cloud.ply
python p6_unisharp/ft/eval_res.py p3_pano/fair021_d2.json p3_pano/fair021_test_idx.txt 4096 1024 \
  p6_unisharp/ft/runs/gg_dw0.15_021/point_cloud/iteration_7000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_dn021/point_cloud/iteration_7000/point_cloud.ply
python p6_unisharp/ft/eval_res.py p3_pano/fair023_d2.json p3_pano/fair023_test_idx.txt 4096 1024 \
  p6_unisharp/ft/runs/gg_dw0.15_023/point_cloud/iteration_7000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_dn023/point_cloud/iteration_7000/point_cloud.ply
echo NORM_DONE
