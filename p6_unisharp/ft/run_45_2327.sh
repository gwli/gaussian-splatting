set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
cd /raid/git/gaussian-splatting
cd /raid/git/gaussian-splatting
# 45k is the cost/quality knee established on 021 (§43b). Bring 023 and 027 to the same
# length so all three scenes sit on the production config.
for S in 023 027; do
  C=p6_unisharp/ft/runs/gg_F/point_cloud/iteration_30000/point_cloud.ply
  [ "$S" = "027" ] && C=p6_unisharp/ft/runs/gg_F2_027/point_cloud/iteration_30000/point_cloud.ply
  echo "=== $S: refine 45k ==="
  env INIT_PLY=$C SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 REFINE_STOP_FRAC=0 RESET_EVERY=999999 \
      DEPTH_DIR=p6_unisharp/ft/depth$S DEPTH_NORM=0.25 TEST_IDX_FILE=p3_pano/fair${S}_test_idx.txt \
    python p3_pano/train_pano_gsplat_sph.py p3_pano/fair${S}_d2.json p6_unisharp/ft/runs/gg_l45$S 45000 4096 2>&1 | grep -E "^\[depth\]|EVAL|Traceback|Error"
  echo "=== $S: coarse | 7k | 30k | 45k ==="
  python p6_unisharp/ft/eval_res.py p3_pano/fair${S}_d2.json p3_pano/fair${S}_test_idx.txt 4096 1024 \
    $C p6_unisharp/ft/runs/gg_dn$S/point_cloud/iteration_7000/point_cloud.ply \
    p6_unisharp/ft/runs/gg_l30$S/point_cloud/iteration_30000/point_cloud.ply \
    p6_unisharp/ft/runs/gg_l45$S/point_cloud/iteration_45000/point_cloud.ply
done
echo L45_2327_DONE
