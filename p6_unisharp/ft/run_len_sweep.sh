set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
cd /raid/git/gaussian-splatting
cd /raid/git/gaussian-splatting
# Refine length is scene-dependent (021 gained +0.64 from 7k->15k, 027 lost 0.14), so
# sweep it properly. One 30k run per scene with checkpoints costs 30k iterations instead
# of 3+7+15+22+30 = 77k, and each checkpoint IS the model you get by stopping there.
for S in 021 023 027; do
  C=p6_unisharp/ft/runs/gg_F_021/point_cloud/iteration_30000/point_cloud.ply
  [ "$S" = "023" ] && C=p6_unisharp/ft/runs/gg_F/point_cloud/iteration_30000/point_cloud.ply
  [ "$S" = "027" ] && C=p6_unisharp/ft/runs/gg_F2_027/point_cloud/iteration_30000/point_cloud.ply
  echo "=== $S: refine to 30k, checkpoints at 3k/7k/15k/22k ==="
  env INIT_PLY=$C SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 REFINE_STOP_FRAC=0 RESET_EVERY=999999 \
      DEPTH_DIR=p6_unisharp/ft/depth$S DEPTH_NORM=0.25 TEST_IDX_FILE=p3_pano/fair${S}_test_idx.txt \
      SAVE_AT=3000,7000,15000,22000 \
    python p3_pano/train_pano_gsplat_sph.py p3_pano/fair${S}_d2.json p6_unisharp/ft/runs/gg_len$S 30000 4096 2>&1 | grep -E "^\[depth\]|PLY\]|EVAL|Traceback|Error"
  echo "=== $S: length sweep ==="
  python p6_unisharp/ft/eval_res.py p3_pano/fair${S}_d2.json p3_pano/fair${S}_test_idx.txt 4096 1024 \
    $C \
    p6_unisharp/ft/runs/gg_len$S/point_cloud/iteration_3000/point_cloud.ply \
    p6_unisharp/ft/runs/gg_len$S/point_cloud/iteration_7000/point_cloud.ply \
    p6_unisharp/ft/runs/gg_len$S/point_cloud/iteration_15000/point_cloud.ply \
    p6_unisharp/ft/runs/gg_len$S/point_cloud/iteration_22000/point_cloud.ply \
    p6_unisharp/ft/runs/gg_len$S/point_cloud/iteration_30000/point_cloud.ply
done
echo LEN_DONE
