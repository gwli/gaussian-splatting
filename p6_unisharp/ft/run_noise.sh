set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
cd /raid/git/gaussian-splatting
cd /raid/git/gaussian-splatting
# Noise floor: identical production config, three seeds, two scenes. Every conclusion in
# this log rests on single runs, and several rest on gaps of 0.04-0.2 dB. This measures
# how large a gap has to be before it means anything.
for S in 023 027; do
  C=p6_unisharp/ft/runs/gg_F/point_cloud/iteration_30000/point_cloud.ply
  [ "$S" = "027" ] && C=p6_unisharp/ft/runs/gg_F2_027/point_cloud/iteration_30000/point_cloud.ply
  for SD in 1 2 3; do
    echo "=== $S seed $SD ==="
    env INIT_PLY=$C SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 REFINE_STOP_FRAC=0 RESET_EVERY=999999 \
        DEPTH_DIR=p6_unisharp/ft/depth$S DEPTH_NORM=0.25 TEST_IDX_FILE=p3_pano/fair${S}_test_idx.txt SEED=$SD \
      python p3_pano/train_pano_gsplat_sph.py p3_pano/fair${S}_d2.json p6_unisharp/ft/runs/gg_sd${SD}_$S 30000 4096 2>&1 | grep -E "EVAL|Traceback|Error"
  done
  echo "=== $S: three seeds + the original run ==="
  python p6_unisharp/ft/eval_res.py p3_pano/fair${S}_d2.json p3_pano/fair${S}_test_idx.txt 4096 1024 \
    p6_unisharp/ft/runs/gg_l30$S/point_cloud/iteration_30000/point_cloud.ply \
    p6_unisharp/ft/runs/gg_sd1_$S/point_cloud/iteration_30000/point_cloud.ply \
    p6_unisharp/ft/runs/gg_sd2_$S/point_cloud/iteration_30000/point_cloud.ply \
    p6_unisharp/ft/runs/gg_sd3_$S/point_cloud/iteration_30000/point_cloud.ply
done
echo NOISE_DONE
