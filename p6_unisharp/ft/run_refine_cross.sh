set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
cd /raid/git/gaussian-splatting
# cross-scene check of the capacity-frozen refine stage (027 gave +0.48 dB on all four
# metrics). 38b's lesson: a single-scene win does not generalise -- verify before adopting.
for S in 021 023; do
  C=p6_unisharp/ft/runs/gg_F_$S/point_cloud/iteration_30000/point_cloud.ply
  echo "=== $S: refine from coarse, densification OFF ==="
  env INIT_PLY=$C SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 REFINE_STOP_FRAC=0 RESET_EVERY=999999 \
      TEST_IDX_FILE=p3_pano/fair${S}_test_idx.txt \
    python p3_pano/train_pano_gsplat_sph.py p3_pano/fair${S}_d2.json p6_unisharp/ft/runs/gg_ng$S 7000 4096 2>&1 | grep -E "resume|DONE|EVAL|Traceback|Error"
  python p6_unisharp/ft/eval_res.py p3_pano/fair${S}_d2.json p3_pano/fair${S}_test_idx.txt 4096 1024 \
    $C p6_unisharp/ft/runs/gg_ng$S/point_cloud/iteration_7000/point_cloud.ply
done
echo CROSS_DONE
