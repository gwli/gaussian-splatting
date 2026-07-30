set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
cd /raid/git/gaussian-splatting
cd /raid/git/gaussian-splatting
# Which is the residual: geometry (pose) or per-frame photometry (auto-exposure)?
# Same frozen model, same held-out views, four fits. If exposure alone recovers most of
# what pose recovers, the "pose headroom" measured since §34 was largely AE drift, and
# the fix is a per-image photometric term rather than better pose estimation.
for S in 021 027; do
  P=p6_unisharp/ft/runs/gg_p0_021/point_cloud/iteration_15000/point_cloud.ply
  [ "$S" = "027" ] && P=p6_unisharp/ft/runs/gg_dn027/point_cloud/iteration_7000/point_cloud.ply
  for D in trans exp rot+exp all; do
    echo "=== $S  $D ==="
    python p6_unisharp/ft/eval_posefit_dof.py p3_pano/fair${S}_d2.json p3_pano/fair${S}_test_idx.txt \
      $P 2048 300 $D 2>&1 | grep -E "PSNR before|correction applied|Error|Traceback"
  done
done
echo EXPFIT_DONE
