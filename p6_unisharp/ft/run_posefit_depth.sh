set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
export HF_HOME=/raid/git/gaussian-splatting/p6_unisharp/.hf
export PYTHONPATH=$PYTHONPATH:/raid/git/gaussian-splatting/p6_unisharp/UniSHARP/UniK3D
cd /raid/git/gaussian-splatting
# §34 measured 1.113 dB of pose headroom on a model WITHOUT depth supervision. Depth
# has since absorbed part of the geometric error, so the headroom must be re-measured
# before deciding how much pose work is worth. Same diagnostic, new models.
for S in 021 023 027; do
  for D in both rot trans; do
    echo "=== $S  $D ==="
    python p6_unisharp/ft/eval_posefit_dof.py p3_pano/fair${S}_d2.json p3_pano/fair${S}_test_idx.txt \
      p6_unisharp/ft/runs/gg_dn$S/point_cloud/iteration_7000/point_cloud.ply 2048 300 $D 2>&1 | tail -3
  done
done
echo POSEFIT_DONE
