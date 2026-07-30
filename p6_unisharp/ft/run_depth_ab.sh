set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
cd /raid/git/gaussian-splatting
C=p6_unisharp/ft/runs/gg_F2_027/point_cloud/iteration_30000/point_cloud.ply
# A/B against the no-depth refine (gg_ng027, 21.499): identical recipe, only the depth
# term differs. Two weights because the log-residual is large (~0.8) on a monocular
# prior, so 0.05 already makes depth ~30% of the loss.
BASE="INIT_PLY=$C SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 REFINE_STOP_FRAC=0 RESET_EVERY=999999 TEST_IDX_FILE=p3_pano/fair027_test_idx.txt DEPTH_DIR=p6_unisharp/ft/depth027"
for Wt in 0.05 0.01; do
  echo "=== refine + depth prior, DEPTH_W=$Wt ==="
  env $BASE DEPTH_W=$Wt python p3_pano/train_pano_gsplat_sph.py p3_pano/fair027_d2.json \
    p6_unisharp/ft/runs/gg_dp${Wt}027 7000 4096 2>&1 | grep -E "\[depth\]|EVAL|DONE|Traceback|Error"
done
echo "=== no-depth refine vs depth-supervised refine ==="
python p6_unisharp/ft/eval_res.py p3_pano/fair027_d2.json p3_pano/fair027_test_idx.txt 4096 1024 \
  $C p6_unisharp/ft/runs/gg_ng027/point_cloud/iteration_7000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_dp0.05027/point_cloud/iteration_7000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_dp0.01027/point_cloud/iteration_7000/point_cloud.ply
echo DEPTH_AB_DONE
