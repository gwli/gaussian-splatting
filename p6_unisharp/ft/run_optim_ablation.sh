set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
cd /raid/git/gaussian-splatting
COARSE=p6_unisharp/ft/runs/gg_F2_027/point_cloud/iteration_30000/point_cloud.ply
FILT='grep -E "resume|DONE|EVAL|Traceback|Error"'
# identical recipe, identical iterations; the ONLY difference is whether the Adam
# moments are restored. This is the control the -0.87 dB claim in 39 never had.
BASE="INIT_PLY=$COARSE SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 GROW_GRAD2D=0.0002 REFINE_STOP_FRAC=0.3 RESET_EVERY=999999 TEST_IDX_FILE=p3_pano/fair027_test_idx.txt"
echo "=== warm optimiser (Adam moments restored) ==="
env $BASE python p3_pano/train_pano_gsplat_sph.py p3_pano/fair027_d2.json p6_unisharp/ft/runs/gg_warm027 7000 4096 2>&1 | eval $FILT
echo "=== cold optimiser (same recipe, moments discarded) ==="
env $BASE COLD_OPTIM=1 python p3_pano/train_pano_gsplat_sph.py p3_pano/fair027_d2.json p6_unisharp/ft/runs/gg_cold027 7000 4096 2>&1 | eval $FILT
echo "=== isolating the optimiser-state term ==="
python p6_unisharp/ft/eval_res.py p3_pano/fair027_d2.json p3_pano/fair027_test_idx.txt 4096 1024 \
  $COARSE p6_unisharp/ft/runs/gg_warm027/point_cloud/iteration_7000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_cold027/point_cloud/iteration_7000/point_cloud.ply
echo ABL_DONE
