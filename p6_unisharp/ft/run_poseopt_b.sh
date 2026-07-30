set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
cd /raid/git/gaussian-splatting
cd /raid/git/gaussian-splatting
C=p6_unisharp/ft/runs/gg_F_021/point_cloud/iteration_30000/point_cloud.ply
echo "=== B: depth refine 15k + in-training pose optimisation ==="
env INIT_PLY=$C SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 REFINE_STOP_FRAC=0 RESET_EVERY=999999 \
    DEPTH_DIR=p6_unisharp/ft/depth021 DEPTH_NORM=0.25 TEST_IDX_FILE=p3_pano/fair021_test_idx.txt \
    POSE_OPT=1 POSE_GAUGE=1 POSE_START=500 POSE_LR_T=0.1 POSE_LR_R=1e-3 \
  python p3_pano/train_pano_gsplat_sph.py p3_pano/fair021_d2.json p6_unisharp/ft/runs/gg_p1_021 15000 4096 2>&1 | grep -vE "^W07"
echo "=== interpolation self-check ==="
python p6_unisharp/ft/apply_pose_deltas.py p6_unisharp/ft/runs/gg_p1_021/point_cloud/iteration_15000/poses.npz \
  p3_pano/fair021_d2.json p3_pano/fair021_d2_posefix.json --selfcheck
echo "=== A' vs B on ORIGINAL held-out poses ==="
python p6_unisharp/ft/eval_res.py p3_pano/fair021_d2.json p3_pano/fair021_test_idx.txt 4096 1024 \
  p6_unisharp/ft/runs/gg_p0_021/point_cloud/iteration_15000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_p1_021/point_cloud/iteration_15000/point_cloud.ply
echo "=== B on INTERPOLATED-corrected held-out poses (the legitimate number) ==="
python p6_unisharp/ft/eval_res.py p3_pano/fair021_d2_posefix.json p3_pano/fair021_test_idx.txt 4096 1024 \
  p6_unisharp/ft/runs/gg_p1_021/point_cloud/iteration_15000/point_cloud.ply
echo POSEOPT_B_DONE
