set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
cd /raid/git/gaussian-splatting
cd /raid/git/gaussian-splatting
# 30k beat 7k on 3/3 scenes and 021 was still climbing, so probe one length further.
# Caveat carried forward: length also stretches the depth anneal, so this compares
# "longer run with its matching schedule", not iterations in isolation.
env INIT_PLY=p6_unisharp/ft/runs/gg_F_021/point_cloud/iteration_30000/point_cloud.ply \
    SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 REFINE_STOP_FRAC=0 RESET_EVERY=999999 \
    DEPTH_DIR=p6_unisharp/ft/depth021 DEPTH_NORM=0.25 TEST_IDX_FILE=p3_pano/fair021_test_idx.txt \
  python p3_pano/train_pano_gsplat_sph.py p3_pano/fair021_d2.json p6_unisharp/ft/runs/gg_l45021 45000 4096 2>&1 | grep -E "^\[depth\]|EVAL|Traceback|Error"
echo "=== 021: 7k vs 30k vs 45k ==="
python p6_unisharp/ft/eval_res.py p3_pano/fair021_d2.json p3_pano/fair021_test_idx.txt 4096 1024 \
  p6_unisharp/ft/runs/gg_dn021/point_cloud/iteration_7000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_len021/point_cloud/iteration_30000/point_cloud.ply \
  p6_unisharp/ft/runs/gg_l45021/point_cloud/iteration_45000/point_cloud.ply
echo LEN45_DONE
