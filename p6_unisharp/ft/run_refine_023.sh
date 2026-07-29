set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
cd /raid/git/gaussian-splatting
C=p6_unisharp/ft/runs/gg_F/point_cloud/iteration_30000/point_cloud.ply
env INIT_PLY=$C SKY_FREEZE_R=2.0 LR_SCALE_POS=0.4 ABSGRAD=1 REFINE_STOP_FRAC=0 RESET_EVERY=999999 \
    TEST_IDX_FILE=p3_pano/fair023_test_idx.txt \
  python p3_pano/train_pano_gsplat_sph.py p3_pano/fair023_d2.json p6_unisharp/ft/runs/gg_ng023 7000 4096 2>&1 | grep -E "resume|DONE|EVAL|Traceback|Error"
python p6_unisharp/ft/eval_res.py p3_pano/fair023_d2.json p3_pano/fair023_test_idx.txt 4096 1024 \
  $C p6_unisharp/ft/runs/gg_ng023/point_cloud/iteration_7000/point_cloud.ply
echo C23_DONE
