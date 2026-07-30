set -x
export PATH=/raid/git/gaussian-splatting/p6_unisharp/venv/bin:$PATH; export VIRTUAL_ENV=/raid/git/gaussian-splatting/p6_unisharp/venv; export PYTHONPATH=/raid/git/gaussian-splatting/p3_pano/gsplat; export NVCC_PREPEND_FLAGS="-I/raid/git/gaussian-splatting/p6_unisharp/venv/lib/python3.12/site-packages/nvidia/cuda_cccl/include"
export HF_HOME=/raid/git/gaussian-splatting/p6_unisharp/.hf
export PYTHONPATH=$PYTHONPATH:/raid/git/gaussian-splatting/p6_unisharp/UniSHARP/UniK3D
cd /raid/git/gaussian-splatting
python p6_unisharp/ft/gen_depth.py p3_pano/fair027_d2.json p6_unisharp/ft/depth027 1024 2>&1 | grep -E "^\[|^  \["
echo DEPTHGEN_DONE
