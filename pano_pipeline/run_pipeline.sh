#!/bin/bash
#
# 全景无人机 → 3DGS 一键流水线
#
# 使用 nvcr.io/nvidia/pytorch:24.12-py3 容器运行全部步骤。
# COLMAP 在单独的容器中运行。
#
# 用法:
#   ./run_pipeline.sh /path/to/panoramas [scene_name]
#
# 目录结构 (自动创建):
#   scenes/<scene_name>/
#   ├── panoramas/          ← 你的全景图 (软链接)
#   ├── input/              ← Stage 1 拆分出的透视图
#   ├── pano_metadata.json  ← 拆分元数据
#   ├── geo.txt             ← GPS 注册文件 (如有)
#   ├── distorted/          ← COLMAP 中间产物
#   ├── sparse/0/           ← COLMAP 输出
#   ├── images/             ← 去畸变后的图像 (3DGS 使用)
#   └── output/             ← 3DGS 训练结果
#

set -e

# ── 参数 ────────────────────────────────────────────────────────

PANO_DIR="$1"
SCENE_NAME="${2:-$(basename "$PANO_DIR")}"
WORK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCENE_DIR="${WORK_ROOT}/scenes/${SCENE_NAME}"
DOCKER_IMAGE="nvcr.io/nvidia/pytorch:24.12-py3"
COLMAP_IMAGE="colmap/colmap:latest"

# 拆分参数
FOV=90
IMG_SIZE=2048
PRESET="standard"    # cubemap / standard / dense

# 训练参数
ITERATIONS=30000

# ── 颜色输出 ────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*"; exit 1; }

# ── 检查参数 ────────────────────────────────────────────────────

if [ -z "$PANO_DIR" ]; then
    echo "用法: $0 <全景图目录> [场景名称]"
    echo ""
    echo "参数说明:"
    echo "  全景图目录   包含 equirectangular 全景图 (8K JPG) 的目录"
    echo "  场景名称     可选, 默认使用目录名"
    echo ""
    echo "示例:"
    echo "  $0 /data/drone_flight/panoramas my_building"
    echo "  $0 /data/drone_flight/panoramas"
    echo ""
    echo "环境变量:"
    echo "  FOV=$FOV              透视图视场角"
    echo "  IMG_SIZE=$IMG_SIZE    透视图边长"
    echo "  PRESET=$PRESET        视角预设 (cubemap/standard/dense)"
    echo "  ITERATIONS=$ITERATIONS  训练迭代次数"
    exit 1
fi

PANO_DIR="$(cd "$PANO_DIR" && pwd)"
[ -d "$PANO_DIR" ] || err "全景图目录不存在: $PANO_DIR"

PANO_COUNT=$(find "$PANO_DIR" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.tif" -o -iname "*.tiff" \) | wc -l)
[ "$PANO_COUNT" -gt 0 ] || err "在 $PANO_DIR 中未找到图片文件"

# ── 准备目录 ────────────────────────────────────────────────────

log "场景名称: ${CYAN}${SCENE_NAME}${NC}"
log "全景图目录: $PANO_DIR ($PANO_COUNT 张)"
log "工作目录: $SCENE_DIR"

mkdir -p "$SCENE_DIR"

# ── Stage 1: 全景图 → 透视图 ────────────────────────────────────

log ""
log "═══════════════════════════════════════════════════"
log " Stage 1: 全景图 → 透视图"
log "═══════════════════════════════════════════════════"
log "FOV=$FOV, SIZE=$IMG_SIZE, PRESET=$PRESET"

if [ -d "$SCENE_DIR/input" ] && [ "$(ls "$SCENE_DIR/input"/*.jpg 2>/dev/null | wc -l)" -gt 0 ]; then
    warn "input/ 目录已存在且非空, 跳过拆分 (删除 input/ 可重新运行)"
else
    docker run --rm \
        -v "$WORK_ROOT":/workspace/gaussian-splatting \
        -v "$PANO_DIR":/workspace/panoramas:ro \
        -v "$SCENE_DIR":/workspace/scene \
        "$DOCKER_IMAGE" \
        bash -c "
            pip install -q piexif opencv-python 2>/dev/null
            cd /workspace/gaussian-splatting
            python pano_pipeline/pano_to_perspective.py \
                -i /workspace/panoramas \
                -o /workspace/scene/input \
                --fov $FOV --size $IMG_SIZE --preset $PRESET
        " 2>&1 | grep -v "^DEPRECATION\|^WARNING.*pip\|^\[notice\]"

    PERSP_COUNT=$(ls "$SCENE_DIR/input"/*.jpg 2>/dev/null | wc -l)
    log "拆分完成: $PERSP_COUNT 张透视图"
fi

# ── Stage 2: COLMAP SfM ─────────────────────────────────────────

log ""
log "═══════════════════════════════════════════════════"
log " Stage 2: COLMAP (特征提取 → 匹配 → 稀疏重建)"
log "═══════════════════════════════════════════════════"

if [ -d "$SCENE_DIR/sparse/0" ] && [ -f "$SCENE_DIR/sparse/0/cameras.bin" ]; then
    warn "sparse/0/ 已存在, 跳过 COLMAP (删除 sparse/ 和 distorted/ 可重新运行)"
else
    # 检查 COLMAP 镜像是否可用, 不可用则用 pytorch 容器里装
    if docker image inspect "$COLMAP_IMAGE" > /dev/null 2>&1; then
        log "使用 COLMAP Docker 镜像"
        COLMAP_CMD="docker run --rm --gpus all \
            -v $SCENE_DIR:/workspace/scene \
            $COLMAP_IMAGE"
    else
        log "COLMAP 镜像不可用, 将在 PyTorch 容器中安装 COLMAP"
        COLMAP_CMD="docker run --rm --gpus all --ipc=host \
            -v $SCENE_DIR:/workspace/scene \
            $DOCKER_IMAGE bash -c 'apt-get update -qq && apt-get install -y -qq colmap 2>/dev/null &&"
    fi

    # 在 pytorch 容器中运行 COLMAP (更可靠)
    docker run --rm --gpus all --ipc=host \
        -v "$SCENE_DIR":/workspace/scene \
        -v "$WORK_ROOT":/workspace/gaussian-splatting \
        "$DOCKER_IMAGE" \
        bash -c "
            set -e
            apt-get update -qq > /dev/null 2>&1
            apt-get install -y -qq colmap > /dev/null 2>&1

            SCENE=/workspace/scene
            mkdir -p \$SCENE/distorted/sparse

            echo '>> 特征提取...'
            colmap feature_extractor \
                --database_path \$SCENE/distorted/database.db \
                --image_path \$SCENE/input \
                --ImageReader.single_camera 1 \
                --ImageReader.camera_model PINHOLE \
                --SiftExtraction.use_gpu 1

            echo '>> 特征匹配...'
            # 如果有 GPS, 使用 spatial_matcher 加速; 否则用 exhaustive_matcher
            if [ -f \$SCENE/geo.txt ]; then
                echo '   (检测到 GPS, 使用 spatial matching)'
                colmap spatial_matcher \
                    --database_path \$SCENE/distorted/database.db \
                    --SiftMatching.use_gpu 1 \
                    --SpatialMatching.is_gps 1 \
                    --SpatialMatching.max_num_neighbors 50
            else
                echo '   (无 GPS, 使用 exhaustive matching)'
                colmap exhaustive_matcher \
                    --database_path \$SCENE/distorted/database.db \
                    --SiftMatching.use_gpu 1
            fi

            echo '>> 稀疏重建 (Bundle Adjustment)...'
            colmap mapper \
                --database_path \$SCENE/distorted/database.db \
                --image_path \$SCENE/input \
                --output_path \$SCENE/distorted/sparse \
                --Mapper.ba_global_function_tolerance 0.000001

            # 如果有 GPS, 进行 geo-registration 对齐坐标系
            if [ -f \$SCENE/geo.txt ]; then
                echo '>> GPS 坐标对齐...'
                colmap model_aligner \
                    --input_path \$SCENE/distorted/sparse/0 \
                    --output_path \$SCENE/distorted/sparse/0 \
                    --ref_images_path \$SCENE/geo.txt \
                    --alignment_type ecef \
                    --robust_alignment 1 \
                    --robust_alignment_max_error 3.0
            fi

            echo '>> 图像去畸变...'
            colmap image_undistorter \
                --image_path \$SCENE/input \
                --input_path \$SCENE/distorted/sparse/0 \
                --output_path \$SCENE \
                --output_type COLMAP

            # 整理 sparse 目录结构
            mkdir -p \$SCENE/sparse/0
            if [ -d \$SCENE/sparse ] && [ ! -f \$SCENE/sparse/0/cameras.bin ]; then
                for f in \$SCENE/sparse/*; do
                    [ \"\$(basename \$f)\" = '0' ] && continue
                    mv \$f \$SCENE/sparse/0/ 2>/dev/null || true
                done
            fi

            echo '>> COLMAP 完成!'
            colmap model_analyzer --path \$SCENE/sparse/0
        " 2>&1 | grep -v "^$\|^Preparing\|^Get:\|^Hit:\|^Reading"

    log "COLMAP 完成"
fi

# ── Stage 3: 3DGS 训练 ──────────────────────────────────────────

log ""
log "═══════════════════════════════════════════════════"
log " Stage 3: 3D Gaussian Splatting 训练"
log "═══════════════════════════════════════════════════"
log "迭代次数: $ITERATIONS"

if [ -f "$SCENE_DIR/output/point_cloud/iteration_${ITERATIONS}/point_cloud.ply" ]; then
    warn "训练结果已存在, 跳过 (删除 output/ 可重新运行)"
else
    docker run --rm --gpus all --ipc=host \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v "$WORK_ROOT":/workspace/gaussian-splatting \
        -v "$SCENE_DIR":/workspace/scene \
        "$DOCKER_IMAGE" \
        bash -c "
            set -e
            cd /workspace/gaussian-splatting
            pip install -q --no-deps plyfile opencv-python joblib \
                submodules/diff-gaussian-rasterization \
                submodules/simple-knn \
                submodules/fused-ssim 2>/dev/null

            python train.py \
                -s /workspace/scene \
                -m /workspace/scene/output \
                --iterations $ITERATIONS
        " 2>&1 | grep -v "^DEPRECATION\|^WARNING.*pip\|^\[notice\]"

    log "训练完成!"
fi

# ── Stage 4: 渲染验证 ───────────────────────────────────────────

log ""
log "═══════════════════════════════════════════════════"
log " Stage 4: 渲染验证图"
log "═══════════════════════════════════════════════════"

docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$WORK_ROOT":/workspace/gaussian-splatting \
    -v "$SCENE_DIR":/workspace/scene \
    "$DOCKER_IMAGE" \
    bash -c "
        set -e
        cd /workspace/gaussian-splatting
        pip install -q --no-deps plyfile opencv-python joblib \
            submodules/diff-gaussian-rasterization \
            submodules/simple-knn \
            submodules/fused-ssim 2>/dev/null

        python render.py -m /workspace/scene/output --skip_test
    " 2>&1 | grep -v "^DEPRECATION\|^WARNING.*pip\|^\[notice\]"

log "渲染完成!"

# ── 汇总 ────────────────────────────────────────────────────────

log ""
log "═══════════════════════════════════════════════════"
log " 全部完成!"
log "═══════════════════════════════════════════════════"

PLY_PATH="$SCENE_DIR/output/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
PLY_SIZE=$(du -sh "$PLY_PATH" 2>/dev/null | cut -f1)
TOTAL_SIZE=$(du -sh "$SCENE_DIR/output" 2>/dev/null | cut -f1)

log ""
log "场景: ${CYAN}${SCENE_NAME}${NC}"
log "点云: $PLY_PATH ($PLY_SIZE)"
log "输出: $SCENE_DIR/output/ ($TOTAL_SIZE)"
log ""
log "查看结果:"
log "  SuperSplat: 打开浏览器拖入 .ply 文件"
log "  渲染图片:   $SCENE_DIR/output/train/ours_${ITERATIONS}/renders/"
log ""
