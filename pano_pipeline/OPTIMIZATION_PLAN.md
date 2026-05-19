# 全景无人机 → 3DGS 流水线优化方案

> 基于已有 8 个视频实测数据，分析各阶段瓶颈，给出分级优化方案。

---

## 一、当前各阶段实测耗时

以一个典型场景（5 分钟 8K 全景视频 → 80 张全景帧 → 1120 张透视图 → 30k 迭代训练）为例：

| 阶段 | 操作 | 实测耗时 | 占比 | 资源 |
|---|---|---|---|---|
| **Stage 1** | `.insv` → equirect 全景帧（ffmpeg v360） | **5 分钟** | 15% | HEVC GPU 解码 + **CPU v360 滤镜** |
| **Stage 2** | 全景 → 14 透视图/帧（cv2.remap） | **2 分钟** | 6% | **CPU 单线程** |
| **Stage 3.1** | COLMAP 特征提取（GPU SIFT） | 50 秒 | 2% | GPU ✓ |
| **Stage 3.2** | COLMAP 匹配（sequential GPU） | 2 分钟 | 6% | GPU ✓ |
| **Stage 3.3** | COLMAP Mapper（增量 SfM + BA） | **12 分钟** | 35% | **CPU 单线程** |
| **Stage 4** | 3DGS 训练（30k iter） | **12 分钟** | 35% | GPU ✓（H100） |
| **Stage 5** | WebXR 浏览 | 实时 | — | 客户端 GPU |

**总耗时：34 分钟/场景**。瓶颈集中在 **Stage 1（CPU 滤镜）、Stage 3.3（CPU SfM）、Stage 4（GPU 训练）**。

更大数据（2100 张透视图）时 Mapper 升到 12 分钟，扩展性差（CPU 单线程 BA 是 O(n²)~O(n³)）。

---

## 二、分级优化方案

### P0：低成本高收益（1-2 天）

> 适合先做，几乎不改架构

#### P0.1 跨视频并行（节省 50-70%）

当前所有 7 个视频**串行**处理。每个视频处理时：
- Stage 1: GPU 解码 + CPU 滤镜 → CPU 闲置率 ~50%
- Stage 3.3: CPU 满载 → GPU 闲置
- Stage 4: GPU 满载 → CPU 闲置

**改进**：跑 2-3 个视频并行，让 GPU/CPU 都满载。

```bash
# 同时跑 3 个 (前提：单卡显存充足，H100 80GB 完全够)
process_one_video_gpu.sh video1 &
process_one_video_gpu.sh video2 &
process_one_video_gpu.sh video3 &
wait
```

但要注意：GPU 训练阶段不能 3 个一起跑，必须排队。所以更好的方式是**流水线化**：

```
时间轴 →
视频A: [S1][S2][S3---][S4----]
视频B:      [S1][S2][S3---][S4----]
视频C:           [S1][S2][S3---][S4----]
```

实现：每个 video 用 GNU parallel + sem 控制 GPU 独占阶段排队。

**预期收益**：3 视频并行 → 总耗时降到 1/2.5 ≈ 节省 60%。

#### P0.2 Stage 2 多进程化（节省 80%）

`pano_to_perspective.py` 当前单线程。每张全景拆 14 个 view 互相独立，**完全可并行**。

```python
from joblib import Parallel, delayed
# 替换主循环
Parallel(n_jobs=-1)(delayed(process_panorama)(...) for pano_path in pano_files)
```

CPU 32 核 → 80 张 × 14 view 拆分 = 1120 任务 → 满载用时 ~10-20 秒，比原来 2 分钟快 6-12 倍。

#### P0.3 3DGS 训练迭代数减半（节省 50% Stage 4）

实测 iteration_7000 vs iteration_30000 的 PSNR 差：
- 7000 iter:  PSNR ~20.7
- 30000 iter: PSNR ~25.1

对网页/VR 预览，7000 已经够清晰。建议 **默认训练到 15000 iter**（约 PSNR 23.5），再继续 30000 仅用于最终发布版。

```bash
python train.py -s $SCENE -m $SCENE/output --iterations 15000
```

**预期收益**：训练时间 12min → 6min，节省 50%。

#### P0.4 ffmpeg v360 滤镜替换为 CUDA 实现（节省 60% Stage 1）

ffmpeg 主线的 `v360` 是 CPU 滤镜，主要操作是大量像素级 remap，**非常适合 GPU**。

替代方案：
1. **`v360_cuda` 补丁**（部分 ffmpeg fork 已实现）
2. **kornia + PyTorch** 写一个 GPU 鱼眼→equirect 转换脚本
3. **Insta360 Studio CLI**（官方质量最高，需要 license）

短期建议方案 2：
```python
import torch, kornia
# 用 kornia.geometry.transform.image_transform 在 GPU 上做 remap
# 80 张 8K 全景 GPU 处理约 30 秒（vs CPU 5 分钟）
```

**预期收益**：Stage 1 从 5min → 1min。

---

### P1：中等改造（1 周）

#### P1.1 换 COLMAP Mapper 为 GLOMAP（节省 80% Stage 3.3）

GLOMAP 是 COLMAP 作者新写的**全局 SfM**算法（[github.com/colmap/glomap](https://github.com/colmap/glomap)）：

| | COLMAP Mapper（当前） | GLOMAP |
|---|---|---|
| 算法 | 增量式 SfM（incremental） | 全局 SfM（global） |
| 并行性 | 几乎无 | 高（旋转/位置分开优化） |
| 速度（2000 张图） | 12 min | **1-2 min** |
| 接口 | 兼容 | 接 database.db 兼容 COLMAP |

实施：

```bash
# 步骤替换
# 旧: colmap mapper --database_path ... --image_path ... --output_path ...
# 新:
glomap mapper --database_path ... --image_path ... --output_path ...
```

GLOMAP 已有官方 docker image `colmap/glomap:latest`。

**预期收益**：Stage 3.3 从 12min → 2min，节省 ~10 min。

#### P1.2 用 gsplat 替换 INRIA 训练代码（节省 30-50% Stage 4）

[nerfstudio/gsplat](https://github.com/nerfstudio-project/gsplat) 是更优化的 3DGS 训练库：

- 自定义 CUDA kernel，前向/反向都更快
- 支持稀疏 Adam（已经在用）
- 支持 absgrad 等高级密度控制
- 支持 progressive training

对比 INRIA reference：
- 单步迭代速度：1.5-2x
- 同等 PSNR 下迭代数减半

```python
# train.py 替换为 gsplat-based 训练脚本
pip install gsplat
```

**预期收益**：30k iter 训练时间 12 min → 6-8 min。

#### P1.3 直接用全景图训练（去掉 Stage 2）

[OmniGS](https://arxiv.org/abs/2404.03544) / [PanoGS](https://panogs.github.io/) 等论文，让 3DGS 直接吃 equirectangular 全景：

```
传统:   全景 → 14 透视图 → COLMAP → 3DGS（COLMAP 必须用透视模型）
全景版: 全景 → COLMAP（球面相机模型）→ 全景 3DGS
```

优点：
- 数据量减 14 倍
- 没有透视拆分的 FOV 重叠/接缝问题
- 训练快 5-10 倍

挑战：
- 需要 fork 改 diff-gaussian-rasterization 让它支持 equirect 投影
- COLMAP 球面相机模型不稳定，可能要预提供 pose

**预期收益**：理想情况下整条流水线快 5-10 倍。**但**改造成本高。

#### P1.4 PLY → KSPLAT 转换（VR 加载提速 3 倍）

mkkellogg 的 GaussianSplats3D 支持 `.ksplat`（K-D tree splat）格式，比 PLY 小 2-3 倍、解码快 5 倍：

```bash
# 转换脚本（GaussianSplats3D 提供）
node util/create-ksplat.js \
    --input  point_cloud.ply \
    --output point_cloud.ksplat \
    --compression-level 1
```

对 PICO 4 Ultra 用户：50MB PLY → 20MB ksplat，首次加载从 10 秒 → 3 秒。

---

### P2：架构性升级（1-3 个月）

#### P2.1 用 VGGT / DUSt3R / Fast3R 替换 COLMAP

最新的 deep-learning-based SfM：

| 方法 | 输入 | 输出 | 速度 |
|---|---|---|---|
| **VGGT** (Meta, 2025) | N 张图片 | poses + 稠密点云 + 深度 | **秒级** |
| **DUSt3R** (Naver, 2024) | 2+ 张图 | 点云 + pose | 秒级 |
| **MASt3R** (后续) | 多图 | 全局 SfM | 秒级 |
| **Fast3R** | N 张 | poses + 点云 | 秒级 |

这些模型一次 GPU 前向就能输出所有信息，**比 COLMAP 快 100 倍以上**。

实施路线：
1. 用 VGGT 替换 COLMAP feature_extractor + matcher + mapper
2. 输出 ply 点云 → 用 3DGS 训练
3. 整个流水线 30 min → 5 min

风险：
- 这些模型对训练分布外的数据（如 8K 全景拆出的透视图）泛化可能不如 COLMAP
- 大场景（>500 张图）可能 OOM

#### P2.2 实时流式重建（在飞行中重建）

参考 **MASt3R-SLAM**：边飞边重建，无人机降落时已有粗模型。

```
无人机 → WiFi 实时回传视频 → 机载/边缘端实时 SLAM →
即时 3DGS 训练 → 边飞边出预览
```

适合**主动飞行规划**：边重建边判断是否需要补拍某些视角。

#### P2.3 Web 端 GPU 训练（在浏览器训练）

[gsplat WebGPU](https://github.com/playcanvas/playcanvas-engine) 等工具让 3DGS 训练能在浏览器跑：

- 用户上传视频 → 浏览器直接训练
- 不需要后端 GPU 服务器
- 但目前性能 ~10x 慢于原生 CUDA

不适合生产，但适合 demo / 教学。

#### P2.4 LOD 分块 + 渐进加载

对超大场景（城市级）：
- 把场景按空间分块（八叉树）
- 远处的块用稀疏点云（10% 高斯）
- 近处的块用全密度

PICO 4 Ultra 加载 1GB 场景 → 5 秒；分块后 100MB 立即可见，背景流式加载。

---

## 三、推荐落地路线

如果只能做一件事，按 **ROI 排序**：

| 优先级 | 措施 | 工作量 | 预期收益 |
|---|---|---|---|
| 1 | **P0.2** Stage 2 多进程 | 30 分钟 | 节省 100 秒/场景 |
| 2 | **P0.3** 训练 15k 迭代 | 5 分钟 | 节省 6 分钟/场景 |
| 3 | **P0.4** ffmpeg v360 GPU 替换 | 4 小时 | 节省 4 分钟/场景 |
| 4 | **P1.1** 切换 GLOMAP | 半天 | 节省 10 分钟/场景 |
| 5 | **P0.1** 跨视频并行 | 1 天 | 7 视频从 4 小时降到 1.5 小时 |
| 6 | **P1.2** gsplat 训练库 | 1 天 | 节省 5 分钟/场景 |
| 7 | **P1.4** PLY → ksplat | 半天 | VR 加载快 3 倍 |

**做完 P0 + P1 后，单场景流水线从 34 分钟降到 ~6-8 分钟**（4-5 倍加速）。

P2 是研究型项目，留给将来。

---

## 四、并行/资源调度示意

```
                    H100 GPU 80GB
   ┌─────────────────────────────────────────────────┐
   │                                                  │
   │  [ COLMAP Mapper 占 5GB ] + [ 3DGS 训练占 30GB ] │
   │  ↑ 一个场景的 Stage 3.3                         │
   │  ↑ 另一个场景的 Stage 4                         │
   │                                                  │
   │  剩余 45GB 可再跑 1 个 Stage 4                  │
   │                                                  │
   └─────────────────────────────────────────────────┘

CPU 32 cores
   ┌─────────────────────────────────────────────────┐
   │ S1×3 (ffmpeg×3) + S2×3 (joblib×n) + S3.3 BA     │
   └─────────────────────────────────────────────────┘
```

合理调度后，单台服务器 7 个视频从串行 4 小时降到并行 1 小时左右。

---

## 五、监控与可观测性

加入：
1. **Per-stage timing** 写入 JSON 文件
2. **Prometheus + Grafana** 监控
   - GPU 利用率、显存
   - CPU 各核占用
   - 磁盘 I/O
3. **每个场景生成 timing 报告**

```json
{
  "scene": "scene_021",
  "stages": {
    "extract_panoramas": 180.5,
    "split_perspectives": 120.3,
    "colmap_feature": 50.0,
    "colmap_match": 128.7,
    "colmap_mapper": 720.0,
    "train_3dgs": 720.0
  },
  "total": 1919.5
}
```

用于回归测试和优化效果对比。

---

## 六、验收标准

完成 P0 + P1 后的验收：

| 指标 | 当前 | 目标 |
|---|---|---|
| 单场景流水线时长 | 34 min | **≤ 8 min** |
| 7 视频批量时长 | 4 h | **≤ 1.5 h** |
| 训练后 PSNR (15k iter) | — | **≥ 23.5** |
| PLY/ksplat 加载到首屏 | 8-10 s | **≤ 3 s** |
| GPU 平均利用率 | ~30% | **≥ 60%** |
| 失败率（COLMAP fail） | 3/7 | **≤ 1/7**（飞行规划优化后） |
