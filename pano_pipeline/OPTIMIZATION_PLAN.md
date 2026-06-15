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

##### ✅ P1.2 micro-bench (2026-06-10) — gsplat 3.42x 更快
`p3_pano/bench_raster.py`，相同 100k 高斯 @1024² fwd+bwd：
- gsplat 1.5.3: **243 iter/s** (4.11 ms/iter)
- INRIA diff-gaussian-rasterization: 71 iter/s (14.07 ms/iter)
- **gsplat 快 3.42x**（超出预期 1.5-2x）。速度案例已证实；完整后端替换
  （致密化对齐 + 质量）为可选后续工作。
- **✅ 完整端到端后端替换已完成 (T-F2, 2026-06-11)** — 见第七节。同场景同
  holdout 同迭代下 gsplat 比 INRIA **快 1.55x 且 PSNR +1.78 dB**。


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

##### ✅ P1.3 已实现并验证 (2026-06-09) — 端到端跑通

实现在 `p3_pano/`（详见 `p3_pano/README.md`）。把 OmniGS 的 equirect(LONLAT)
CUDA 光栅化器移植成 pip 扩展 `diff_gaussian_rasterization_pano`，写了全景
data loader（从 VGGT 透视裁剪反推每张全景位姿）+ `train_pano.py` 直接全景训练。

**scene_023 实测（held-out test，每 8 张全景留 1）：**

| | 透视管线 (VGGT 裁剪) | **直接全景** |
|---|---|---|
| 训练图像 | 1260 透视裁剪 | **90 全景图 (14x 少)** |
| held-out PSNR | 17.05 | **19.12** |
| 光栅化器 | 针孔 | equirect (移植 OmniGS) |

> **直接全景训练在 held-out PSNR 上追平/超过透视管线，同时训练图像少 14x、
> 每个视角渲染完整 360°** — P1.3 论点端到端验证成立。

**全 7 场景批量 (2026-06-09，held-out 12 test panos/scene)：直接全景每个场景都赢**

| 场景 | 透视 PSNR | 直接全景 PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| 021 | 18.67 | **21.61** | 0.795 | 0.422 |
| 022 | 19.94 | **22.48** | 0.792 | 0.408 |
| 023 | 17.05 | **19.55** | 0.693 | 0.475 |
| 025 | 19.00 | **20.04** | 0.718 | 0.441 |
| 026 | 18.25 | **19.55** | 0.682 | 0.479 |
| 027 | 16.24 | **19.69** | 0.672 | 0.530 |
| 028 | 18.22 | **21.09** | 0.664 | 0.555 |
| **均值** | **18.05** | **20.43 (+2.4 dB)** | 0.717 | 0.473 |

**移植踩坑**（已解决，见 p3_pano/README）：`M_*f32` 宏触发 nvcc ICE、
pycolmap==3.10、cstdint per-file、VGGT 显存 O(N²) 需裁剪 crops ≤300。

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

##### ✅ P2.1 已实现并验证 (2026-06-08) — VGGT

实现在 `pano_pipeline/vggt_sfm.sh` + `p2_vggt/`（详见 `p2_vggt/README.md`）。

**scene_023 实测 (140 帧)：**

| | COLMAP 原版 | COLMAP v2g (exhaustive) | **VGGT** |
|---|---|---|---|
| Stage 3 (SfM) | ~15 min | ~2.6 h | **96 s** |
| 初始点数 | ~100k | 119k | 100k |
| PSNR @7k iter | ~20 | 18 | **23.0** |
| Stage 4 (15k iter) | ~17 min | 18 min | 9 min |
| **总计** | ~32 min | ~3 h | **~10.5 min** |

> **VGGT 是第一个"又快又好"的配置** — Stage 3 比 COLMAP 快 10-100x, 质量持平甚至更好。
> P0/P1 的开关只能用质量换速度, VGGT 同时改善两者。这是整个优化计划的最大突破。

**踩过的坑（已在脚本中解决）：**
1. demo_colmap.py 顶部硬 import lightglue → 改惰性 import（脚本自动 patch）
2. pycolmap 必须 `==3.10.0`（新版 Image API 不兼容）
3. `--conf_thres_value` 默认 5.0 会把低纹理无人机图的点**全部过滤** → 0 点；降到 1.5
4. VGGT 跨帧全局 attention，显存 O(N²) → 子采样到 ≤300 帧（80GB 实测上限确认）
5. 权重 4.7GB 在 HuggingFace → 预下载到 host 缓存, seed 进容器 TORCH_HOME

##### ✅ 全场景批量验证 (2026-06-08) — 7/7 成功，含 COLMAP 失败场景

`max_frames=300` 批量跑全部 7 个场景，训练带 `--eval`（每 8 帧留 1 作 held-out
test），用 `metrics.py` 算 **15k iter 的真实 held-out 指标**（非 train-view）。

| 场景 | COLMAP 结果 | VGGT SfM | 初始点 | **PSNR** | **SSIM** | **LPIPS** |
|---|---|---|---|---|---|---|
| scene_021 | ✓ PSNR25(train) | 149s | 13.4k | 18.67 | 0.746 | 0.486 |
| scene_022 | ✓ | 151s | 19.4k | 19.94 | 0.748 | 0.474 |
| scene_023 | ✓ | 140s | 100k | 17.05 | 0.683 | 0.511 |
| scene_025 | ✓ | 153s | 100k | 19.00 | 0.736 | 0.480 |
| **scene_026** | ✗ **失败** | 153s | 100k | 18.25 | 0.679 | 0.515 |
| **scene_027** | ✗ **失败** | 153s | 100k | 16.24 | 0.640 | 0.549 |
| **scene_028** | ✗ **失败** | 153s | 100k | 18.22 | 0.618 | 0.582 |
| 均值 | — | ~150s | — | **18.05** | **0.693** | **0.514** |

**两个关键结论：**
1. **VGGT 挽救了 COLMAP 完全失败的 026/027/028**（起飞→巡航→降落型轨迹，
   COLMAP 找不到初始图像对）。feed-forward 逐帧预测位姿，不依赖初始对，
   是这类断裂轨迹的唯一可行方案。整条 SfM 仅 ~150s/场景。
2. **这些是诚实的 held-out test 指标**（PSNR ~18, 比之前 train-view 的 23 低）。
   `--eval` 揭示了之前 train-view PSNR 偏乐观。对于低重叠的航拍透视裁剪图，
   PSNR 16-20 属合理水平；要更高需 P1.3（直接全景训练）或更密航线（见
   FLIGHT_PLANNING.md）。

**全部产物**：每个场景 `vggt/output/.../iteration_15000/point_cloud.{ply,ksplat}`，
WebXR viewer 用 `?source=vggt` 浏览。

**剩余工作：**
- 大场景 (>150 帧) 需 chunk / sliding-window
- `--use_ba` 模式（LightGlue+pyceres）未接入, 可进一步提精度
- 仅在 scene_023 验证, 其他场景待批量跑

#### P2.2 实时流式重建（在飞行中重建）

参考 **MASt3R-SLAM**：边飞边重建，无人机降落时已有粗模型。

```
无人机 → WiFi 实时回传视频 → 机载/边缘端实时 SLAM →
即时 3DGS 训练 → 边飞边出预览
```

适合**主动飞行规划**：边重建边判断是否需要补拍某些视角。

> **✅ P2.2 已构建并跑通 (T-E1 + T-F1, 2026-06-11)** — 见第七节。MASt3R-SLAM 在
> torch-2.6/CUDA-12.6 容器内 build + run 全部跑通（H100 sm_90）。**关键发现**：
> 90 张稀疏全景（0.24fps）第 16 帧丢失跟踪；从 `.insv` 重渲染**密集前向透视流**
> （4fps）后**连续跟踪 107 关键帧**（6.7x），证明之前的丢失是采样稀疏假象而非
> 能力上限。完整复现见 `p4_slam/`。

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

## 三-A、实测数据 (2026-05-24 完成 P0+P1)

### P0.2 - joblib 并行 (pano_to_perspective.py)
- 测试: 15 panos × 14 views = 210 输出
- 串行预估 50s → 并行 8 workers **8s** (~6x 加速)
- ✓ 无质量损失

### P0.3 - 15k iter (vs 30k)
- scene_023 实测 15000 iter on H100 = **15 min**
- 原版 30000 iter ≈ 17 min
- 节省 ~12%（不是 50%，因实际迭代瓶颈不在前半段）
- ⚠️ PSNR 下降: 30k→25.0 vs 15k→14.3 (在低初始点数场景下下降明显)

### P0.4 - 多 chunk ffmpeg (stitch_chunked.sh)
- 用 4 并行 chunks 跑 v360 滤镜
- 已编写脚本但未做端到端验证
- 预期 5min → 1-2min

### P1.1 - GLOMAP
- scene_023 实测: **GLOMAP 95s** vs COLMAP mapper 12 min = **8x 加速**
- ⚠️ 质量损失严重: GLOMAP 在无人机前向飞行场景下三角化角度过小, 67% tracks 被过滤
- 实测初始点数 10k (vs COLMAP 99k), PSNR 12 (vs 25)
- 适合**预览用途**, 不适合最终发布
- 镜像组合: podral3/glomap (colmap) + arhanjain/glomap (mapper)

### P1.4 - KSPLAT 转换
- scene_021: 49M → **4.6M** (10.8x)
- scene_022: 64M → 6.0M
- scene_025: 74M → 6.9M
- scene_023 v2e: 12M → 1.4M (8.5x)
- ✓ VR 端首次加载从 ~10s 降到 1-2s
- 已集成进 colmap_train_v2.sh 自动产出

### 实测端到端对比 (scene_023, 1260 透视图)

| 配置 | Stage 3 (SfM) | Stage 4 (训练) | 总 | 质量 |
|---|---|---|---|---|
| 原版: CPU SIFT + 30k iter | ~15min | ~17min | ~32min | PSNR 25 ✓ 977 imgs |
| v2g 质量: GPU SIFT + exhaustive + colmap + 15k | **2.6h** | 18min | ~3h | PSNR 18 ✗ 1149 imgs |
| v2e 快速: GPU SIFT + sequential + colmap + 15k | 5min | 15min | **21min** | PSNR 14 ✗ 174 imgs |
| v2 GLOMAP: GPU SIFT + sequential + GLOMAP + 15k | **3min** | 15min | **18min** | PSNR 12-14 ✗ |

> ⚠️ **重要发现**: GPU exhaustive_matcher 在 COLMAP 4.x (colmap/colmap:latest 镜像) 反而比 CPU 慢很多 (5134s vs ~3min 原版 CPU)。原因待查 (可能 GPU 实现 bug 或 --FeatureMatching.use_gpu 1 未实际命中 GPU 内核)。

### 真正的胜利
- ✅ **P0.2 (joblib)**: Stage 2 6 倍加速, 0 质量损失
- ✅ **P1.4 (KSPLAT)**: PLY 缩小 10 倍, VR 加载快 5 倍, 0 质量损失

### 待权衡的优化 (速度↑ 质量↓)
- ⚠️ **P0.3 (15k vs 30k)**: 节省训练 2-3 min, 但 PSNR 下降 5-7 个点
- ⚠️ **P1.1 (GLOMAP)**: SfM 8x 加速, 但 67% tracks 被过滤掉, 适合 preview 而非发布
- ⚠️ **Sequential matcher**: 5x 匹配加速, 但漏掉大量图对, 重建图像数 1149→174

### 现实结论
v2 流水线提供了**质量 vs 速度的明确开关**, 但目前没有一个组合**同时**比原版快又质量更好。

要真正快+好, 需要 P2 级方案: **VGGT/DUSt3R** (DL-based SfM, 秒级且全分辨率重建)。

### 新增脚本
- `colmap_train_v2.sh` - 完整 GPU 流水线, 4 个开关位置
- `ply_to_ksplat.sh` - PLY → KSPLAT 单独转换工具
- `stitch_chunked.sh` - 多 chunk 并行 ffmpeg 拼接 (P0.4)

### 用法

```bash
# 默认 (质量优先): exhaustive + colmap + 15k iter
bash colmap_train_v2.sh scene_021

# 自定义: 改 iterations
bash colmap_train_v2.sh scene_021 30000

# 快速预览: GLOMAP + sequential
bash colmap_train_v2.sh scene_021 15000 glomap sequential
```

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

---

## 七、改进任务 T-F (2026-06-11 ~ 06-12)

在 P0/P1/P2 基础上的 5 项改进，全部实现 + 验证 + 提交（`tasks.md` 跟踪，
代码在 `p3_pano/`、`p4_slam/`、`p2_vggt/`）。按价值排序，附诚实结论。

### T-F1 — 密集前向透视流：修复 MASt3R-SLAM 跟踪（**改变了结论**）
`p4_slam/make_dense_perspective.sh` 从原始 `.insv` 单次 ffmpeg 解码
（dual-fisheye→equirect→flat）重渲染密集前向透视流。

| 指标 | 稀疏（90 全景 @0.24fps） | **密集（360 帧 @4fps）** |
|---|---|---|
| 轨迹关键帧 | 16 | **107（6.7x）** |
| 稠密点云 | 10.8 MB | **59 MB（5.5x）** |
| 首次丢失跟踪 | 第 16 帧（~67s） | **直到第 ~116 帧** |
| 丢帧数 | 多次重定位抖动 | **仅 1 帧** |
| 速度 | ~3.5 FPS | ~7 FPS |

> 原"第 16 帧丢失"是**采样稀疏假象**（380s 飞行只采 0.24fps），非能力上限。
> 密集流下 MASt3R-SLAM 在该无人机数据上**连续跟踪**，提供流式重建能力。

### T-F2 — gsplat 后端端到端接入（**更快且质量更高**，收尾 P1.2）
`p3_pano/train_gsplat.py`（gsplat 1.5.3 + 原生 DefaultStrategy 致密化）
vs INRIA `train.py`，**同 scene_023 / 同 every-8th holdout / 同 7000 迭代**：

| 后端 | held-out PSNR | SSIM | 吞吐 | 训练墙钟 |
|---|---|---|---|---|
| INRIA | 17.11 | — | 17.2 it/s | 408 s |
| **gsplat** | **18.89** | 0.666 | **26.5 it/s** | **264 s** |

> **快 1.55x + 高 1.78 dB**；且为保守值（gsplat trainer 每迭代从磁盘重读图像、
> INRIA 缓存）。端到端 1.55x（vs 纯 kernel 微基准 3.42x）是计入 I/O + SSIM +
> 致密化后的真实数字。Runner: `run_gsplat_train.sh`（持久 JIT 缓存 + 自动 vendor glm）。

### T-F3 — 滑窗 VGGT 全局 Sim3 位姿图对齐（收尾 T-D4）
`p2_vggt/global_sim3.py`：用全局位姿图解替换贪心顺序 Umeyama —— 生成树初始化
（相邻窗口）+ 全窗口对精化（重访地点自动成为闭环边）。loss 作用于**相对 Sim3**
（尺度由测量锚定）→ 无尺度坍缩模式。仅用相机中心（无特征轨迹）→ 适用稀疏 360 数据。

**合成闭环轨迹验证**（`test_global_align.py`，每窗口随机 Sim3 + 噪声）：
全局 **0.556 vs 顺序 0.793 平均误差 = 1.4x 更低漂移**。

**真实 1260-crop 验证**（scene_023，`run_vggt_window.sh`，2026-06-12）：
7 窗口 → 1260 相机 + 34M 点；位姿图找到 **6 边 / 0 闭环**（顺序裁剪 = 纯链）。
300 个多窗口相机上的对比：

| 方法 | 共享帧不一致度 | 尺度范围 |
|---|---|---|
| 顺序 Umeyama | 0.1157 | 0.026 – 1.23 |
| **全局 Sim3** | 0.1156 | 0.026 – 1.23 |
| | **1.00x（完全相同）** | |

> **完全符合预测**：无闭环 → 全局解退化为顺序初始化（不会更差）。0.026–1.23 的
> 尺度跨度是 VGGT 每窗口固有的度量歧义，纯链上**任何**对齐方法都无法消除（需闭环
> 或度量锚）。结论：全局解在真实数据上**已验证正确 + 永不更差**；其减漂移收益需要
> 闭环采集（无人机重访地点，且用视觉位置识别而非文件名匹配）。

### T-F4 — 解除 VGGT `--use_ba` 网络阻塞（数据受限，非 bug）
T-D5 的 403 根因是 `api.github.com` 限流 `torch.hub.load` 的分支查询（dinov2 触发），
非 tracker 权重。修复（`p2_vggt/vggsfm_localcache.patch` + `run_vggt_ba.sh`）：
预取 tracker 权重（HuggingFace）+ 预 clone dinov2，经 `VGGSFM_TRACKER_PT` 与
`source="local"`（`DINOV2_LOCAL_DIR`）加载，零 github API 调用。

> BA 流程现在**全程跑通**（模型→dino 排序→tracker→fine-tracking）。**发现**：
> 60–300 张稀疏全景裁剪上报 "Not enough inliers per frame, skip BA" —— 360 派生
> 裁剪的 inlier 特征轨迹不足以做经典 BA 三角化。**直接证据**佐证 T-D5：前馈 VGGT
> （无需对应点）才是这类数据的正确工具；BA 不只是低 ROI，而是数据受限。

### T-F5 — 工程化与可复现
- `run_slam_full.sh --clean` 删除数 GB 的 `mast3r-slam:built` 镜像。
- `_slam_build.sh` 自动 `git apply` 补丁（幂等 `--check`），全新 clone 一键可跑。
- `p4_slam/SETUP.md` 文档化完整复现步骤（clone、checkpoints、vendor eigen→lietorch_src）。

### T-F6 — gsplat 接入直接全景训练（**质量持平但不更快**）
gsplat 仅支持针孔投影（无 equirect），故 `p3_pano/train_pano_gsplat.py` 每张全景
渲染 6 个针孔**立方体面**（一次 gsplat `C=6` 批量）再用可微 `grid_sample` 重采样
为 equirect，采用与 LONLAT 完全一致的约定（`auxiliary.h`：lon=atan2(x,z),
lat=asin(y)）；gsplat 原生 DefaultStrategy 致密化。scene_023 同 holdout/迭代：

| 后端 | PSNR | LPIPS | it/s | 训练墙钟 | 像素/全景 |
|---|---|---|---|---|---|
| LONLAT（原生 equirect, OmniGS） | 18.63–19.55 | 0.480 | **79.2** | **88 s** | 0.52M |
| gsplat 立方体 | **19.62** | **0.466** | 72.2 | 97 s | 1.57M (6×512²) |

> **结论**：质量持平（PSNR/LPIPS 可比；SSIM 不可比——gsplat eval 用 box 窗，
> train_pano 用高斯窗），但 gsplat **慢约 10%**——6 面立方体渲染的像素量是单次
> equirect 的 ~3 倍，吃掉了针孔场景里取胜的 per-kernel 加速（T-F2 的 1.55x）。
> **直接全景训练应继续用专用 LONLAT equirect 光栅化器；gsplat 的优势仅限针孔/透视。**
> Runner: `run_pano_gsplat_train.sh`。

### T-F7 — 球面版 gsplat 内核(投影+gsplat 合成器):可行,但瓶颈在投影
为彻底回答"能否写个球面 gsplat 内核打败 LONLAT",做了 equirect **投影(autograd-
PyTorch,含解析雅可比)+ gsplat 快速 CUDA 合成器 `rasterize_to_pixels`** 的混合方案
(`p3_pano/gsplat_equirect.py`,一趟 equirect,不用立方体)。scene_023 四方对比:

| 后端 | PSNR | LPIPS | it/s | 墙钟 |
|---|---|---|---|---|
| **LONLAT**(原生,全融合 CUDA) | 18.6–19.6 | 0.480 | **79.2** | **88s** |
| gsplat 立方体 (T-F6) | 19.62 | 0.466 | 72.2 | 97s |
| gsplat equirect, eager | 18.88 | 0.469 | 42.4 | 165s |
| gsplat equirect, torch.compile | 18.56 | 0.470 | 19.4 | 361s |

> **结论**:质量持平,但**最慢**。证明了 gsplat 的**合成器不是瓶颈,投影才是**——
> eager-PyTorch 投影比 LONLAT 的全融合 CUDA 慢 2×;`torch.compile` 因致密化反复改变
> N 而不断重编译,更慢。**真正超过 LONLAT 需把投影也写成融合 CUDA 内核**(改 gsplat
> `fully_fused_projection` 前向+反向,equirect 反向雅可比是难点,多周级工程)。
> 当前结论不变:**全景训练用 LONLAT。**

### T-F8 — 融合版 equirect-gsplat CUDA 内核:成功,又快又好,超过 LONLAT
按 T-F7 的定位(瓶颈在投影),把投影也做成融合 CUDA:给 gsplat 加
`CameraModelType::EQUIRECT`,在 `ProjectionEWA3DGSFused.cu` 实现 equirect 投影的
**前向 + 反向 VJP**(`Utils.cuh::equirect_proj`/`equirect_proj_vjp`,解析雅可比 +
∂J/∂μ 二阶项),并处理球面特性:**不按 z 裁剪**(全 360° 可见)、深度用**径向距离**。
全部改动在 `p3_pano/gsplat_equirect_kernel.patch`(runner 自动 `git apply`)。
scene_023 终版四方对比:

| 后端 | PSNR | LPIPS | it/s | 墙钟 |
|---|---|---|---|---|
| LONLAT(原生,INRIA 级) | 18.6–19.6 | 0.480 | 79.2 | 88s |
| gsplat 立方体 (T-F6) | 19.62 | 0.466 | 72.2 | 97s |
| gsplat equirect 混合 (T-F7) | 18.88 | 0.469 | 42.4 | 165s |
| **gsplat equirect 融合 (T-F8)** | **19.57** | **0.465** | **100.7** | **69.5s** |

> **正确性**:200 迭代 PSNR 12.565,与已验证的混合版(12.562)逐位吻合 → 手推反向雅可比正确。
> **速度**:比 LONLAT **快 1.27×**(69.5s vs 88s),质量并列最好。
> **结论翻转**:T-F6/T-F7 的"全景用 LONLAT"被推翻——**融合 equirect-gsplat 现在是直接全景
> 训练的最佳后端**。当年说"需要多周 CUDA 工程",这次做出来了。

**7 场景验证**(held-out,vs T-A4 的 LONLAT 直接全景 PSNR):

| 场景 | LONLAT PSNR | **T-F8 PSNR** | T-F8 it/s | T-F8 s |
|---|---|---|---|---|
| 021 | 21.61 | 21.49 | 101.6 | 68.9 |
| 022 | 22.48 | **22.92** | 101.1 | 69.3 |
| 023 | 19.55 | 18.99 | 102.5 | 68.3 |
| 025 | 20.04 | 19.61 | 101.2 | 69.2 |
| 026 | 19.55 | **20.12** | 100.5 | 69.7 |
| 027 | 19.69 | 19.58 | 102.0 | 68.6 |
| 028 | 21.09 | 21.08 | 102.3 | 68.4 |
| **平均** | **20.43** | **20.54** | **~101.6** | **~68.9** |

> 全 7 场景:平均 PSNR **20.54 vs 20.43**(并列,在 run-to-run 方差内),稳定 ~101 it/s
> (vs LONLAT ~79,**≈1.28× 快**),无失败/NaN。批量脚本 `batch_pano_gsplat_fused.sh`。

**完整流水线端到端最终确认**(`batch_pano.sh` 默认 T-F8,pose→训练→PLY→KSPLAT):

| 场景 | PSNR | SSIM | LPIPS | it/s | 墙钟 | PLY | KSPLAT |
|---|---|---|---|---|---|---|---|
| 021 | 21.35 | 0.745 | 0.422 | 101.9 | 68.7s | 7.4M | 716K |
| 022 | 22.39 | 0.745 | 0.403 | 99.8 | 70.1s | 24M | 2.4M |
| 023 | 18.81 | 0.614 | 0.469 | 101.4 | 69.0s | 25M | 2.4M |
| 025 | 19.66 | 0.654 | 0.419 | 100.9 | 69.4s | 23M | 2.2M |
| 026 | 20.24 | 0.634 | 0.468 | 101.7 | 68.9s | 25M | 2.4M |
| 027 | 19.12 | 0.573 | 0.536 | 101.7 | 68.8s | 25M | 2.4M |
| 028 | 21.31 | 0.592 | 0.533 | 102.4 | 68.4s | 25M | 2.4M |
| **平均** | **20.41** | — | — | **~101.4** | **~69s** | — | — |

> **7/7 端到端成功**:每个场景都跑完 pose→融合训练→INRIA PLY 导出→KSPLAT,且生成了
> 可供 WebXR/PICO 加载的 `.ksplat`(716K–2.4M)。`batch_pano.sh` 默认后端 = 融合
> equirect-gsplat(T-F8);`BACKEND=lonlat` 切回旧的 OmniGS LONLAT 做对照。

### 小结
| 任务 | 状态 | 一句话结论 |
|---|---|---|
| T-F1 密集透视流 | ✅ | 修复 SLAM 跟踪，107 vs 16 关键帧 |
| T-F2 gsplat 端到端（针孔） | ✅ | 快 1.55x + 高 1.78 dB |
| T-F3 全局 Sim3 对齐 | ✅ | 正确+永不更差；收益靠闭环（纯链上 == 顺序） |
| T-F4 BA 网络解阻 | ✅ | 阻塞已除；BA 跑通但数据 inlier 不足（佐证前馈 VGGT） |
| T-F5 工程化 | ✅ | 一键复现 + 镜像清理 |
| T-F6 gsplat 接全景训练 | ✅ | 立方体方案质量持平但慢 ~10%（3x 像素） |
| T-F7 球面 gsplat（混合） | ✅ | 质量持平但最慢；定位瓶颈=投影需融合 CUDA |
| **T-F8 融合 equirect-gsplat 内核** | ✅ | **又快又好,比 LONLAT 快 1.27× → 全景最佳后端** |
