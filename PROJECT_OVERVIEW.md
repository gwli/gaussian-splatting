# 全景无人机 → 3DGS 项目总览

从 INRIA 3D Gaussian Splatting 原始仓库出发，逐步演进为一套**影翎 Antigravity A1
8K 360° 全景无人机 → 三维高斯重建 → VR 浏览**的完整流水线，并在其上做了系统性
的提速优化与研究型扩展。本文梳理从头到现在的全部工作。

> 详细任务清单见 `tasks.md`；优化方案与实测数据见
> `pano_pipeline/OPTIMIZATION_PLAN.md`；各子系统文档见 `p2_vggt/`、`p3_pano/`、
> `p4_slam/`、`webxr_viewer/`。

---

## 一、工作阶段总览

| 阶段 | 内容 | 关键产物 |
|---|---|---|
| **0. 环境与基线** | nvcr.io PyTorch Docker 跑通原版 3DGS（truck/drjohnson/playroom） | CUDA 扩展构建、Web/VR 查看器 |
| **1. 全景无人机流水线** | `.insv` → equirect → 透视 → COLMAP → 3DGS | 飞行规划、`pano_pipeline/`、scenes 021–028 |
| **2. 分级优化 P0/P1/P2** | 并行、迭代数、GLOMAP、KSPLAT、直接全景、**VGGT** | `OPTIMIZATION_PLAN.md` |
| **3. 工程化 backlog (T-A…T-E)** | 全指标、一键流水线、gsplat 基准、窗口 VGGT、BA、SLAM | `tasks.md` |
| **4. 研究型改进 (T-F1…T-F6)** | 密集 SLAM、gsplat 端到端、全局对齐、BA 解阻、全景 gsplat | `p4_slam/`、`p3_pano/` |

---

## 二、最终流水线（推荐路径）

```
影翎 A1 8K 全景无人机 (.insv, 双鱼眼 HEVC, 带 GPS)
  │  ffmpeg v360 (dual-fisheye → equirect, GPU 解码)
  ▼
equirect 全景帧 (4096×2048)
  │  pano_to_perspective.py (多进程, 14 视角/帧)
  ▼
透视裁剪 (≤300, VGGT 显存上限)
  │  ★ VGGT 前馈 SfM (秒级, 10–100× 快于 COLMAP, 无需对应点)
  ▼
相机位姿 + 稠密初始点云 (sparse/0)
  │  ┌─ 透视管线: INRIA / gsplat 训练
  │  └─ ★ 直接全景: train_pano.py (LONLAT equirect 光栅化器, 14× 少图, +2.4dB)
  ▼
3DGS 模型 (.ply) → KSPLAT 转换
  ▼
WebXR 查看器 (PICO 4 Ultra / VR, HTTPS)
```

---

## 三、关键成果

### 重大突破：VGGT 前馈 SfM（P2.1）
整个优化计划的最大突破。用 VGGT 替换 COLMAP 的特征提取+匹配+建图：
- **快 10–100×**（秒级 vs COLMAP 12 分钟建图）
- **质量持平或更好**，且**挽救了 COLMAP 完全失败的 026/027/028**（起飞→巡航→
  降落型轨迹），7/7 场景成功。
- "又快又好"，不像 P0/P1 的开关只能拿质量换速度。

### 直接全景训练（P1.3）—— 移植 OmniGS LONLAT 光栅化器
不再切成透视图，直接在全景图上训练（每相机一个 equirect LONLAT 投影）：

| 场景平均 (held-out, 7 scenes) | 透视管线 | **直接全景** |
|---|---|---|
| PSNR | 18.05 | **20.43 (+2.4 dB)** |
| 训练图像 | 1260 透视裁剪 | **90 全景 (14× 少)** |

### gsplat 后端（T-C1 / T-F2 / T-F6）—— 用对工具
- **针孔训练**：gsplat 端到端比 INRIA **快 1.55× 且 +1.78 dB**（scene_023，同 holdout/迭代）。
- **全景训练**：gsplat 立方体方案质量持平但**慢 ~10%**（6 面 3× 像素）→ 全景仍用 LONLAT。
- 边界清晰：**针孔用 gsplat，全景用 LONLAT**。

### 流式重建 MASt3R-SLAM（T-E1 / T-F1）
- 在 torch-2.6/CUDA-12.6 容器（H100 sm_90）**build + run 全部跑通**。
- **关键发现**：90 张稀疏全景（0.24fps）第 16 帧丢跟踪；从 `.insv` 重渲染**密集
  前向透视流**（4fps）后**连续跟踪 107 关键帧**（6.7×）——证明丢失是采样稀疏
  假象而非能力上限。

### VR 浏览
WebXR 查看器支持 PICO 4 Ultra，`?source=colmap|vggt|pano` 切换，自签 HTTPS。

---

## 四、诚实的负面结果（同样有价值）

| 项 | 结论 |
|---|---|
| **T-B1** 分块 stitch | ✗ 无加速：ffmpeg v360 本就多线程，并行实例只是抢同核 |
| **T-D5/T-F4** 经典 BA | BA 流程跑通，但稀疏全景裁剪 inlier 不足 → "skip BA"；佐证前馈 VGGT 才是对的工具 |
| **T-F3** 全局 Sim3 对齐 | 正确且永不更差，但收益**仅来自闭环**；纯链式航线（本数据）退化为顺序对齐 |
| **T-F6** gsplat 接全景 | 质量持平但更慢（立方体 3× 像素）→ 全景不用 gsplat |

> 这些"负面"结果划清了每种技术的适用边界，避免了过度工程。

---

## 五、仓库结构

| 目录/文件 | 内容 |
|---|---|
| `pano_pipeline/` | 全景流水线核心：`pano_to_perspective.py`、`OPTIMIZATION_PLAN.md`、飞行规划 |
| `p2_vggt/` | VGGT 前馈 SfM：`vggt_window.py`（窗口+全局 Sim3）、`global_sim3.py`、`run_vggt_ba.sh` |
| `p3_pano/` | 直接全景 + gsplat：`train_pano.py`（LONLAT）、`train_pano_gsplat.py`（立方体）、`train_gsplat.py`、`bench_raster.py`、`diff_gaussian_rasterization_pano/`（GPLv3，见 `LICENSE-NOTICE.md`） |
| `p4_slam/` | MASt3R-SLAM：`run_slam_full.sh`、`make_dense_perspective.sh`、`FEASIBILITY.md`、`SETUP.md` |
| `webxr_viewer/` | PICO 4 / VR 的 WebXR 查看器 |
| `tasks.md` | 全任务清单与状态（T-A…T-F） |

---

## 六、技术栈与踩坑要点

- **Docker**：nvcr.io/nvidia/pytorch:24.12-py3（torch 2.6 / CUDA 12.6 / py3.12）；
  numpy 1.x vs 2.x ABI（`--no-deps` / `numpy<2`）；headless 用 `QT_QPA_PLATFORM=offscreen`。
- **CUDA 扩展**：sm_90 (Hopper/H100) gencode；`--no-build-isolation` 避免拉错 torch；
  glm/eigen 等子模块需 vendor（gitlab 不稳）。
- **网络**：`api.github.com` 限流 `torch.hub.load` → 本地缓存权重 + `source="local"`。
- **关键工具**：COLMAP/GLOMAP、VGGT、gsplat 1.5.3、OmniGS（equirect 光栅化）、
  MASt3R-SLAM（lietorch/faiss/asmk）、ffmpeg v360、KSPLAT。

---

## 七、给后来者的一句话建议

1. **SfM 用 VGGT**（前馈，秒级，挽救 COLMAP 失败场景）。
2. **全景重建用直接全景 + LONLAT 光栅化器**（少图、高质、原生 equirect）。
3. **针孔/透视训练用 gsplat**（快 1.55×）。
4. **流式重建用 MASt3R-SLAM**，但要喂**密集前向透视流**（不是稀疏全景）。
5. 大规模航线的窗口合并用 `global_sim3.py`，**前提是有闭环**。
