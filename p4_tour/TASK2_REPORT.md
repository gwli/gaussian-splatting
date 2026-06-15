# task2 — 全景无人机 → 3DGS → 观光高光视频(技术报告 / MVP)

目标(见 `../task2.md`):从全景无人机视频重建 3DGS 场景,用**虚拟相机**重新规划
观光路线,渲染适合发布的高光视频(优先 16:9 / 9:16,可选 360° equirect)。

## 一、调研:复用已有 vs 新增

task2 的"重建"半程在本仓库**已完成并验证**(见 `../pano_pipeline/OPTIMIZATION_PLAN.md`、
`../tasks.md`):

| task2 步骤 | 现状 | 复用 |
|---|---|---|
| 1 采集 / 2 预处理(抽帧/转视图) | ✅ `prep_pano.sh`(stitch + 透视裁剪),**FPS 选帧**(T-G,默认) | 复用 |
| 3 相机位姿与初始几何 | ✅ **VGGT 前馈 SfM**(10–100× 快于 COLMAP,挽救失败场景) | 复用 |
| 4 3DGS 重建 | ✅ **融合 equirect-gsplat 内核**(T-F8,比 LONLAT 快 1.28×) | 复用 |
| **5 虚拟观光路线** | 🆕 本次实现 | — |
| **6 视频渲染** | 🆕 本次实现 | — |
| 7 后期剪辑 | ⏳ 转场/调色/音乐留给 NLE,本 MVP 出原始高光片段 | — |

→ 本次只需补 **5+6**:虚拟相机路径 + 渲染成片。

## 二、设计与实现:`tour_render.py` + `run_tour.sh`

`tour_render.py`:
1. **加载训练好的 .ply**(INRIA 格式)→ gsplat 参数(means/quats/exp(scale)/
   sigmoid(opacity)/SH K=16)。
2. **从全景相机推导场景坐标系**(关键):
   - 中心 = 相机中心中位数;
   - **重力上方向**:全景是 gravity-aligned equirect,视图 +y=下,故世界下方向 =
     mean(R_wpᵀ·[0,1,0]),取反得 up——避免 VGGT 任意坐标系下"上"未知的问题;
   - 半径 = 相机水平展布中位数 ×1.25。
3. **虚拟相机路径**(以观感为目标,平滑稳定):
   - `orbit` 环绕(smoothstep 缓动,~0.9 圈);
   - `fly` 沿采集轨迹的 Catmull-Rom 样条飞行(look-ahead + 偏向中心);
   - `dolly` 推进式揭示(远→近)。
   - 全部 look-at + 重力 up 构造 viewmat。
4. **渲染**:perspective 用 gsplat 针孔光栅化(16:9 / 9:16,可调 hfov);
   `--mode equirect` 用 T-F8 融合内核出 2:1 360°。
5. `run_tour.sh`:容器内渲染帧 → ffmpeg(libx264, crf18, yuv420p, faststart)成 mp4。

用法:
```
bash p4_tour/run_tour.sh <ply> <pano_cams.json> <out.mp4> \
     [shots=orbit,fly,dolly] [res=1920x1080] [fps=30] [secs/shot=6] [mode=perspective|equirect]
```

## 三、验证(scene_023,T-F8 训练模型,104801 高斯)

| 产物 | 规格 | 大小 |
|---|---|---|
| `tour_023_highlight.mp4` | 1920×1080, 30fps, 18s(orbit+fly+dolly 各 6s) | 9.2 MB |
| `tour_023_vertical.mp4` | 1080×1920, 30fps, 12s(orbit+dolly) | 3.0 MB |

- 端到端跑通:训练 .ply → 虚拟路径 → gsplat 渲染 → mp4,`ffprobe` 校验规格正确,
  抽帧确认渲染的是**真实重建场景**(草地/地形 + 地平线 + 天空),沿平滑虚拟轨迹运动。
- equirect 360° 模式已实现(`--mode equirect`,走 T-F8 内核)。

## 四、已知问题与建议(对应 task2「关键风险」)

- **离开采集体积 → 雾化/空洞**:相机升太高/拉太远会进入欠观测区,渲染变灰雾
  (实测复现:orbit 抬到 0.6×radius 时整屏灰天)。**对策**:虚拟路径必须贴近采集
  视点(本实现已把 orbit/fly/dolly 都限制在采集高度附近)。这是全景航拍数据的硬约束。
- **天空/远景模糊**:pano 训练模型远景是低频高斯,天空发灰——观感上接受,或后期替换天空。
- **几何偏软**:1024×512 全景训练 + 90 帧,细节有限;提分路线:更高训练分辨率、
  更多/FPS 选帧、按 task2 建议补采重点区域的环绕/多高度素材。
- **360° 输出更易暴露缺陷**:equirect 全景会把欠观测区的雾化放大,建议重建完整度
  足够再输出。

## 五、后续任务 —— **全部完成并验证(2026-06-15)**

1. **✅ 导演级关键帧路径** —— `tour_render.py --keyframes <json>`:场景相对柱坐标
   关键帧(az 方位/r 半径×/h 高度×/t 秒),Catmull-Rom 平滑插值 + look-at 中心。
   样例 `keyframes_demo.json`(5 关键帧揭示运镜)。验证:`tour_023_director.mp4`(15s)。
2. **✅ 后期制作** —— `post.sh <in> <out> [标题] [音乐]`:调色(对比/饱和/S 曲线/锐化/
   暗角)+ 淡入淡出 + 标题字幕 + 可选 BGM(纯 ffmpeg,不重渲)。验证:
   `tour_023_graded.mp4`(标题"SCENE 023 · 3DGS AERIAL TOUR" + 调色)。
3. **✅ POI 兴趣点感知路径** —— `tour_render.py --poi auto|x,y,z;...`:auto 用体素密度
   自动检测最密结构区(避开天空/雾)作 POI,逐个环绕 dwell。验证:`tour_023_poi.mp4`
   (自动检出 3 个 POI → 3 段运镜)。
4. **✅ 多场景批量 + 360° equirect** —— `batch_tour.sh`:7 场景各出 highlight + 调色版,
   外加 scene_023 的 **360° equirect 成片**(走 T-F8 内核)。验证:`p4_tour/out/` 下
   7×2 普通视频(021–028,横屏 1080p)+ `scene_023_equirect360.mp4`(1024×512 2:1,
   抽帧确认为有效全景),7/7 成功。

### 产物清单(`p4_tour/out/`,git-ignored,可经 `batch_tour.sh` 复现)
| 类型 | 文件 | 规格 |
|---|---|---|
| 7 场景横屏高光 | `scene_0NN_highlight.mp4` | 1920×1080 18s |
| 7 场景调色成片 | `scene_0NN_graded.mp4` | 1920×1080 18s(调色+标题) |
| 360° 全景成片 | `scene_023_equirect360.mp4` | 1024×512 2:1 12s |
| 导演关键帧 | `tour_023_director.mp4` | 1280×720 15s |
| POI 兴趣点 | `tour_023_poi.mp4` | 1280×720 12s |
| 竖屏 | `tour_023_vertical.mp4` | 1080×1920 12s |

### 仍可继续(超出 task2 范围的工程化)
- 转场:多片段 crossfade(需分镜分段渲染);
- POI 升级:结合 `.insv` GPS / 人工标注景点;
- 节奏剪辑:按音乐节拍自动切镜。
