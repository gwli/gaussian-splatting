# task_ft.md — 把 UniK3D / UniSHARP 深度先验微调到"水乡室外"域

## 0. 一句话目标
UniSHARP 在我们这批航拍全景上"能用但不保真",根因是它的 UniK3D 深度先验训练在
**OmniRooms 室内仿真**(≤100m、有天花板、无天空、≤30cm 位移)。本任务把这个先验
**微调到室外水乡分布**,且**不需要任何深度真值**——用无人机自身在航线上的真实平移
做光度新视角监督。

相关上下游:`[[unisharp]]` 方法解剖见对话;数据来自 `data/8kpano/scenes/scene_NNNhf_pano/`,
位姿来自 `p3_pano/pano_cams_scene_NNNhf.json`(VGGT,240 帧/场景)。

---

## 1. 为什么这次"零视差"不挡路(和之前 3DGS 失败的区别)
- 之前 3DGS 失败 = **优化内三角化**需要同一机位的同步基线,我们没有 → 乱麻。
- 本任务 = **微调前馈先验**,监督方式不同:
  `source 帧 → 预测高斯 → 渲染到 target 位姿 → 对齐真实 target 帧(RGB+感知)`。
  无人机沿航线**真实平移**,真带基线的 (source,target) 对要多少有多少 → 合法监督。
- 深度真值不需要:UniK3D 自己跑一遍当**伪深度弱锚**(`aux_depth_scale_loss`),
  真正拉动微调的是光度损失。这正是仓库里 `datasets/re10k.py`(真实视频、无 GT)
  已有的配方;全景版模板是 `datasets/sim_panorama.py`。

---

## 2. 代码事实依据(已逐一核对,避免拍脑袋)
| 事实 | 出处 |
|---|---|
| 监督 = 光度新视角(RGB+ResNet50 感知)+ UniK3D 伪深度尺度锚 | `losses/unisharp_loss.py` `_ResNet50Perceptual`;`cli/unified_trainer.py` `_aux_ray_losses` |
| 真实视频路子已存在:自动用 UniK3D 生成伪深度,远景按阈值置 0 | `datasets/re10k.py` `pseudo_depth_autogen / pseudo_far_depth_invalid_m=30` |
| 全景训练对是**固定朝向、纯平移**(`_SimFrame` 只有 `position_xyz`) | `datasets/sim_panorama.py:143-148` |
| 全景 dataset 布局:scene_dir 内 rgb + 含"depth"路径的深度文件 + `pose_root/<scene>.csv` | `sim_panorama.py:_scan_scene_frames / _parse_pose_csv` |
| 深度头可训(`requires_grad_(True)`),位置=半径×射线、网络只出有界残差 | `models/unisharp_feature.py:UniK3DCopiedDepthHead`;`models/gaussian_composer.py:_forward_mean` |
| UniK3D 全景深度 API:`infer(rgb, camera=Spherical([fx,fy,cx,cy,W,H,π,π/2]))` | `UniK3D/unik3d/models/unik3d.py:284`;`utils/camera.py:346` |

---

## 3. 命门与风险(按严重度,A 档先验证可行性)
1. **相对位姿质量(最致命)**:光度监督的位姿一偏,梯度全错。
   - 缓解:只用**短帧距**对(相邻 1–8 帧,VGGT 相对位姿比全局重建可靠得多);
     `pano_cams` 的 `R_wp/C` 已现成。**`.insv` 内无 GPS、无 ffmpeg 可见 IMU**(实测:
     双鱼眼 HEVC + mov_text 字幕,无 gpmd)→ 度量位姿路堵死,只能用 VGGT。
2. **尺度一致性**:UniK3D 深度是**米制**,VGGT 的 `C` 是**任意尺度**。位移必须换算到米,
   否则光度损失里"相机移动了多远"与深度不在同一单位 → 系统性错位。
   - 缓解:`--pos-scale` 把 VGGT 单位拉到米。估法:`scale ≈ median(UniK3D 半径距离) /
     median(VGGT 点云到相机距离)`,或用已知地标高度(~15m)反推。A 档先给经验值并在
     验收里检查。
3. **天空 / 远景**:室外几十~几百米,远超室内 30m。
   - 缓解:伪深度里把天空像素置 0(=无效),`far_depth_invalid_m` 抬到 ~150m。
4. **基线参数**:`pair_max_translation_m=0.5` 是室内默认,航拍是米级 → 放大
   (更大的真实基线 = 更强监督,这次是好事)。
5. **过拟合/灾难遗忘**:数据量小,**先 A(只调 heads)/ B(+LoRA),不要直接全量 C**。

---

## 4. 路线 A → B → C
| 档 | 解冻 | 数据量 | 何时上 |
|---|---|---|---|
| **A 只调 heads** | 冻结 UniK3D,只训 `DirectPredictionHead`+`UniK3DCopiedDepthHead` | 1–2 场景、几百对 | **现在,验证域微调是否有效** |
| **B +LoRA 编码器** | A + `pixel_encoder` 加 LoRA | 3–5 场景 | A 见效后 |
| **C 全量微调 UniK3D** | 整个 backbone | 全部场景 + 增广 | 仅当 B 仍受限且数据够 |

---

## 5. A 档数据构造管线(本次已落地)
脚本在 `p6_unisharp/ft/`,产物在 `p6_unisharp/ft/data/`(已 gitignore)。

```
原始: data/8kpano/scenes/scene_NNNhf_pano/panoramas/pano_*.jpg  (240 ERP/场景)
      p3_pano/pano_cams_scene_NNNhf.json                        (R_wp, C / 帧)
  │
  ├─[1] derotate_and_pose.py  按 R_wp 把每帧 ERP 去旋转到统一世界朝向(纯平移残差)
  │        → ft/data/<scene>/NNNNN.jpg        (canonical ERP)
  │        → ft/poses/<scene>.csv             (frame,x,y,z = C*pos_scale, 米)
  │
  ├─[2] pseudo_depth.py       UniK3D(vitl) 在 canonical ERP 上出米制径向深度
  │        + 天空置 0、远景 >far 置 0
  │        → ft/data/<scene>/depth/NNNNN.npy  (float32 H×W, 0=无效)
  │        → ft/data/<scene>/sky/NNNNN.png    (mask, 供 B 档光度降权)
  │
  └─[3] make_pairs.py         按平移幅度[min,max]+帧距筛 (src,tgt) 对
           → ft/pairs/<scene>.jsonl
```
布局即 `SimPanorama` 约定:`root=ft/data, pose_root=ft/poses, scene=<scene>`。

运行:`bash p6_unisharp/ft/build_ft_dataset.sh 027 023`

### 验收(数据档)——027 全量已过
- [x] canonical ERP:027 出 **240 帧** + pose.csv;叠加 100/104/108 三帧,地平线不散
      → 去旋转朝向一致 ✓(拖影来自真实平移基线,符合预期);
- [x] pairs:813 对 / 229 帧有配对(pos_scale=1.0);
- [x] 伪深度:240 帧离线跑通,目视**地面近(蓝)/地平线远(红)/天空黑**,物理合理 ✓;
- [⚠] 但 **valid 跨帧 0.11–0.98、med 3.3–287m 抖动大** → UniK3D 在此 OOD 序列不稳
      (正是微调要修的);A 档靠光度兜底、伪深度仅弱锚。

### 5.2 重力对齐(B 档,已实现)
去旋转后地平线呈正弦波 → canonical 帧对齐到 **VGGT 世界系而非重力垂直**(VGGT 系不水平
+ 无人机横滚)。UniSHARP 训练在**正立 ERP**,倾斜 ERP 是额外 OOD。
`derotate_and_pose.py --gravity-level` 已加重力校平,左乘全局 `R_level`(同一旋转,不破坏
帧间纯平移),两种估"上"法:
- **`--level-method horizon`(默认,推荐)**:把每帧 sky/ground 边界拟合到倾斜大圆
  `sin(lon)nx+tan(lat)ny+cos(lon)nz=0`(齐次 LS),pano 系法向 →`R_wp^T`→ 世界"上",多帧平均。
- `--level-method trajectory`:相机轨迹 PCA 最小轴;**仅平飞有效,banked footage 失败**
  (实测 027 给出 ≈+Y 的错误估计,地平线没拉平)。

**实测(027)**:horizon 法 12 帧一致性 `spread=0.005`,修正 ~8°,亮带从"骑在赤道上方/斜穿"
变为**对称居中于赤道**。残留弯曲是高空斜俯视的真实几何(远景在 ERP 本就弯),非全局旋转可消。
B 档重建数据:`GRAVITY=1 bash build_ft_dataset.sh 027`(rgb 旋转后伪深度需一并重算)。

### 5.1 冒烟实测发现(027,2 帧)——两个真实结论
1. **伪深度严重 OOD**:UniK3D 把整景压在 **56–150m**(med 84–134m),且 far=150 截断后
   **只剩 16–35% 有效**。这正面印证"室内先验在室外失真"——但也说明**伪深度只能当很弱的
   尺度锚,真信号必须靠光度**。调整:`FAR_INVALID_M` 抬到 **~300**(留住更多远景),
   A 档训练时把 `aux_depth_scale_loss_weight` 调小。
2. **尺度修正(推翻先前猜测)**:相邻帧位移 med≈0.20 VGGT 单位,×30fps≈**6 单位/秒**——
   与无人机 ~6m/s 巡航吻合,说明 **VGGT 单位已近似米,`POS_SCALE≈1` 即可**,不要用先前
   "≈10"的值(会把基线灌大 10 倍 → 光度监督全错)。**严谨估法**:取一对,用光流视差
   `disparity ≈ B/Z` + UniK3D 的 Z 反解 B,与 `||C_tgt-C_src||·pos_scale` 对齐定标
   (留作 step3.5,A 档先用 1.0)。
3. 推论:基线 0.2–1m / 深度 ~100m → B/Z 很小,远景光度信号弱。**对策**:训练对偏向
   `--max-gap` 大端取更大基线;近景(地面/建筑)才是主要监督来源,远景靠伪深度弱锚兜底。

---

## 6. A 档训练(已落地并跑通)
`train_a.py`(launcher)+ `train_a.sh`(wrapper)驱动**现成 CLI**,无需另写 trainer。
直接复用 `SimPanorama`(我们的数据布局就是按它造的)。
```
bash p6_unisharp/ft/train_a.sh 027              # STEPS/LR0/UNIK3D_LR0/... 见脚本头
```
**关键:stock CLI 没有 resume,且训练分支的 UniK3D 离线加载有 bug**。launcher 用三处
干净 monkeypatch 解决(不改 gitignore 的 clone):
1. **加载预训练权重**:模型构造后 `load_from_checkpoint(INIT_CKPT, strict=False)`
   → 实测 `missing=0 unexpected=1`,确是**从预训练微调**而非从零;
2. **离线 UniK3D**:adapter 的 `UniK3DHub(config_variant="eval")` 在官方 clone 上必抛
   TypeError 且 offline 时拒绝回退 → patch `unik3d_adapter.load_unik3d_model` 直接走
   `UniK3D.from_pretrained`(缓存,离线);
3. **配对/尺度**:SimPanorama 写死 `pair_max_translation_m=0.5`、`position_scale=0.01`
   (室内默认)→ 经 env 改 `SIM_PAIR_MAX_TR=6`、`SIM_PAIR_MIN_OVERLAP=0.1`、
   **`position_scale=1.0`**(否则米制位移被缩 100×、与米制伪深度单位不一致 → 0 对)。

**冻结策略**(纯靠 CLI 分组 LR,无需 patch):`--unik3d-encoder-lr0/1 0` 冻 DINOv2 编码器;
`--unik3d-lr0/1 1e-5` 轻调 UniK3D 解码器+深度头(**即"微调深度先验"**);`--lr0 1e-4` 训 heads。

**冒烟实测(027,8 步)**:`dataset=sim` 采到对、loss 9.0–9.8 无 NaN、tgt(新视角)loss≈2.7、
~1.1s/步(gsplat 扩展首次编译 ~187s)。✅ 端到端可训。

## 6.1 正式训练 + 留出评测(023,已完成)✅
完整管线已跑通并验证微调**确实涨点**。
1. `make_holdout.py --scene scene_023hf --val-frac 0.2` → train 192 帧 / val 48 帧(symlink,不复制);
2. `make_val_manifest.py --scene scene_023hf_val` → 验证用显式配对 manifest(`scene|src|tgt,..`);
3. 训练:`SC_NAME=scene_023hf_train STEPS=2000 ... train_a.sh 023`
   → loss **8.67→2.46**,tgt(新视角)**2.01→1.77**,存档 step_{500..2000}.pt;
4. 评测:`eval_a.py`(=`run_validation` + 同款 monkeypatch),留出 48 对,微调前 vs step_2000:

| | PSNR | SSIM |
|---|---|---|
| 微调前(pretrained) | 18.713 | 0.6231 |
| 微调后(step_2000)  | **19.121** | **0.6275** |
| 增量 | **+0.41 dB** | **+0.0044** |

**结论:A 档域微调在留出帧上确有正向提升(非过拟合)**。幅度温和符合 A 档预期
(冻结编码器、单场景 2k 步、伪深度/VGGT 位姿有噪声、8° 重力残差)。

### 评测踩坑(已固化)
- `run_validation` 的 sim manifest 是 **`scene|src|tgt1,tgt2,..`** 三段式,不是场景名列表;
- `_load_model` 用 `strict=True` 载入,预训练 ckpt 多 `payload.depth_alignment` → eval_a.py 强制 strict=False;
- 渲染路径有**偶发异步 CUDA 错误**(同帧单独跑正常)→ eval_a.sh 固定 `CUDA_LAUNCH_BLOCKING=1`(慢但稳);
- 验证 SimPanorama 同样写死 `pair_max_translation_m=0.5/position_scale=0.01`,且原生 ERP 分辨率 → monkeypatch 改米制 + `max_long_edge=512`。

### 验收(微调档)
- [x] 留出帧新视角 **PSNR/SSIM** 微调后优于微调前(+0.41dB / +0.0044);
- [ ] LPIPS(blocking 下太慢,暂用 FAST 跳过;后续单跑一次全指标);
- [ ] 深度直方图从"~150m 轨道"展开(待 B 档);
- [ ] 大位移 novel view 鬼影/拉花减少(对比 `[[unisharp]]` 当前 027 fused)。

## 6.2 B 档(多场景 + 解冻编码器)— 已做,结论:不及预期
`MANIFEST_SCENES="scene_023hf_train 021 022 025 026 028"`(均重力校平)+ `ENC_LR0=1.5e-6`
(解冻 DINOv2 编码器)+ 3000 步。同一 023 留出 48 对评测:

| 配置 | PSNR | SSIM |
|---|---|---|
| 基线(pretrained) | 18.713 | 0.6231 |
| A 档(023 单场景 2k,**冻结**编码器) | **19.121** | 0.6275 |
| B 档(6 场景 3k,**解冻**编码器) | 19.111 | **0.6318** |

**结论:多场景+解冻编码器没有在 PSNR 上超过 A 档**(19.11≈19.12,噪声内),仅 SSIM +0.0043。
解读:评测只在 023,训练别的场景对 023 留出帧帮助有限;~19.1 dB 像是当前数据质量
(伪深度噪声/VGGT 位姿/重力残差)的天花板,A、B 都触顶。**真正的杠杆在数据质量与
单场景特化,不在模型容量**。`train_a.sh` 已支持 `MANIFEST_SCENES`/`ENC_LR0` 复现。

## 6.3 长训练(023 单场景 6000 步)— 步数 vs 指标曲线
逐 checkpoint 在同一 023 留出 48 对评测(曲线 `runs/long_curve.png`):

| 步数 | PSNR | SSIM |
|---|---|---|
| 0(baseline) | 18.713 | 0.6231 |
| 1500 | 19.127 | **0.6360**(SSIM 峰) |
| 3000 | 19.275 | 0.6317 |
| 4500 | 19.251 | 0.6288 |
| 6000 | **19.295** | 0.6301 |

**结论:增益主要在前 1500 步拿到(+0.41dB,SSIM 最高);之后 PSNR 缓爬到 +0.58dB
但 SSIM 单调下滑 → 轻度过拟合。最佳折中 ≈ step 1500。A/B 档点都落在同一平台
→ ~19.3 dB 是数据质量天花板,与步数/场景数/编码器无关。**

## 6.4 全指标(含 LPIPS)— 重要修正:感知质量退化
对 baseline 与 long step_3000 跑完整指标(去 `--fast`):

| | PSNR ↑ | SSIM ↑ | **LPIPS ↓** |
|---|---|---|---|
| pretrained | 18.713 | 0.6231 | **0.4665** |
| ft step_3000 | 19.275 | 0.6317 | **0.4987** |

**⚠ LPIPS 变差(0.4665→0.4987),尽管 PSNR/SSIM 升。** 即微调把输出推向更贴合
像素亮度/低层结构,却牺牲了感知真实感(更平滑/模糊)——光度+L2 监督的典型副作用。
**净质量混合偏负:不能只看 PSNR 宣称"变好"。** LPIPS 是最接近人眼的指标,它退化说明
当前 A 档配方(强光度、弱噪声伪深度锚)对"看起来更真"并无帮助,甚至有害。
**修正建议:提高 `--lambda-percep`、降 `--lambda-color`,或改善伪深度/位姿质量;
单纯调步数/场景/编码器都救不了感知质量(§6.2/6.3 已证天花板)。**

## 6.5 深度直方图(预测高斯径向距离)— 因果链闭合
对 023 留出帧 00200,预训练 vs 微调 step_3000 各出 ply,直方图 `‖xyz‖`
(`depth_hist.sh`/`depth_hist.py`,图 `runs/depth_hist_00200.png`):

| | 中位 | p10–p90 | <50m |
|---|---|---|---|
| pretrained | **2.1m** | 1.3–10.2 | 100% |
| ft step_3000 | **98.5m** | 80.8–99.4 | 4% |

**预训练把整景压在 1–10m(OmniRooms 室内先验,尺度完全错);微调把它推到室外
~100m 尺度——这正是本任务的几何目标,且解释了 PSNR/SSIM 提升(尺度对了,重投影更准)。
但微调把深度塌缩成 ~100m 的窄壳层(p10–p90 仅 80–99m,几乎无相对结构),因为喂的
UniK3D 伪深度本身就平(§5.1:OOD 下 med~100m、低方差)。壳层几何 = 视差细节丢失 =
LPIPS 退化。**

## 7. 总结论(四项实验闭环)
1. **域微调确实把深度先验从室内 ~2m 纠正到室外 ~100m**(几何目标达成),带来 +0.4–0.6dB
   PSNR / +SSIM 的留出增益,且非过拟合(留出帧)。
2. **但感知质量(LPIPS)退化**,根因是伪深度锚本身平坦 → 微调学到一个 ~100m 窄壳层,
   丢了相对深度结构。
3. **天花板在数据/监督质量,不在模型容量**:多场景、解冻编码器、6000 步都触同一 ~19.3dB
   平台;step 1500 即近最优。
4. **下一步唯一有效杠杆 = 提升监督质量**:更好的伪深度(多视一致/单目-视频深度而非单帧
   UniK3D)、真实稀疏深度锚、或重权感知损失;而非继续堆步数/场景/参数。
全部脚本(`derotate_and_pose`/`pseudo_depth`/`make_pairs`/`build_ft_dataset`/`make_holdout`/
`make_val_manifest`/`train_a`/`eval_a`/`depth_hist`)均已落地、跑通、可复现。

---

## 7. 当前状态
- [x] 可行性论证 + 代码核对 + 风险定级(本文件)
- [x] A 档数据脚本:`derotate_and_pose.py / pseudo_depth.py / make_pairs.py / build_ft_dataset.sh`
- [ ] 跑 027/023 生成数据并过"数据档验收"
- [ ] A 档训练封装 + 冻结策略 + 留出评测
