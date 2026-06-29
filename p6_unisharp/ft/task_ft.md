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

### 5.2 重力对齐(目视新发现 → B 档 TODO)
去旋转后地平线呈**正弦波**,说明 canonical 帧对齐到的是 **VGGT 世界系,而非重力垂直**
(VGGT 系不水平 + 无人机横滚)。影响:跨帧一致 → 纯平移契约成立(**A 档可用**);
但 UniSHARP 训练在**正立 ERP**,倾斜 ERP 是额外一层 OOD。
**B 档改进**:在 `derotate_and_pose.py` 加重力校平——拟合地平线正弦相位/幅度估出"上"向量,
左乘一个全局 `R_level` 把所有帧拉正(全局同一旋转,不破坏帧间纯平移关系)。

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

## 6. 训练档(A 档,下一步,本次不实现)
1. 写 `WatertownPanoDataset` ≈ `SimPanorama` 薄封装(读上面布局);或直接复用
   `SimPanorama(root=ft/data, pose_root=ft/poses, far_depth_invalid_m=150,
   pair_max_translation_m=8)`。
2. 在 `unified_trainer` 里冻结除两个 head 外的参数,光度+感知+伪深度尺度锚。
3. 留出帧(每场景尾 20%)做验证。

### 验收(微调档)
- [ ] 留出帧新视角 **PSNR/LPIPS** 微调后优于微调前;
- [ ] 深度直方图从"室内 30m 截断"展开到室外量级;
- [ ] 大位移(2–6m)novel view 鬼影/拉花减少(对比 `[[unisharp]]` 当前 027 fused)。

---

## 7. 当前状态
- [x] 可行性论证 + 代码核对 + 风险定级(本文件)
- [x] A 档数据脚本:`derotate_and_pose.py / pseudo_depth.py / make_pairs.py / build_ft_dataset.sh`
- [ ] 跑 027/023 生成数据并过"数据档验收"
- [ ] A 档训练封装 + 冻结策略 + 留出评测
