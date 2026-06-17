# SR_COMPARISON — 360 视频超分方法实测对比

对 task_sr.md 列出的方案逐一实现并在**真实 027 8K 素材**(水乡航拍)上量化对比。
评测协议:自监督降采样回升(真实帧当 GT → bicubic 降 → 各方法升回 → 比),
指标 PSNR↑ / WS-PSNR↑(纬度加权) / SSIM↑ / LPIPS↓ / sharpness(Laplacian 方差) /
每帧耗时。代码全部自带(无 basicsr/mmcv 依赖),权重均为官方发布。

## 一、单帧超分(SISR)—— 真实 8K 细节裁块,×2

GT = 从真实 8K equirect 赤道带裁 1024² **原生细节块**(8 帧),LR = ÷2,各方法 ×2 还原。

| 方法 (代号) | PSNR | WS-PSNR | SSIM | LPIPS↓ | sharpness | s/帧 |
|---|---|---|---|---|---|---|
| **M0** lanczos+锐化 | **33.61** | **32.61** | 0.9974 | 0.2355 | 207 | **0.03** |
| **M1** Real-ESRGAN RRDBNet | 31.21 | 30.39 | 0.9833 | 0.3046 | 188 | 0.14 |
| **M3** SwinIR(transformer) | 31.38 | 30.46 | **0.9962** | **0.2433** | **629** | 5.24 |

**读法(重要)**:这里退化很轻(干净的 ÷2),且 GT 本身是 x265 压过的 8K,高频有限。
- **lanczos PSNR/保真最高** —— 轻退化 + 偏软 GT 下,插值天然最贴近 GT;但 sharpness 平庸,
  本质没补细节。
- **SwinIR 细节恢复最强**(sharpness 629,是其它 3× 以上)且 **SSIM 0.9962 / LPIPS 0.2433**
  几乎与 lanczos 持平 —— **学习型里"细节/保真"折中最好**,代价是慢(5.2 s/帧,transformer)。
- **Real-ESRGAN(M1)在本场景最弱**(LPIPS 0.305 最差):它是为**重度真实退化**调的 GAN,
  喂给它轻退化会"过处理"、编出偏离 GT 的纹理。

> 结论 1:退化越轻 → 经典插值越够用;要真正补细节选 **SwinIR**;Real-ESRGAN 适合**重度退化/老素材**。

## 二、360 感知(cubemap)—— 全 equirect 帧,×2

GT = 全景帧降到 2048×1024(含极点),对比 equirect 直接 SR vs cubemap 分面 SR。

| 方法 | PSNR | WS-PSNR | SSIM | LPIPS↓ | sharpness | s/帧 |
|---|---|---|---|---|---|---|
| M1 rrdbnet(equirect 直接) | 33.35 | 32.19 | 0.9878 | 0.2485 | 108 | 0.15 |
| **M2** rrdbnet-cube(6 面) | 32.70 | 31.76 | 0.9727 | 0.2812 | 47 | 0.58 |

> 结论 2:cubemap **在本素材上反而更差**(PSNR/sharpness 都降)。原因:cube 往返要**两次
> 重采样**(equirect→面→equirect),引入模糊;而本片高频内容都在赤道带(航拍地面),
> **极点是天空/地面低频**,cube 修极点畸变的收益 < 重采样损失。
> cubemap 只在**极点也有丰富细节**时才划算;否则别用。

## 三、视频超分(VSR)—— 连续 16 帧,×4

GT = 连续 16 帧 1024² 原生裁块,LR = ÷4(更重的退化),对比逐帧 SISR vs 双向 VSR。
额外测**时间一致性 warp-error**(用 GT 光流把上一帧 warp 到当前帧比残差;低=不闪)。

| 方法 | PSNR | SSIM | LPIPS↓ | **warp-err↓(闪烁)** |
|---|---|---|---|---|
| M1 Real-ESRGAN RRDBNet ×4(逐帧) | 31.31 | 0.9529 | 0.3665 | 2.722 |
| **M4** Real-BasicVSR ×4(双向传播) | **32.43** | **0.9606** | **0.3535** | **2.198** |

> 结论 3:**Real-BasicVSR 全面胜出** —— PSNR **+1.12 dB**、LPIPS 更低,且**时间闪烁降 19%**
> (2.722→2.198)。退化越重(÷4),学习型/VSR 的优势越明显;**时间一致性是 SISR 补不了的硬伤,
> 正是 VSR 的价值**。

## 四、总推荐

| 场景 | 推荐 | 理由 |
|---|---|---|
| 轻退化 / 只要快 | **M0 lanczos**(`enhance.sh 2x`) | 保真最高、零成本 |
| 单帧要真细节 | **M3 SwinIR** | 细节恢复最强、保真损失最小 |
| 重度真实退化 / 老旧压缩素材 | **M1 Real-ESRGAN** 或 **M4** | GAN 退化建模 |
| **视频成片(本项目主线)** | **M4 Real-BasicVSR** | 质量+时间一致性最佳,消灭逐帧闪烁 |
| 极点也有细节的全景 | + **M2 cubemap** 包装 | 否则 cube 的重采样得不偿失 |

**面向 task3 的落地建议**:给 `enhance.sh` 增加 `swinir` 与 `vsr`(Real-BasicVSR)两个引擎;
8K 成片走 **VSR**(时间一致),快速预览走 lanczos;cubemap 仅在素材极点细节丰富时启用。

## 五、性能差异原因分析

### 5.1 画质差异 —— 为什么排名是这样

**(a) 退化强度 × 模型训练分布(最关键)**
Real-ESRGAN / Real-BasicVSR 都是在**重度合成退化**(多次模糊+降采样+噪声+JPEG/视频压缩)上训练的。
- 喂**轻退化**(干净 ÷2)→ 输入分布与训练不符,模型仍按"重退化"假设强行去模糊/造纹理,
  结果**过处理**、编出偏离 GT 的细节 → ÷2 时 Real-ESRGAN 的 LPIPS 最差(0.305)。
- 喂**重退化**(÷4)→ 正中训练分布 → Real-BasicVSR 在 ÷4 全面领先。
→ "哪个 SR 好"高度依赖**退化强度要与模型训练分布匹配**。

**(b) 指标本身偏向保真,而 GAN 用保真换感知**
PSNR/SSIM 度量"逐像素贴近 GT"。GAN(Real-ESRGAN)主动**牺牲保真换视觉锐利**,
会在 PSNR 上吃亏;当 GT 又偏软(8K 已被 x265 压过)时,**插值(lanczos)天然最贴 GT**,
所以 lanczos 在轻退化下 PSNR 反而最高 —— 这是 SR 基准的经典"PSNR≠观感"陷阱,故我们同时报 LPIPS。

**(c) SwinIR 为何"细节最强且保真不掉"**
SwinIR 是 **L1 重建训练(非 GAN)+ 窗口自注意力**:
- 非 GAN → 不编假纹理 → 保真(SSIM 0.9962)与 lanczos 持平;
- 自注意力的**长程上下文**比 RRDBNet 的局部卷积更会"推断"真实高频 → sharpness 629(其它 3×),
  且这些高频是**贴合 GT 的**(LPIPS 0.243 优于 RRDBNet)。
→ 单帧"既要细节又要保真"它最优。

**(d) cubemap 为何反而更差**
equirect→cube→SR→cube→equirect 要做**两次 grid_sample 双线性重采样**,每次都是一次低通(模糊)。
该模糊是确定的损失;而 cube 的收益(均匀采样、消除极点拉伸)只有当**极点本身有高频细节**时才兑现。
本片极点是天空/地面(低频),赤道才是航拍地物(高频)→ **收益≈0,损失实打实** → PSNR/sharpness 双降。

**(e) VSR 为何质量+稳定双赢**
- **质量**:相邻帧之间有**亚像素位移**,双向传播 + 光流对齐把多帧信息聚合到当前帧,
  等于"多帧融合超分",比单帧凭空猜更接近真值 → ÷4 时 PSNR 比逐帧 RRDBNet +1.12dB。
- **时间一致性**:SISR 逐帧独立 → 每帧"猜"的高频不一样 → 闪烁;VSR 的循环状态让相邻帧
  共享传播特征 → warp-error 2.20 vs 2.72(**闪烁 −19%**)。这是单帧方法**结构上补不了**的。

### 5.2 速度差异 —— 为什么差几十倍

| 方法 | s/帧(×2,~1K) | 计算结构 | 慢/快的根因 |
|---|---|---|---|
| lanczos | 0.03 | 固定可分离卷积核 | 无网络,O(像素),CPU 即可 |
| RRDBNet | 0.14 | 23 个残差稠密块,纯卷积,**在 LR 分辨率算**最后 pixelshuffle 上采 | 卷积对 GPU/tensor-core 友好,且大部分算力在低分辨率 |
| SwinIR | 5.24 | 窗口自注意力(transformer)多层,**接近输入分辨率算** | 注意力 + LayerNorm/GELU 开销大;**fp16 会 NaN → 只能 fp32**,吃不到 tensor-core 加速 → 比 RRDBNet 慢 ~35× |
| VSR | 重(×4) | 逐帧卷积传播 + SPyNet 6 级光流金字塔 + **双向两遍** + warp | 比 RRDBNet 多了光流估计与双向遍历;且 ×4、需按窗口分块控显存 |

要点:
- **算在什么分辨率上**决定成本量级:RRDBNet/ESRGAN 主体在 LR 算(便宜),SwinIR 在高分辨率做注意力(贵)。
- **fp16 可用性**:RRDBNet/VSR 支持 fp16(tensor-core 提速~2×);**SwinIR fp16 数值溢出(softmax/LayerNorm 动态范围)→ 强制 fp32**,既慢又占显存。
- **VSR 的显存**正比于"窗口帧数 × 输出分辨率",故必须**滑窗分块**(本实现 `VSR_WIN/VSR_OVERLAP`);
  8K 成片应"低分辨率拼接(如 1920×960)→ VSR ×4 → 8K",而非在 8K 上直接超分。

### 5.3 选型速记
- 要快/保真、退化轻 → **lanczos(ffmpeg)**;要单帧真细节 → **SwinIR**(慢但最好);
- 老旧/重压缩素材 → **Real-ESRGAN**;**视频成片 → Real-BasicVSR(VSR)**(质量+不闪);
- 仅当极点有细节 → 加 **cubemap**。

## 六、实现清单(`p5_sr/sr_methods/`)
- `sr_lib.py` —— RRDBNet / SwinIR 加载、tiled 推理、equirect↔cubemap(torch grid_sample,
  往返误差 0.002)、统一 dispatch。
- `vendor/network_swinir.py` —— 官方 SwinIR 架构(权重 key 对齐)。
- `vsr_realbasicvsr.py` —— **自写 Real-BasicVSR**(image_cleaning + SPyNet 光流 + 双向传播 +
  pixelshuffle ×4),`strict=True` 加载官方权重,**零 mmcv 依赖**。
- `sr_eval.py` / `temporal_eval.py` —— SISR / VSR 评测(PSNR/WS-PSNR/SSIM/LPIPS/warp-err)。
- `sr_compare.sh`(`GT_MODE=crop|scale`)/ `vsr_temporal.sh` —— 一键复现本报告。
