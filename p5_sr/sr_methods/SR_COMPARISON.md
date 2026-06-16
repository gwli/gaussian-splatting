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

## 五、实现清单(`p5_sr/sr_methods/`)
- `sr_lib.py` —— RRDBNet / SwinIR 加载、tiled 推理、equirect↔cubemap(torch grid_sample,
  往返误差 0.002)、统一 dispatch。
- `vendor/network_swinir.py` —— 官方 SwinIR 架构(权重 key 对齐)。
- `vsr_realbasicvsr.py` —— **自写 Real-BasicVSR**(image_cleaning + SPyNet 光流 + 双向传播 +
  pixelshuffle ×4),`strict=True` 加载官方权重,**零 mmcv 依赖**。
- `sr_eval.py` / `temporal_eval.py` —— SISR / VSR 评测(PSNR/WS-PSNR/SSIM/LPIPS/warp-err)。
- `sr_compare.sh`(`GT_MODE=crop|scale`)/ `vsr_temporal.sh` —— 一键复现本报告。
