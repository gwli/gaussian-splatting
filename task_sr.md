# task_sr — 360 视频超分方法对比

针对"压缩过的真实 8K 360 航拍(Insta360/Antigravity `.insv`)→ 提画质 / 2x"这一具体场景,
把候选超分方案逐一实现,并在**统一基准**上量化对比,给出推荐。

## 一、待对比方案

| 代号 | 方法 | 类型 | 关键点 / 预期 |
|---|---|---|---|
| **M0** | Lanczos + 锐化 | 经典插值 | `enhance.sh 2x` 现状。快、零模型,但不补真细节,放大后软。基线。 |
| **M1** | Real-ESRGAN(RRDBNet)×2,**equirect 直接** | 学习型 SISR + GAN | 现状 `realesrgan_infer.py`。能补纹理,但单帧、极点被拉伸、赤道缝合被放大。 |
| **M2** | Real-ESRGAN(RRDBNet)×2,**cubemap 分面** | SISR + 360 感知 | equirect→立方体 6 面(近透视,匹配模型分布)→ 逐面 SR → 合回。修极点畸变。 |
| **M3** | **SwinIR**(transformer SISR)×? | 更强 SISR 骨干 | 比 RRDBNet 保真度高;对比"换更好骨干"的收益。 |
| **M4** | **Real-BasicVSR**(视频超分) | VSR + 真实退化 | 多帧对齐 + 时间传播,理论上质量+时间一致性最好。最契合压缩视频。冲刺项。 |
| (备注) | StableSR / SUPIR(扩散) | 生成式 SR | 细节惊艳但会幻想、极重、视频时间不稳。仅列为未来项,不在本轮跑。 |

## 二、评测协议(无 GT → 自监督降采样回升)

真实素材没有"更高清的标准答案",故用标准做法:**拿真实高清帧当 GT,降采样造低清,再让各方法
还原,与 GT 比**。

- **GT**:从 027 的 8K 成片(`p5_sr/out/scene_027_8k_highlight.mp4`,7680×3840 equirect)
  采样 K 帧(默认 8 帧,均匀分布)。为控成本可在**中心纬度裁剪 patch**(赤道带,
  避免极点黑边主导指标),patch 默认 1280×1280。
- **LR**:对 GT 做 bicubic ↓2 → SR 各方法 ×2 → 回到 GT 尺寸。
- **指标**:
  - **PSNR**↑、**SSIM**↑(保真度)
  - **LPIPS**↓(感知质量,piq/lpips)
  - **WS-PSNR**↑(纬度加权 PSNR,ERP 专用;对全帧评测时用)
  - **Sharpness**(Laplacian 方差,无参考清晰度)
  - **runtime/帧**、**显存峰值**(工程成本)
- **VSR 额外**:在真实短序列上测**时间一致性**(相邻帧对齐后残差 / warp-error),
  量化"闪烁"——这是 SISR 的主要短板,VSR 的主要卖点。

## 三、实现计划(逐一)

1. `sr_methods/rrdbnet.py` —— 复用现有 RRDBNet(从 `realesrgan_infer.py` 抽出可 import)。
2. `sr_methods/cubemap_sr.py` —— equirect↔cubemap(v360 或自写映射)+ 逐面 SR(M2)。
3. `sr_methods/swinir.py` —— SwinIR 架构 + 官方 Real-SR 权重(M3)。
4. `sr_methods/vsr_realbasicvsr.py` —— Real-BasicVSR(M4,优先用官方权重;依赖不可行则记录并给离线方案)。
5. `sr_eval.py` —— 评测协议:造 LR、跑某方法、算 PSNR/SSIM/LPIPS/WS-PSNR/sharpness/runtime。
6. `sr_compare.sh` —— 编排:抽 GT 帧 → 各方法 → 汇总成表 → 写 `SR_COMPARISON.md`。

## 四、产出

- `SR_COMPARISON.md`:指标总表 + 同一 patch 的各方法并排截图 + 推荐结论。
- 把胜出方法接回 `enhance.sh` 作为新引擎选项。

## 五、结论(已实测 —— 见 `p5_sr/sr_methods/SR_COMPARISON.md`)

全部 M0–M4 已实现并在真实 027 8K 素材上跑通对比。**实测修正了部分预期**:
- M3 **SwinIR** 单帧细节恢复最强(sharpness 3×),保真损失最小 —— ✅。
- M2 **cubemap 反而更差**(本素材极点低频,cube 往返两次重采样的模糊 > 修畸变收益)—— ✗ 预期。
- M4 **Real-BasicVSR 全面胜出**(PSNR +1.12dB、时间闪烁 −19%)—— ✅,视频成片首选。
- 轻退化下 **lanczos 保真最高**;Real-ESRGAN 适合重度退化而非本轻退化场景。

最终推荐:**视频成片走 M4(VSR);单帧要细节走 M3(SwinIR);cubemap 仅极点细节丰富时用。**
详见 `SR_COMPARISON.md` 三张指标表与复现脚本。
