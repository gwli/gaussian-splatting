# task3 — 消费级 360 相机/无人机视频 → 标准化 equirect → 超分/画质增强(技术报告)

目标(见 `../task3.md`):**支持 Insta360 X3 和 Antigravity A1 的 360° 视频输入,自动
标准化为 equirect 格式,并提供 2x 超分或同分辨率画质增强两种输出模式。**

## 一、调研:两类设备的原始格式(已用真实文件验证)

输入素材在 `../data/8kpano/*.insv`(8 个,665 MB–17 GB)。用 `ffprobe`/`exiftool`
实测:

| 项 | 实测结果 |
|---|---|
| 容器 | `mov/mp4`(`major_brand=avc1`),扩展名 `.insv`(Insta360 封装) |
| 视频流 | **两路 HEVC,各 3840×3840**(`hvc1`,Main,yuvj420p,bt709)= **双鱼眼** |
| 帧率 | 29.97 fps(30000/1001) |
| 码率 | 每路 ~84 Mbps |
| 字幕流 | 1 路 `mov_text`,handler=`INS.Subtitle`,内容是逐帧 **`F:NNNN` 对焦值** |
| 元数据 | 仅 15 个 QuickTime 标签,**无 GPS/陀螺仪遥测**(与 task2 结论一致) |
| 旁文件 | `PRX_*.prx` = Insta360 低分代理(预览用) |

**关键结论**:
- 这些 `.insv` **不是单个标准 equirect**,而是 **双鱼眼**(前/后镜头各一路方形流)。
  "8K 360°" 指的是拼接后的 equirect 分辨率:双 3840² → **7680×3840(2:1,8K 级)**。
- Insta360 X3 与影翎 Antigravity A1 都走 Insta360 生态封装(`.insv` + `.prx`),
  原始即双鱼眼;**不能默认直接喂超分**,必须先拼接成 equirect(对应 task3 "额外注意点")。
- 陀螺仪/防抖元数据是 Insta360 私有 trailer,`ffmpeg`/`exiftool` 读不到——若要用,
  需 Insta360 Studio/SDK 导出。本流水线用 ffmpeg `v360` 直接拼接,不依赖私有 SDK。

陀螺仪那一路虽读不到,但拼接所需的镜头朝向是固定的(前/后),`v360` 双鱼眼模型即可。

## 二、设计:`p5_sr/` 六阶段流水线

完全对齐 task3 "处理流程更新":

```
设备原始视频 (.insv 双鱼眼 / 已拼接 equirect)
  └─① probe360.py   格式识别:双鱼眼? equirect? 分辨率/fps/codec/码率/色彩空间
  └─② standardize.sh 标准化:双鱼眼→equirect(v360 拼接+缝合)或 equirect 直通归一化
  └─③ enhance.sh    预处理+超分:去噪/去压缩伪影(deband)/锐化  +  2x 或 同分辨率
  └─④ (③ 内重新编码:libx265 / nvenc,bt709,faststart)
  └─⑤ inject360.sh  写 360 spherical 元数据(Google spatial-media)
  └─⑥ validate.sh   发布兼容性验证:2:1、codec、pixfmt、faststart、球面元数据、可解码
run_sr.sh = ①→⑥ 编排器
```

两种输出模式(task3 最终目标):
- **`2x` 超分**:`run_sr.sh in out.mp4 2x [ffmpeg|realesrgan]`
  - `ffmpeg` 引擎:去噪→deband→**lanczos 2x**→锐化(快,8K 可行)。
  - `realesrgan` 引擎:抽帧→**RRDBNet x2/x4(自带实现,加载官方权重,分块推理)**→重编码(质量高)。
- **`enhance` 同分辨率画质增强**:`run_sr.sh in out.mp4 enhance` —— 去噪 + deband 去压缩
  伪影 + 细节锐化 + 轻微颗粒,**保持 8K 不变**(对应 task3 对 A1 8K 源"保持 8K 输出"的建议)。

### 关键实现点
- **probe 驱动**:`standardize.sh auto` 用 `probe360.py` 的 `ingest` 字段自动选
  `stitch`(双鱼眼)或 `passthrough`(已 equirect),无需人工判断设备。
- **自带 Real-ESRGAN**:`realesrgan_infer.py` 是纯 PyTorch 重写的 RRDBNet(键名对齐
  BasicSR),**不装 basicsr/realesrgan**(避开其旧 torchvision 依赖),直接加载官方
  `RealESRGAN_x2plus.pth`/`x4plus`;分块(tile)推理把 8K 帧塞进显存。
- **编码器**:默认 `libx265`(CPU)。本机 H100 是计算卡,**无 NVENC 硬件编码器**
  (`OpenEncodeSessionEx: No capable devices found`),故 `NVENC` 默认关;有 NVENC 的
  机器设 `NVENC=1` 走 `hevc_nvenc`。
- **色彩/发布**:统一 bt709 + `yuv420p` + `+faststart`(moov 前置),平台兼容。
- **360 元数据**:Google spatial-media 注入器首次运行自动 vendoring,写
  `Spherical=true / ProjectionType=equirectangular`,YouTube/VR 播放器即可识别。

## 三、验证(真实 `VID_20260326_074057_024.insv`,2s 切片)

端到端 `run_sr.sh ... enhance ffmpeg 2`(为快用 `CAP_W=1920`):

```
① probe   : layout=dual_fisheye device=insta360_or_antigravity_insv ingest=stitch
            equirect_out=7680x3840 (8K-class)  fps=29.97
② stitch  : 双鱼眼 → 1920x960 equirect HEVC, 60 帧  (抽帧确认为有效全景:广场/雕塑/天空带)
③ enhance : hqdn3d+deband+unsharp+grain → 4.5 MB
⑤ inject  : Spherical=true  ProjectionType=equirectangular  ✅
⑥ validate: 6/6 PASS — 2:1(ratio=2.000)/hevc/yuv420p/faststart/球面元数据/可解码
```

各模式单测:
- **2x ffmpeg**:1920×960 → **3840×1920** HEVC ✅
- **realesrgan x2**:抽帧→下载官方 `RealESRGAN_x2plus.pth`→自带 RRDBNet 推理→重编码
  (见 `验证记录` 节)。
- **同分辨率 enhance**:保持输入分辨率,去噪/去伪影/锐化 ✅

> 关于全分辨率成本:`nlmeans` 去噪在 1920×960 仅 0.34 fps(60 帧 174 s),对 8K 不可行,
> 故默认 `hqdn3d`(快);`DENOISE=nlmeans` 为高质量可选项。8K 全片建议后台批处理。

## 四、用法

```bash
# 一键:任意 .insv/equirect → 发布级 360 mp4
bash p5_sr/run_sr.sh <input> <out.mp4> [2x|enhance] [ffmpeg|realesrgan] [secs] [ss]

# 例:8K 同分辨率画质增强(保持 7680x3840)
bash p5_sr/run_sr.sh data/8kpano/VID_..._024.insv out/enh.mp4 enhance ffmpeg

# 例:2x 模型超分(Real-ESRGAN)
SCALE=2 TILE=512 bash p5_sr/run_sr.sh data/8kpano/VID_..._024.insv out/sr.mp4 2x realesrgan

# 分阶段也可单独调用 probe360.py / standardize.sh / enhance.sh / inject360.sh / validate.sh
```

env:`CAP_W`(限制输出宽,默认 7680)`IFOV`(鱼眼视场,默认 200)`SCALE`(2|4)
`TILE`/`FP16`(模型分块/半精)`CRF` `NVENC` `DENOISE`(hqdn3d|nlmeans|none)`MODEL`(权重 URL)。

## 五、已知限制
- `.insv` 双鱼眼缝合用 `v360` 平均混合,极点/缝合处有轻微叠影;若需无缝可接 Insta360
  SDK 的光流缝合(私有,未集成)。
- Real-ESRGAN 全 8K 逐帧很慢(需分块 + 时间),适合离线批处理;实时/快速场景用 ffmpeg 引擎。
- 本机 H100 无 NVENC,编码走 CPU x265;有 NVENC 的机器 `NVENC=1` 可大幅加速重编码。
