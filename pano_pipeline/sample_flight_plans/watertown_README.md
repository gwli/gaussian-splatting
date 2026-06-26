# 水乡古镇 360 采集航点套件（地标高度 15 m）

为 3DGS 可重建质量设计的成套航线（公式见 `../FLIGHT_PLANNING.md §二·五`）。
**坐标是占位符 `30.9000,120.5000`（江南示意）——飞前替换成真实场地中心。**
源 `.insv` 无 GPS，故这里给的是相对几何；用 DJI/影翎 APP 导入 KML 后平移到实际位置即可。

## 套件构成

| 文件 | 模式 | 关键参数 | 作用 | 航时 |
|---|---|---|---|---|
| `watertown_grid.{kml,csv}` | 蛇形栅格 | D=20m, 行距 4.3m, 双高度 20/30m AGL, 5 m/s | 全镇整体覆盖（屋顶+街巷上下文） | ~36 min |
| `watertown_orbit.{kml,csv}` | 环绕 | R=15m, 4 高度 5/10/15/20m, 1.5 m/s, 36 点/圈 | 单个地标精拍（塔/庙/桥） | ~4 min |
| `watertown_canal.{kml,csv}` | 直线双向 | 长 ~218m, ±12m 偏置, 双高度 8/15m, 1.5 m/s | 沿运河两岸立面 | ~10 min |

总计约 **50 min → 3 块电池**（Antigravity A1 单电 ~20 min）。每块电池飞一个高度层/一段,落地换电后**从重叠处接着飞**。

## 为什么这样设计（量化依据）

- **D（距离）**:栅格 D=20m → GSD 3.3cm（整体上下文够用）；地标环绕 R=15m → **GSD 2.4cm**（厘米级细节）。想更细就把 `--radius` 降到 10–12m。
- **视差**:环绕每 10° 一帧 → **B/D ≈ 0.17**（三角化角 ~10°,深度可解）；栅格行距 4.3m=0.21D（横向视差）；这是之前单次航拍**最缺**的。
- **多高度**:栅格 2 层、环绕 4 层 → 提供 Z 轴视差。
- **速度**:1.5–5 m/s,快门 ≥1/500s → 运动模糊 <1px。
- **闭环**:每个圆环飞 360°+（生成时已含闭合点）；栅格/运河两段**首尾与相邻段重叠**,务必保留。

## 飞行铁律（否则前功尽弃）

1. 起飞后**先爬到目标高度再进入航线**;起飞/降落画面**不要**混入采集段(单独录或丢弃)。
2. **匀速、不悬停、不纯垂直**(悬停=零基线,垂直=零视差)。
3. 地标至少飞 **420°**(多 60° 重叠)保证首尾匹配。
4. 每个地标都跑一套 `orbit`(复制本文件、改 `--center` 到各地标)。
5. 顺光、风速 <5 m/s、快门固定 ≥1/500s、关闭机内强锐化/HDR。

## 重新生成 / 调参

```bash
# 整体栅格(改场地中心/范围/目标距离)
python ../sample_flight_plans/generate_waypoints.py plan \
  --center <lat,lon> --size 150x150 --subject-dist 20 --output watertown_grid

# 某地标环绕(逐个地标改 center;想更细把 radius 调到 10–12)
python generate_waypoints.py orbital --center <lat,lon> \
  --radius 15 --altitudes 5,10,15,20 --speed 1.5 --points-per-orbit 36 --output landmarkN

# 运河直线(改 start/end 为运河两端)
python generate_waypoints.py linear --start <lat,lon> --end <lat,lon> \
  --offset 12 --altitudes 8,15 --speed 1.5 --output canalN
```

采回的数据按 `prep_pano.sh`(stitch 4K + VGGT)→ `train_pano_gsplat_sph.py`(2048×1024)
流程跑;有了真实视差,3DGS 才能重建出清晰、可自由漫游的古镇。
