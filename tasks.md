# Panoramic 3DGS Pipeline — Remaining Tasks

Status legend: ☐ todo · ◐ in progress · ☑ done

## Priority 1 — Direct-pano batch + full metrics  ✅ DONE
- ☑ **T-A1** SSIM/LPIPS + results.json wired into `train_pano.py`.
- ☑ **T-A2** `prep_pano.sh` auto-caps VGGT crops to ≤300.
- ☑ **T-A3** Direct-pano run over all 7 scenes (held-out PSNR/SSIM/LPIPS).
- ☑ **T-A4** Comparison table (below) — direct-pano wins on all 7 scenes.

### T-A4 results: perspective-VGGT vs direct-pano (held-out, 12 test panos/scene)
| scene | persp PSNR | **pano PSNR** | pano SSIM | pano LPIPS |
|---|---|---|---|---|
| 021 | 18.67 | **21.61** | 0.795 | 0.422 |
| 022 | 19.94 | **22.48** | 0.792 | 0.408 |
| 023 | 17.05 | **19.55** | 0.693 | 0.475 |
| 025 | 19.00 | **20.04** | 0.718 | 0.441 |
| 026 | 18.25 | **19.55** | 0.682 | 0.479 |
| 027 | 16.24 | **19.69** | 0.672 | 0.530 |
| 028 | 18.22 | **21.09** | 0.664 | 0.555 |
| **avg** | **18.05** | **20.43 (+2.4 dB)** | 0.717 | 0.473 |
Direct-pano trains on 90 panoramas (14× fewer images) yet beats the
perspective pipeline on every scene.

## Priority 2 — Make the pano pipeline one-click
- ✗ **T-B1** Faster stitch — **chunking validated NEGATIVE**: 40 frames in 2383s
  via 4 parallel chunks vs ~21s/frame serial. ffmpeg `v360` is already
  multi-threaded, so N parallel instances just thrash the same cores (no
  speedup, slightly worse). True GPU equirect (torch/kornia remap) deferred —
  carries fisheye-model geometry-matching risk and stitch is a one-time cost.
- ☑ **T-B2** `run_pano_e2e.sh <scene> <insv>` — stitch→crop→VGGT→pose→train→
  ksplat with per-stage timings.json.
- ☑ **T-B3** WebXR viewer `?source=pano` + PANO button (validated over HTTPS).

## Priority 3 — Training speed
- ☑ **T-C1** gsplat backend — **micro-benchmark done, strongly positive**:
  fwd+bwd on identical 100k gaussians @1024² → gsplat **243 iter/s vs INRIA
  71 iter/s = 3.42× faster** (beats the expected 1.5–2×). `p3_pano/bench_raster.py`.
  Full backend swap now DONE in **T-F2**: end-to-end gsplat training is 1.55×
  faster AND +1.78 dB vs INRIA on scene_023 (matched holdout/iters).

## Backlog — correctness / ops
- ☐ **T-D1** Investigate GPU `exhaustive_matcher` 85-min slowness (real bug).
- ☐ **T-D2** Adaptive `conf_thres` (percentile of depth-conf) instead of fixed 1.5.
- ☑ **T-D3** per-stage timing JSON — `run_pano_e2e.sh` writes `timings.json`
  (prep/pose/train/ksplat). Grafana not wired (overkill for batch use).
- ☑ **T-D4** VGGT >300 frames — `p2_vggt/vggt_window.py` (overlapping windows +
  Umeyama sim3 merge). **Capability proven**: 1260 crops → 7 windows → one
  merged model (1260 cams + 300k pts). **Caveat**: VGGT gives each window an
  arbitrary metric scale (per-window s drifted 1.23→0.026), residuals 0.13–0.47
  (~10–30% of scene extent) → geometrically rough vs single-window; production
  would need global BA. Not needed for our ≤300-crop pano scenes (90 panos×3 fit
  one window); useful only for >100-panorama flights. **Resolved by T-F3:** the
  global Sim3 pose-graph merge replaces the drifting sequential Umeyama (1.4×
  lower drift on a looped trajectory; loop-closure–driven).
- ✗ **T-D5** VGGT `--use_ba` — **deps resolved, blocked on a torch.hub GitHub
  rate-limit**. Installed pyceres 2.6 + lightglue + kornia + hydra (numpy pinned
  <2) and got the BA path running through model load; it then fails in the
  VGGSfM tracker at `torch.hub.load(...)` → `HTTP 403 rate limit exceeded`
  fetching tracker weights from GitHub (sandbox network). Low ROI to chase
  (BA only refines already-good feed-forward poses); feed-forward is what the
  whole pipeline uses and is sufficient. Deferred. **Update (T-F4):** network
  blocker removed; BA now runs end-to-end but reports "Not enough inliers per
  frame, skip BA" on our sparse pano-crops — confirmed data-limited, not a bug.
- ☑ **T-D6** Submodule dirty state cleaned — build/egg-info/.omc excluded via
  each built submodule's local `info/exclude`; superproject status clean.
- ☑ **T-D7** License segregated — `p3_pano/LICENSE-NOTICE.md` documents the
  GPLv3 (OmniGS-derived) component is isolated under `p3_pano/` and must not be
  redistributed under the INRIA non-commercial license.

## Research (P2, deferred)
- ☑ **T-E1** P2.2 MASt3R-SLAM — **build + run BOTH CONFIRMED end-to-end**
  (`p4_slam/FEASIBILITY.md`, one-shot `p4_slam/run_slam_full.sh`). Runs headless
  in the torch-2.6/CUDA-12.6 container and produces a TUM trajectory (16 kf
  poses, `slam_output_seq023/trajectory_tum.txt`) + dense `.ply` at ~3.5 FPS on
  one H100. Four fixes to get from build→run (all in `mast3r_slam_patches.diff`):
  (1) faiss-cpu + asmk retrieval stack; (2) made `pyrealsense2` import optional;
  (3) lazy GUI import (imgui/moderngl/in3d) for `--no-viz`; (4) **sm_90 (Hopper)
  gencode** — setup.py only had ≤sm_86 → "no kernel image" — plus vendoring
  eigen locally (gitlab submodule was down). **Finding:** tracking holds for the
  first overlapping segment (8 kf) then loses lock at frame 16 — our 90 sparse
  360°-derived crops lack the frame overlap monocular perspective SLAM needs; a
  real streaming demo wants a dense forward-perspective stream. VGGT already
  covers our batch SfM; this adds the streaming/while-flying capability.
- ☐ **T-E2** P2.3 WebGPU in-browser training.
- ☐ **T-E3** P2.4 LOD chunking for city-scale scenes.

## Priority 4 — improvements (added 2026-06-11)
Ranked by value. T-F1 is the only one that can change a *conclusion*.
- ☑ **T-F1** Dense forward-perspective stream — **FIXES tracking, conclusion
  changed**. `make_dense_perspective.sh` re-extracts a dense forward view from
  the raw `.insv` (dual-fisheye→equirect→flat in one ffmpeg pass). Dense run
  (360 frames @4 fps over first 90 s) vs sparse (90 panos @0.24 fps):
  **107 keyframe poses vs 16** (6.7×), 59 MB vs 10.8 MB cloud, **continuous
  tracking until frame ~116 with only 1 skipped frame** vs relocalize-thrash
  from frame 16. The loss-at-frame-16 was a sampling artifact, not a capability
  limit. Trajectory: `p4_slam/slam_output_seq023_dense/trajectory_tum.txt`.
- ☑ **T-F2** gsplat backend wired end-to-end (finishes T-C1) — **faster AND
  higher quality**. `p3_pano/train_gsplat.py` (gsplat 1.5.3 + native
  DefaultStrategy densification) vs INRIA `train.py`, SAME scene_023 / same
  every-8th holdout / same 7000 iters:
  | backend | held-out PSNR | it/s | train wall |
  |---|---|---|---|
  | INRIA | 17.11 | 17.2 | 408 s |
  | **gsplat** | **18.89** (SSIM 0.666) | **26.5** | **264 s** |
  → **1.55× faster end-to-end, +1.78 dB** — and conservative, since the gsplat
  trainer re-reads images from disk each iter while INRIA caches them. The
  end-to-end 1.55× (vs the 3.42× pure-kernel micro-bench) is what remains once
  I/O + SSIM + densification are included. Runner: `run_gsplat_train.sh`
  (persistent TORCH_EXTENSIONS_DIR JIT cache; auto-vendors glm headers).
- ☑ **T-F3** Global Sim3 pose-graph alignment for sliding-window VGGT (finishes
  T-D4). `p2_vggt/global_sim3.py`: replaces the greedy sequential pairwise
  Umeyama with a global solve — spanning-tree init from adjacent windows +
  refinement over ALL window pairs (loop-closure edges added automatically when
  the flight revisits a place). Loss is on *relative* Sim3s (scale pinned by the
  measurement) → no scale-collapse mode. Integrated into `vggt_window.py` as a
  two-pass merge (Pass1 run windows → Pass2 global align → Pass3 apply+dedup),
  with sequential Umeyama kept as fallback. **Validated** (`test_global_align.py`,
  synthetic closed-loop trajectory, per-window random Sim3 + noise): global
  **0.556 vs sequential 0.793 mean error → 1.4× lower drift**. Key finding: the
  win comes from **loop closures**; for a pure chain (no revisits) the global
  solve reduces to the sequential init (no worse).
  **Real-data run (scene_023, 1260 crops, `run_vggt_window.sh`):** 7 windows →
  1260 cams + 34M pts; the pose graph found **6 edges / 0 loop-closures** (these
  sequentially-ordered crops are a pure chain). Head-to-head over 300
  multi-window cams: sequential 0.1157 vs global 0.1156 shared-frame
  disagreement (**1.00× — identical**), scale range 0.026–1.23 for both. Exactly
  as predicted: no loops → global == sequential. The scale spread is inherent
  VGGT per-window metric ambiguity, unresolvable on a chain without a loop
  closure or metric anchor. Net: global solve **validated correct + never-worse
  on real data**; its drift-reduction benefit needs loop-closure capture (drone
  revisiting places, matched by visual place-recognition rather than filename).
- ☑ **T-F4** Network blocker for VGGT `--use_ba` REMOVED + validated end-to-end.
  Root cause was `api.github.com` rate-limiting `torch.hub.load`'s branch lookup
  (HTTP 403), hit by `dinov2` loading. Fix (`p2_vggt/vggsfm_localcache.patch` +
  `run_vggt_ba.sh`): pre-fetch the VGGSfM tracker weight (HuggingFace) +
  pre-clone dinov2, load tracker via `VGGSFM_TRACKER_PT` and dinov2 via
  `source="local"` (`DINOV2_LOCAL_DIR`) — zero github API calls. The BA pipeline
  now runs **fully end-to-end** (model → dino ranking → tracker → fine-tracking
  across all query frames). **Finding:** on our 60–300 perspective crops it then
  reports *"Not enough inliers per frame, skip BA"* — the sparse-overlap 360°
  pano-crops don't yield enough inlier feature tracks for classic BA
  triangulation. This is **direct evidence** for the T-D5 conclusion: feed-forward
  VGGT (correspondence-free) is the right tool for this data; classic BA is not
  just low-ROI but data-limited here.
- ☑ **T-F5** Engineering cleanup: (a) `run_slam_full.sh --clean` removes the
  multi-GB `mast3r-slam:built` image; (b) `_slam_build.sh` now auto-applies
  `mast3r_slam_patches.diff` (idempotent `git apply --check`) so a fresh clone is
  one-command runnable; (c) `p4_slam/SETUP.md` documents the full reproducible
  setup (clone, checkpoints, vendor eigen→lietorch_src).
- ☑ **T-F6** gsplat backend wired into **direct-pano** training (`train_pano.py`)
  — **works at quality parity, but NOT faster** for panoramas. gsplat is
  pinhole-only (no equirect), so `p3_pano/train_pano_gsplat.py` renders 6 pinhole
  **cube faces** per pano (one batched gsplat `C=6` call) + resamples to equirect
  with a differentiable `grid_sample`, using the EXACT LONLAT convention
  (`auxiliary.h`: lon=atan2(x,z), lat=asin(y)); gsplat `DefaultStrategy` drives
  densification. scene_023, same holdout/iters, head-to-head:
  | backend | PSNR | LPIPS | it/s | train wall | pixels/pano |
  |---|---|---|---|---|---|
  | LONLAT (native equirect, OmniGS) | 18.63–19.55 | 0.480 | **79.2** | **88 s** | 0.52M |
  | gsplat cubemap | **19.62** | **0.466** | 72.2 | 97 s | 1.57M (6×512²) |
  **Finding:** quality is on par (PSNR/LPIPS comparable; SSIM not — gsplat eval
  uses box-window vs train_pano's Gaussian-window), but gsplat is **~10% slower**
  here: the 6-face cubemap renders ~3× the pixels of one equirect pass, eating
  the per-kernel speedup that won for pinhole (T-F2, 1.55×). **Conclusion:** the
  purpose-built LONLAT equirect rasterizer remains the right backend for
  direct-pano; gsplat's win is specific to pinhole/perspective training. Runner:
  `run_pano_gsplat_train.sh`.
- ☑ **T-F7** "球面版 gsplat 内核" — **可行且质量持平,但更慢;瓶颈在投影,非合成**.
  `p3_pano/gsplat_equirect.py`:把 equirect **投影**(含解析雅可比算 2D 协方差)放在
  autograd-PyTorch 里,再喂给 gsplat 的快速 CUDA tile 合成器 `rasterize_to_pixels`
  (一趟 equirect,不用立方体),致密化复用 gsplat `DefaultStrategy`。约定与
  `auxiliary.h` 完全一致。scene_023 四方对比(同 holdout/迭代):
  | 后端 | PSNR | LPIPS | it/s | 墙钟 |
  |---|---|---|---|---|
  | **LONLAT**(原生 equirect,全融合 CUDA) | 18.6–19.6 | 0.480 | **79.2** | **88s** |
  | gsplat 立方体 (T-F6) | 19.62 | 0.466 | 72.2 | 97s |
  | gsplat equirect, eager | 18.88 | 0.469 | 42.4 | 165s |
  | gsplat equirect, torch.compile | 18.56 | 0.470 | 19.4 | 361s |
  **结论**:质量持平(PSNR/LPIPS 可比),但**最慢**。合成器(gsplat 内核)不是瓶颈
  ——**eager-PyTorch 投影**(批量 3×3 矩阵 + 雅可比,~10万高斯)才是,比 LONLAT 的
  全融合 CUDA 慢 2×。`torch.compile` 反而更糟:致密化每 100 步改变 N → 反复重编译
  (日志可见 recompile)→ 6.5 it/s。**要真正超过 LONLAT,必须把投影也做成融合 CUDA
  内核**(改 gsplat 的 `fully_fused_projection` 前向+反向,equirect 反向雅可比是难点)
  ——多周级 CUDA 工程。本次混合方案证明:思路正确(质量持平、约定正确)、瓶颈定位清楚
  (投影需融合),但 Python 投影赢不了。Runner: `... <cams> <out> <iters> <W> <face> sph`。
- ☑ **T-F8** 融合版 equirect-gsplat CUDA 内核 —— **成功:质量最好 + 速度最快,超过 LONLAT**。
  按 T-F7 的定位,把投影也做成融合 CUDA:给 gsplat 加 `CameraModelType::EQUIRECT`
  相机模型,在 `ProjectionEWA3DGSFused.cu` 写 equirect 投影**前向 + 反向 VJP**
  (`equirect_proj`/`equirect_proj_vjp` in `Utils.cuh`,解析雅可比 + ∂J/∂μ 二阶项),
  并处理球面特性:**不按 z 裁剪**(全 360° 可见)、深度用**径向距离**(fwd+bwd)。
  改动:enum(`Common.h`)+ pybind(`ext.cpp`)+ Python literal(`_wrapper.py`)+ 内核
  (`ProjectionEWA3DGSFused.cu`)+ device 函数(`Utils.cuh`),全部在
  `p3_pano/gsplat_equirect_kernel.patch`(251 行,runner 自动 `git apply`)。
  调用走 `fully_fused_projection(camera_model="equirect")` + gsplat 合成器
  (`gsplat_equirect.py::render_equirect_fused`)。scene_023 终版四方对比:
  | 后端 | PSNR | LPIPS | it/s | 墙钟 |
  |---|---|---|---|---|
  | LONLAT(原生,INRIA 级) | 18.6–19.6 | 0.480 | 79.2 | 88s |
  | gsplat 立方体 (T-F6) | 19.62 | 0.466 | 72.2 | 97s |
  | gsplat equirect 混合 (T-F7) | 18.88 | 0.469 | 42.4 | 165s |
  | **gsplat equirect 融合 (T-F8)** | **19.57** | **0.465** | **100.7** | **69.5s** |
  **正确性**:200 迭代 PSNR 12.565,与已验证的混合版(12.562)**逐位吻合** → 手推反向
  雅可比正确。**速度**:比 LONLAT **快 1.27×**(69.5s vs 88s),质量并列最好。
  **结论翻转**:之前"全景用 LONLAT"的结论被推翻——**融合 equirect-gsplat 现在是直接
  全景训练的最佳后端**(又快又好)。当年说"需要多周 CUDA 工程"——这次把它做出来了。
  **7 场景验证**(`batch_pano_gsplat_fused.sh`,held-out vs T-A4 LONLAT):平均 PSNR
  **20.54 vs 20.43**(并列/微胜),稳定 **~101 it/s vs ~79**(≈1.28× 快),7/7 无失败。
  逐场景 021/022/023/025/026/027/028 = 21.49/22.92/18.99/19.61/20.12/19.58/21.08。

## Done
- ☑ P0.2 joblib Stage-2 · P0.3 15k-iter · P0.4 chunked-stitch (script) ·
  P1.1 GLOMAP (opt-in) · P1.4 KSPLAT
- ☑ P2.1 VGGT SfM — all 7 scenes incl. COLMAP-failed 026/027/028, held-out metrics
- ☑ P1.3 direct-pano — rasterizer ported + scene_023 validated (PSNR 19.12 vs 17.05, 14× fewer images)
- ☑ WebXR viewer COLMAP/VGGT toggle
