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
  solve reduces to the sequential init (no worse). A full 1260-crop VGGT re-run
  is the optional end-use (the >100-panorama case from T-D4).
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

## Done
- ☑ P0.2 joblib Stage-2 · P0.3 15k-iter · P0.4 chunked-stitch (script) ·
  P1.1 GLOMAP (opt-in) · P1.4 KSPLAT
- ☑ P2.1 VGGT SfM — all 7 scenes incl. COLMAP-failed 026/027/028, held-out metrics
- ☑ P1.3 direct-pano — rasterizer ported + scene_023 validated (PSNR 19.12 vs 17.05, 14× fewer images)
- ☑ WebXR viewer COLMAP/VGGT toggle
