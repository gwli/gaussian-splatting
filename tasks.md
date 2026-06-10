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
- ☐ **T-C1** P1.2: evaluate gsplat as the training backend (1.5–2× faster).

## Backlog — correctness / ops
- ☐ **T-D1** Investigate GPU `exhaustive_matcher` 85-min slowness (real bug).
- ☐ **T-D2** Adaptive `conf_thres` (percentile of depth-conf) instead of fixed 1.5.
- ☐ **T-D3** per-stage timing JSON + optional Prometheus/Grafana.
- ☐ **T-D4** VGGT >300 frames: sliding-window / chunked attention for big scenes.
- ☐ **T-D5** VGGT `--use_ba` (LightGlue+pyceres) path for higher accuracy.
- ☐ **T-D6** Clean `submodules/simple-knn` dirty state.
- ☐ **T-D7** License: OmniGS GPLv3 mixed into repo — document/segregate clearly.

## Research (P2, deferred)
- ☐ **T-E1** P2.2 streaming/SLAM reconstruction (MASt3R-SLAM).
- ☐ **T-E2** P2.3 WebGPU in-browser training.
- ☐ **T-E3** P2.4 LOD chunking for city-scale scenes.

## Done
- ☑ P0.2 joblib Stage-2 · P0.3 15k-iter · P0.4 chunked-stitch (script) ·
  P1.1 GLOMAP (opt-in) · P1.4 KSPLAT
- ☑ P2.1 VGGT SfM — all 7 scenes incl. COLMAP-failed 026/027/028, held-out metrics
- ☑ P1.3 direct-pano — rasterizer ported + scene_023 validated (PSNR 19.12 vs 17.05, 14× fewer images)
- ☑ WebXR viewer COLMAP/VGGT toggle
