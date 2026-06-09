# Panoramic 3DGS Pipeline — Remaining Tasks

Status legend: ☐ todo · ◐ in progress · ☑ done

## Priority 1 — Direct-pano batch + full metrics
- ☐ **T-A1** Wire SSIM/LPIPS into `train_pano.py` (currently PSNR-only): render
  held-out test views, run `metrics.py`-style PSNR/SSIM/LPIPS, write
  `results.json` per scene.
- ☐ **T-A2** `prep_pano.sh`: auto-cap VGGT crop input to ≤300 (≈3/pano with
  front crop) instead of the manual curation done for scene_023.
- ☐ **T-A3** Batch direct-pano over the other scenes (021,022,025,026,027,028);
  collect timings + held-out PSNR/SSIM/LPIPS.
- ☐ **T-A4** Comparison table: perspective-VGGT vs direct-pano across all scenes.

## Priority 2 — Make the pano pipeline one-click
- ☐ **T-B1** GPU panorama stitch (replace CPU v360 ~30min/scene): validate
  `stitch_chunked.sh` end-to-end and/or a kornia/torch GPU equirect path.
- ☐ **T-B2** Single entrypoint `run_pano_e2e.sh <scene> <insv>` chaining
  stitch→crop→VGGT→pose→train→ksplat with per-stage timing JSON.
- ☐ **T-B3** WebXR viewer: add a `pano` source/button for `*_pano` scenes.

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
