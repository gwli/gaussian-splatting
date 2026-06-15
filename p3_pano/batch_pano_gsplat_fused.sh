#!/bin/bash
# T-F8 validation: run the fused equirect-gsplat kernel on all 7 pano scenes,
# held-out eval (every 8th pano). gsplat CUDA is JIT-cached, so each scene is fast.
set -e
ROOT=/raid/git/gaussian-splatting
RUN=$ROOT/p3_pano/run_pano_gsplat_train.sh
LOG=$ROOT/data/8kpano
for s in 021 022 023 025 026 027 028; do
  echo "=================== scene_$s (fused T-F8) ==================="
  GSPLAT_EQUIRECT_FUSED=1 bash "$RUN" \
    p3_pano/pano_cams_scene_$s.json \
    data/8kpano/scenes/scene_${s}_pano/output_pano_fused \
    7000 1024 512 sph > "$LOG/pano_fused_$s.log" 2>&1 || echo "  scene_$s FAILED"
  grep -aE '\[EVAL\]' "$LOG/pano_fused_$s.log" | tail -1
  cp "$ROOT/data/8kpano/scenes/scene_${s}_pano/output_pano_fused/results.json" \
     "$LOG/pano_fused_${s}_result.json" 2>/dev/null || true
done
echo "=================== SUMMARY (fused T-F8) ==================="
for s in 021 022 023 025 026 027 028; do
  python3 -c "import json;d=json.load(open('$LOG/pano_fused_${s}_result.json'));print(f\"scene_$s: PSNR {d['PSNR']:.2f}  SSIM {d['SSIM']:.3f}  LPIPS {d['LPIPS']}  {d['iter_s']} it/s  {d['train_s']}s  N={d['n_gaussians']}\")" 2>/dev/null || echo "scene_$s: no result"
done
