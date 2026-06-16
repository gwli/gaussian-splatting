#!/usr/bin/env python3
"""task_sr: benchmark SR methods by self-supervised downscale-restore.

For each GT equirect frame: LR = bicubic down x`scale`; each method upscales LR
back to GT size; we score vs GT with PSNR / SSIM / LPIPS / WS-PSNR (ERP
latitude-weighted) + a no-reference sharpness, and record runtime.

Usage:
  sr_eval.py <gt_frames_dir> <out_json> <methods csv> \
     --scale 2 --rrdb <x2.pth> --swinir <x2.pth> [--tile 0] [--fp16] [--montage dir]
"""
import os, sys, json, time, glob, argparse, math
import numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sr_lib

def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse < 1e-10 else 10 * math.log10(1.0 / mse)

def ws_psnr(a, b):
    # ERP latitude weighting: w = cos(lat); lat varies along rows
    H = a.shape[0]
    lat = (np.arange(H) + 0.5) / H * math.pi - math.pi / 2
    w = np.cos(lat)[:, None, None]
    se = (a - b) ** 2 * w
    mse = se.sum() / (w.sum() * a.shape[1] * a.shape[2])
    return 100.0 if mse < 1e-10 else 10 * math.log10(1.0 / mse)

def sharpness(a):
    g = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gtdir"); ap.add_argument("out"); ap.add_argument("methods")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--rrdb", default=""); ap.add_argument("--swinir", default="")
    ap.add_argument("--tile", type=int, default=0); ap.add_argument("--pad", type=int, default=16)
    ap.add_argument("--fp16", action="store_true"); ap.add_argument("--montage", default="")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    try:
        import piq; have_lpips = True
        lpips_fn = piq.LPIPS()
    except Exception as e:
        print(f"[eval] piq/LPIPS unavailable ({e}); skipping LPIPS"); have_lpips = False

    # preload models
    models = {}
    for m in methods:
        w = a.rrdb if m.startswith("rrdbnet") else a.swinir if m.startswith("swinir") else ""
        models[m] = sr_lib.make_model(m, w, a.scale, dev) if w or m == "lanczos" else (None, 0)

    gts = sorted(glob.glob(os.path.join(a.gtdir, "*.png")))
    print(f"[eval] {len(gts)} GT frames, methods={methods}, scale=x{a.scale}, dev={dev}")
    if a.montage: os.makedirs(a.montage, exist_ok=True)
    agg = {m: {"psnr": [], "ws_psnr": [], "ssim": [], "lpips": [], "sharp": [], "t": []} for m in methods}

    for gi, gp in enumerate(gts):
        gt = cv2.imread(gp, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
        H, W = gt.shape[:2]
        H -= H % a.scale; W -= W % a.scale; gt = np.ascontiguousarray(gt[:H, :W])
        lr = cv2.resize(gt, (W // a.scale, H // a.scale), interpolation=cv2.INTER_CUBIC)
        panel = [(_lab(gt, "GT"))]
        for m in methods:
            model, win = models[m]
            t0 = time.time()
            sr = sr_lib.run_method(m, model, win, np.ascontiguousarray(lr), a.scale,
                                   a.tile, a.pad, dev, a.fp16)
            torch.cuda.synchronize() if dev == "cuda" else None
            dt = time.time() - t0
            sr = np.clip(sr[:H, :W], 0, 1).astype(np.float32)
            agg[m]["psnr"].append(psnr(gt, sr)); agg[m]["ws_psnr"].append(ws_psnr(gt, sr))
            agg[m]["sharp"].append(sharpness(sr)); agg[m]["t"].append(dt)
            try:
                import piq
                gt_t = torch.from_numpy(gt.transpose(2, 0, 1))[None]
                sr_t = torch.from_numpy(sr.transpose(2, 0, 1))[None]
                agg[m]["ssim"].append(float(piq.ssim(sr_t, gt_t, data_range=1.0)))
                if have_lpips:
                    agg[m]["lpips"].append(float(lpips_fn(sr_t, gt_t)))
            except Exception:
                pass
            panel.append(_lab(sr, m))
            print(f"  [{gi+1}/{len(gts)}] {m:14s} psnr={agg[m]['psnr'][-1]:.2f} "
                  f"ws={agg[m]['ws_psnr'][-1]:.2f} sharp={agg[m]['sharp'][-1]:.0f} t={dt:.2f}s")
        if a.montage:
            cw = 480
            crops = [cv2.resize(p[H//2-120:H//2+120, W//2-160:W//2+160], (cw, 360)) for p in panel]
            mont = np.concatenate(crops, axis=1)
            cv2.imwrite(os.path.join(a.montage, f"cmp_{gi:02d}.png"),
                        (mont[:, :, ::-1] * 255).astype(np.uint8))

    def mean(x): return round(float(np.mean(x)), 4) if x else None
    res = {m: {k: mean(v) for k, v in d.items()} for m, d in agg.items()}
    json.dump({"scale": a.scale, "n_frames": len(gts), "results": res}, open(a.out, "w"), indent=2)
    print("\n=== SUMMARY (mean) ===")
    hdr = f"{'method':14s} {'PSNR':>7s} {'WS-PSNR':>8s} {'SSIM':>7s} {'LPIPS':>7s} {'sharp':>7s} {'s/frame':>8s}"
    print(hdr); print("-" * len(hdr))
    for m in methods:
        r = res[m]
        print(f"{m:14s} {r['psnr']:7.2f} {r['ws_psnr']:8.2f} "
              f"{(r['ssim'] or 0):7.4f} {(r['lpips'] or 0):7.4f} {r['sharp']:7.0f} {r['t']:8.2f}")
    print(f"\n[eval] wrote {a.out}")

def _lab(img, txt):
    o = img.copy()
    cv2.putText(o, txt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (1, 1, 0), 2)
    return o

if __name__ == "__main__":
    main()
