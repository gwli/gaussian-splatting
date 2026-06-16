#!/usr/bin/env python3
"""task_sr M4: temporal comparison of per-frame SISR vs video-SR (RealBasicVSR).

On a CONSECUTIVE real clip (x4 downscale-restore), compares:
  - rrdbnet-x4 : Real-ESRGAN per frame (no temporal modeling)
  - realbasicvsr : bidirectional VSR over the whole clip
Metrics per method (mean over frames):
  PSNR / SSIM / LPIPS vs GT  +  temporal warp-error (flicker; lower=stabler).
Warp-error: cv2 DIS optical flow on GT between t-1,t; warp SR_{t-1}->t; MSE to SR_t
in well-matched regions (small GT residual). This is the metric SISR loses on.

Usage:
  temporal_eval.py <gt_clip_dir> <out_json> --rrdb <x4.pth> --vsr <RealBasicVSR.pth> [--fp16]
"""
import os, sys, json, glob, math, time, argparse
import numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sr_lib, vsr_realbasicvsr as V

def psnr(a, b):
    m = np.mean((a - b) ** 2); return 100.0 if m < 1e-10 else 10 * math.log10(1.0 / m)

def warp_err(srs, gts):
    """mean temporal warp error over consecutive frames, using GT flow."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    errs = []
    for t in range(1, len(srs)):
        g0 = cv2.cvtColor((gts[t-1]*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor((gts[t]*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        flow = dis.calc(g1, g0, None)                       # warp t-1 -> t sampling
        H, W = g1.shape
        gx, gy = np.meshgrid(np.arange(W), np.arange(H))
        mx = (gx + flow[..., 0]).astype(np.float32); my = (gy + flow[..., 1]).astype(np.float32)
        warp_prev = cv2.remap((srs[t-1]*255).astype(np.float32), mx, my, cv2.INTER_LINEAR)
        warp_gtprev = cv2.remap((gts[t-1]*255).astype(np.float32), mx, my, cv2.INTER_LINEAR)
        # only count pixels where GT warps well (occlusion/flow-error mask)
        valid = (np.abs(warp_gtprev - gts[t]*255).mean(2) < 12)
        d = np.abs(warp_prev - (srs[t]*255)).mean(2)
        if valid.sum() > 1000:
            errs.append(float(d[valid].mean()))
    return float(np.mean(errs)) if errs else None

def metrics(name, srs, gts, lpips_fn):
    import piq
    ps = [psnr(g, s) for g, s in zip(gts, srs)]
    ss, lp = [], []
    for g, s in zip(gts, srs):
        gt_t = torch.from_numpy(g.transpose(2, 0, 1))[None].float()
        sr_t = torch.from_numpy(s.transpose(2, 0, 1))[None].float()
        ss.append(float(piq.ssim(sr_t, gt_t, data_range=1.0)))
        if lpips_fn is not None: lp.append(float(lpips_fn(sr_t, gt_t)))
    we = warp_err(srs, gts)
    r = {"psnr": round(np.mean(ps), 3), "ssim": round(np.mean(ss), 4),
         "lpips": round(np.mean(lp), 4) if lp else None,
         "warp_err": round(we, 3) if we else None}
    print(f"  {name:14s} psnr={r['psnr']:.2f} ssim={r['ssim']:.4f} "
          f"lpips={r['lpips']} warp_err={r['warp_err']} (lower warp=stabler)")
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gtdir"); ap.add_argument("out")
    ap.add_argument("--rrdb", required=True); ap.add_argument("--vsr", required=True)
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        import piq; lpips_fn = piq.LPIPS()
    except Exception: lpips_fn = None
    files = sorted(glob.glob(os.path.join(a.gtdir, "*.png")))
    gts = [cv2.imread(f)[:, :, ::-1].astype(np.float32) / 255.0 for f in files]
    H, W = gts[0].shape[:2]; H -= H % 4; W -= W % 4
    gts = [np.ascontiguousarray(g[:H, :W]) for g in gts]
    lrs = [cv2.resize(g, (W // 4, H // 4), interpolation=cv2.INTER_CUBIC) for g in gts]
    print(f"[temporal] {len(gts)} frames, GT {W}x{H}, LR {W//4}x{H//4}, x4")

    # per-frame RRDBNet x4
    rr, _ = sr_lib.make_model("rrdbnet", a.rrdb, 4, dev)
    t0 = time.time()
    sr_rr = [np.clip(sr_lib.run_method("rrdbnet", rr, 0, np.ascontiguousarray(lr), 4, 0, 16, dev, a.fp16)[:H, :W], 0, 1)
             for lr in lrs]
    t_rr = time.time() - t0
    # RealBasicVSR over the clip
    vm = V.load(a.vsr, dev)
    t0 = time.time()
    sr_vsr = V.run_clip(vm, [np.ascontiguousarray(lr) for lr in lrs], dev, a.fp16)
    sr_vsr = [np.clip(s[:H, :W], 0, 1).astype(np.float32) for s in sr_vsr]
    t_vsr = time.time() - t0

    print("=== temporal results (mean) ===")
    res = {"rrdbnet_x4_perframe": metrics("rrdbnet-x4", sr_rr, gts, lpips_fn),
           "realbasicvsr_x4": metrics("realbasicvsr", sr_vsr, gts, lpips_fn)}
    res["rrdbnet_x4_perframe"]["sec_total"] = round(t_rr, 2)
    res["realbasicvsr_x4"]["sec_total"] = round(t_vsr, 2)
    json.dump({"n_frames": len(gts), "results": res}, open(a.out, "w"), indent=2)
    print(f"[temporal] wrote {a.out}")

if __name__ == "__main__":
    main()
