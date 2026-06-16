#!/usr/bin/env python3
"""task3 stage-1: identify & classify a consumer 360 source video.

Handles the two target devices:
  - Insta360 X3 / Antigravity A1 raw `.insv`  -> dual-fisheye (2 video streams,
    each square WxW HEVC) + an INS.Subtitle metadata track.
  - Already-stitched equirect mp4/mov         -> single 2:1 stream.

Emits a JSON profile describing layout/res/fps/codec/bitrate/colorspace and a
recommended ingest plan (stitch dual-fisheye -> equirect, or pass-through).

Usage:
  probe360.py <input> [out_profile.json]            # runs ffprobe itself (needs ffprobe on PATH)
  probe360.py --json <ffprobe.json> <name> [out]    # parse pre-extracted ffprobe JSON (no ffprobe needed)
The second form lets the orchestrator run ffprobe in the ffmpeg container and
parse on the host (where python3 lives but ffprobe may not).
"""
import sys, json, subprocess, os

def ffprobe(path):
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", path]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)

def classify(info, path):
    streams = info.get("streams", [])
    vids = [s for s in streams if s.get("codec_type") == "video"]
    subs = [s for s in streams if s.get("codec_type") == "subtitle"]
    fmt = info.get("format", {})

    def fps_of(s):
        r = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"
        try:
            a, b = r.split("/"); return round(float(a) / float(b), 3) if float(b) else 0.0
        except Exception:
            return 0.0

    prof = {
        "input": os.path.basename(path),
        "container": fmt.get("format_name", "?"),
        "duration_s": round(float(fmt.get("duration", 0) or 0), 3),
        "size_bytes": int(fmt.get("size", 0) or 0),
        "n_video_streams": len(vids),
        "n_subtitle_streams": len(subs),
        "subtitle_handlers": [s.get("tags", {}).get("handler_name", "") for s in subs],
        "video": [],
    }
    for s in vids:
        w, h = int(s.get("width", 0)), int(s.get("height", 0))
        prof["video"].append({
            "index": s.get("index"), "codec": s.get("codec_name"),
            "w": w, "h": h, "fps": fps_of(s),
            "bitrate_kbps": round(int(s.get("bit_rate", 0) or 0) / 1000) if s.get("bit_rate") else None,
            "pix_fmt": s.get("pix_fmt"), "color_space": s.get("color_space"),
            "color_primaries": s.get("color_primaries"),
            "color_transfer": s.get("color_transfer"),
            "color_range": s.get("color_range"),
        })

    # ---- layout decision ----
    v0 = prof["video"][0] if prof["video"] else {"w": 0, "h": 0}
    w, h = v0["w"], v0["h"]
    ar = (w / h) if h else 0
    n = len(vids)
    insta_meta = any("INS" in (s.get("tags", {}).get("handler_name", "") or "")
                     for s in subs + vids)

    if n >= 2 and abs(ar - 1.0) < 0.05:
        layout, ingest = "dual_fisheye", "stitch"          # Insta360/Antigravity raw
    elif n == 1 and abs(ar - 2.0) < 0.1:
        layout, ingest = "equirect", "passthrough"          # already stitched 2:1
    elif n == 1 and abs(ar - 1.0) < 0.05 and w > 2000:
        layout, ingest = "single_fisheye_or_dualfit", "stitch"
    else:
        layout, ingest = "unknown", "passthrough"

    # stitched equirect resolution: dual square WxW -> 2W wide x W tall (8K class)
    if layout == "dual_fisheye":
        eq_w, eq_h = 2 * w, w
    elif layout == "equirect":
        eq_w, eq_h = w, h
    else:
        eq_w, eq_h = (2 * h if h else w), h
    # round to even
    eq_w, eq_h = eq_w - eq_w % 2, eq_h - eq_h % 2

    device = "insta360_or_antigravity_insv" if (insta_meta and layout == "dual_fisheye") \
        else ("equirect_stitched" if layout == "equirect" else "unknown")

    prof.update({
        "layout": layout, "device_guess": device, "ingest": ingest,
        "has_insta_metadata": insta_meta,
        "equirect_out": {"w": eq_w, "h": eq_h, "note": "8K-class" if eq_w >= 7000 else
                         ("4K-class" if eq_w >= 3000 else "SD")},
        "fps": v0.get("fps", 0) if prof["video"] else 0,
    })
    return prof

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    if sys.argv[1] == "--json":
        info = json.load(open(sys.argv[2]))
        src = sys.argv[3]
        out_arg = sys.argv[4] if len(sys.argv) > 4 else None
    else:
        src = sys.argv[1]
        info = ffprobe(src)
        out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    prof = classify(info, src)
    txt = json.dumps(prof, indent=2)
    print(txt)
    if out_arg:
        with open(out_arg, "w") as f:
            f.write(txt)
        print(f"\n[probe360] wrote {out_arg}", file=sys.stderr)
