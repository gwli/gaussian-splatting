#!/usr/bin/env python3
"""task2: beat-synced editing. Detects beats in a music WAV (self-contained numpy
beat tracker: spectral-flux onset -> autocorrelation tempo -> beat phase), then
builds an edit plan that HARD-CUTS between camera shots on the beats.

Modes:
  beat_sync.py gen-click <out.wav> <bpm> <secs>           # synth a test beat track
  beat_sync.py plan <music.wav> <segments.json> <K> <out_plan.json>
       # cut every K beats, cycling through the rendered shots; plan = [{clip,src_start,dur}]
"""
import sys, json, wave, struct, numpy as np

def read_wav(path):
    w = wave.open(path, "rb"); sr = w.getframerate(); n = w.getnframes()
    ch = w.getnchannels(); sw = w.getsampwidth()
    raw = w.readframes(n); w.close()
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dt).astype(np.float64)
    if ch > 1: x = x.reshape(-1, ch).mean(1)
    return x / (np.abs(x).max() + 1e-9), sr

def gen_click(out, bpm, secs):
    sr = 22050; t = np.arange(int(sr * secs)) / sr
    x = np.zeros_like(t)
    period = 60.0 / bpm
    for b in np.arange(0, secs, period):
        i = int(b * sr); dur = int(0.06 * sr)
        env = np.exp(-np.arange(dur) / (0.02 * sr))
        tone = np.sin(2 * np.pi * 70 * np.arange(dur) / sr) * env  # kick-ish
        x[i:i + dur] += tone[:max(0, min(dur, len(x) - i))][:len(x) - i]
    x = (x / (np.abs(x).max() + 1e-9) * 30000).astype(np.int16)
    w = wave.open(out, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(x.tobytes()); w.close()
    print(f"[beat] wrote click track {out} ({bpm} bpm, {secs}s)")

def onset_envelope(x, sr, hop=512, win=1024):
    n = 1 + (len(x) - win) // hop
    win_fn = np.hanning(win)
    mags = np.empty((n, win // 2 + 1))
    for i in range(n):
        seg = x[i * hop:i * hop + win] * win_fn
        mags[i] = np.abs(np.fft.rfft(seg))
    flux = np.maximum(0, np.diff(mags, axis=0)).sum(1)
    flux = np.concatenate([[0], flux])
    flux -= flux.mean(); flux = np.maximum(0, flux)
    return flux, sr / hop  # envelope + its frame rate

def detect_beats(x, sr):
    env, fr = onset_envelope(x, sr)
    # tempo via autocorrelation in 60..180 bpm
    ac = np.correlate(env, env, "full")[len(env) - 1:]
    lo, hi = int(fr * 60 / 180), int(fr * 60 / 60)
    lag = lo + int(np.argmax(ac[lo:hi]))
    period = lag / fr
    # phase: offset maximizing onset energy at beat grid
    best_off, best_s = 0, -1
    for off in range(lag):
        idx = np.arange(off, len(env), lag)
        s = env[idx].sum()
        if s > best_s: best_s, best_off = s, off
    beats = np.arange(best_off, len(env), lag) / fr
    bpm = 60.0 / period
    return beats, bpm

if sys.argv[1] == "gen-click":
    gen_click(sys.argv[2], float(sys.argv[3]), float(sys.argv[4])); sys.exit(0)

# plan mode:  beat_sync.py plan <music> <segments.json> <K> <out>
_, _mode, music, segj, K, out = sys.argv
K = int(K)
x, sr = read_wav(music)
beats, bpm = detect_beats(x, sr)
total = len(x) / sr
cuts = list(beats[::K])
if not cuts or cuts[0] > 0.05: cuts = [0.0] + cuts
if cuts[-1] < total - 0.1: cuts.append(total)
segs = json.load(open(segj))["segments"]; fps = json.load(open(segj))["fps"]
shot_len = [s["nframes"] / fps for s in segs]
nshots = len(segs)
plan, cursor = [], [0.0] * nshots
for j in range(len(cuts) - 1):
    dur = cuts[j + 1] - cuts[j]
    sh = j % nshots
    start = cursor[sh]
    if start + dur > shot_len[sh]:        # loop within the shot
        start = 0.0
    plan.append({"clip": sh, "src_start": round(start, 3), "dur": round(dur, 3)})
    cursor[sh] = start + dur
json.dump({"bpm": round(float(bpm), 1), "n_beats": len(beats), "cut_every": K,
           "n_segments": len(plan), "total": round(total, 2), "plan": plan},
          open(out, "w"), indent=1)
print(f"[beat] {bpm:.1f} bpm, {len(beats)} beats -> {len(plan)} beat-cut segments (every {K})")
