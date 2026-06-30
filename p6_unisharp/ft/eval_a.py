#!/usr/bin/env python3
"""Evaluation launcher: runs unisharp.validation.run_validation with the same two
monkeypatches train_a.py needs (offline UniK3D load + SimPanorama metre-scale
pairing), so a held-out sim scene actually yields pairs and the model loads
offline. Pass-through CLI args go straight to run_validation.

  python eval_a.py --checkpoint <ckpt> --dataset sim --data-root <ft/data>
    --sim-pose-root <ft/poses> --manifest-file <val.txt> --out-dir <dir>
"""
import os
import runpy

# --- offline UniK3D load (adapter's config_variant call fails offline) --------
import unisharp.utils.unik3d_adapter as _ad
def _offline_load_unik3d(backbone="vitl", pretrained=True, device=None, cache_root=None, **_kw):
    import torch as _t
    from unik3d.models import UniK3D as _U
    m = _U.from_pretrained(f"lpiccinelli/unik3d-{backbone}")
    if not hasattr(m, "resolution_level"):
        m.resolution_level = 0
    m.eval()
    dev = device if device is not None else _t.device("cuda" if _t.cuda.is_available() else "cpu")
    return m.to(dev)
_ad.load_unik3d_model = _offline_load_unik3d

# --- force strict=False load (pretrained has an extra 'payload.depth_alignment'
# buffer from a newer arch; validation's _load_model uses strict=True) ----------
from unisharp.models.unisharp_feature import UnisharpFeatureModel  # noqa: E402
_orig_load = UnisharpFeatureModel.load_from_checkpoint
def _load_nonstrict(self, ckpt_path, strict=False):
    m, u = _orig_load(self, ckpt_path, strict=False)
    print(f"[eval_a] load_from_checkpoint(strict=False): missing={len(m)} unexpected={len(u)}", flush=True)
    return m, u
UnisharpFeatureModel.load_from_checkpoint = _load_nonstrict

# --- SimPanorama metre-scale pairing (validation also hardcodes 0.5 / 0.01) ---
from unisharp.datasets.sim_panorama import SimPanoramaDataset  # noqa: E402
_orig_sim_init = SimPanoramaDataset.__init__
def _sim_init(self, *a, **k):
    mt = os.environ.get("SIM_PAIR_MAX_TR")
    ov = os.environ.get("SIM_PAIR_MIN_OVERLAP")
    if mt is not None and "pair_max_translation_m" in k: k["pair_max_translation_m"] = float(mt)
    if ov is not None and "pair_min_depth_overlap" in k: k["pair_min_depth_overlap"] = float(ov)
    k["position_scale"] = float(os.environ.get("SIM_POSITION_SCALE", "1.0"))
    # validation builds SimPanorama at NATIVE ERP res (no max_long_edge); the model
    # was trained/smoke-tested at 512 long-edge -> force the same to avoid the
    # full-res cubemap forward blowing up (CUDA error in the decoder pad).
    k["max_long_edge"] = int(os.environ.get("SIM_MAX_LONG_EDGE", "512"))
    _orig_sim_init(self, *a, **k)
SimPanoramaDataset.__init__ = _sim_init

runpy.run_module("unisharp.validation.run_validation", run_name="__main__")
