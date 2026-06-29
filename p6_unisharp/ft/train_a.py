#!/usr/bin/env python3
"""A-tier training launcher. The stock `unisharp.cli train-feature` has NO
resume/init flag (it trains UniSHARP heads from scratch on top of a pretrained
UniK3D) — useless for fine-tuning. This launcher applies two clean monkeypatches
(no edits to the gitignored UniSHARP clone) and then hands off to the stock CLI:

  1) after the model is built, load the pretrained UniSHARP checkpoint (INIT_CKPT)
     so we *fine-tune* from it instead of from scratch;
  2) relax SimPanorama's hardcoded pair_max_translation_m=0.5 / overlap=0.6
     (indoor defaults) to aerial-scale values via env, so our metre-scale
     baselines actually yield pairs.

Freeze policy is done purely with the CLI's per-group LRs (no patch needed):
  --unik3d-encoder-lr0/1 0  -> freeze the DINOv2 encoder
  --unik3d-lr0/1 <small>    -> gently adapt UniK3D decoder + depth head (the prior)
  --lr0/1 <normal>          -> train UniSHARP heads/composer

Usage: python train_a.py train-feature --data-root-sim ... (see train_a.sh)
"""
import os
import sys

from unisharp.models.unisharp_feature import UnisharpFeatureModel
from unisharp.datasets.sim_panorama import SimPanoramaDataset

# --- patch 1: load pretrained weights right after construction -----------------
_orig_model_init = UnisharpFeatureModel.__init__
def _model_init(self, *a, **k):
    _orig_model_init(self, *a, **k)
    ck = os.environ.get("INIT_CKPT", "").strip()
    if ck:
        missing, unexpected = self.load_from_checkpoint(ck, strict=False)
        print(f"[train_a] INIT_CKPT loaded: {ck}\n"
              f"[train_a]   missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    else:
        print("[train_a] WARNING: no INIT_CKPT — training heads FROM SCRATCH", flush=True)
UnisharpFeatureModel.__init__ = _model_init

# --- patch 1b: offline UniK3D load ---------------------------------------------
# The adapter calls hubconf UniK3D(config_variant="eval") which the official
# UniK3D clone rejects (TypeError) and then refuses to fall back when
# HF_HUB_OFFLINE=1. Bypass it with the offline-cache from_pretrained that
# pseudo_depth.py already proved works.
import unisharp.utils.unik3d_adapter as _ad  # noqa: E402
def _offline_load_unik3d(backbone="vitl", pretrained=True, device=None, cache_root=None, **_kw):
    import torch as _t
    from unik3d.models import UniK3D as _U
    m = _U.from_pretrained(f"lpiccinelli/unik3d-{backbone}")
    if not hasattr(m, "resolution_level"):
        m.resolution_level = 0
    m.eval()
    dev = device if device is not None else _t.device("cuda" if _t.cuda.is_available() else "cpu")
    return m.to(dev)
# unisharp_feature does `from ...unik3d_adapter import load_unik3d_model` INSIDE
# __init__, so the adapter module attribute is what gets re-imported — patch it.
_ad.load_unik3d_model = _offline_load_unik3d

# --- patch 2: relax SimPanorama pairing for aerial (metre) baselines -----------
_orig_sim_init = SimPanoramaDataset.__init__
def _sim_init(self, *a, **k):
    mt = os.environ.get("SIM_PAIR_MAX_TR")
    ov = os.environ.get("SIM_PAIR_MIN_OVERLAP")
    gp = os.environ.get("SIM_MAX_INDEX_GAP")
    if mt is not None and "pair_max_translation_m" in k: k["pair_max_translation_m"] = float(mt)
    if ov is not None and "pair_min_depth_overlap" in k: k["pair_min_depth_overlap"] = float(ov)
    if gp is not None and "max_index_gap" in k: k["max_index_gap"] = int(gp)
    # CRITICAL: default position_scale=0.01 shrinks our ~metre VGGT translations
    # 100x and breaks unit-consistency with the metric pseudo-depth used for pair
    # overlap -> force to 1.0 (our pose.csv is already ~metric, see task_ft.md §3.2).
    k["position_scale"] = float(os.environ.get("SIM_POSITION_SCALE", "1.0"))
    print(f"[train_a] SimPanorama: max_tr={k.get('pair_max_translation_m')} "
          f"overlap={k.get('pair_min_depth_overlap')} idx_gap={k.get('max_index_gap')} "
          f"pos_scale={k.get('position_scale')}", flush=True)
    _orig_sim_init(self, *a, **k)
SimPanoramaDataset.__init__ = _sim_init

from unisharp.cli import main_cli  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("train-feature")
    main_cli()
