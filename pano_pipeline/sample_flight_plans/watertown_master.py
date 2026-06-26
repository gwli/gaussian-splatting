#!/usr/bin/env python3
"""Combine the water-town set (grid + landmark orbit + canal) into ONE master KML
with labeled folders, per-phase colours, and a takeoff/home marker — for a single
APP import. Reuses generate_waypoints geometry. Coords are placeholders; the whole
plan shifts rigidly when you move the HOME point in the APP.
  python watertown_master.py [out.kml]
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_waypoints import meters_to_latlon, compute_plan

OUT = sys.argv[1] if len(sys.argv) > 1 else "watertown_master.kml"

# --- site (placeholder Jiangnan center; replace by moving HOME in the APP) ---
HOME = (30.9000, 120.5000)
GRID_SIZE = (150.0, 150.0)          # W x H meters
LANDMARKS = [                        # (name, lat, lon)  add one per tower/temple/bridge
    ("Landmark-1", 30.9003, 120.5005),
]
CANAL = ((30.8995, 120.4995), (30.9008, 120.5012))   # (start, end)

# --- phase 1: grid (physics-planned) ---
p = compute_plan(D=20, tri_deg=12, shutter=500)
S, ALTS = p["line_spacing"], p["altitudes"]
W, H = GRID_SIZE
nrows = max(2, int(H / S) + 1)
grid_layers = []
for alt in ALTS:
    coords = []
    for r in range(nrows):
        y = -H/2 + r * S
        x0, x1 = (-W/2, W/2) if r % 2 == 0 else (W/2, -W/2)
        for x in (x0, x1):
            lat, lon = meters_to_latlon(HOME[0], HOME[1], x, y)
            coords.append((lon, lat, alt))
    grid_layers.append((f"Grid {alt:.0f}m AGL  (D=20m GSD~3.3cm spacing {S}m)", coords))

# --- phase 2: landmark orbits R=15m, 4 altitudes ---
orbit_layers = []
for (nm, la, lo) in LANDMARKS:
    for alt in (5, 10, 15, 20):
        coords = []
        for i in range(37):                     # 360°+ (closes loop)
            a = 2 * math.pi * i / 36
            lat, lon = meters_to_latlon(la, lo, 15 * math.cos(a), 15 * math.sin(a))
            coords.append((lon, lat, alt))
        orbit_layers.append((f"{nm} orbit {alt}m  (R=15m GSD~2.4cm)", coords))

# --- phase 3: canal two-way, ±12m, 2 altitudes ---
(s_la, s_lo), (e_la, e_lo) = CANAL
dN = (e_la - s_la); dE = (e_lo - s_lo)
import math as _m
norm = _m.hypot(dN, dE) or 1.0
ux, uy = -dN / norm, dE / norm                  # perpendicular (deg space, approx)
canal_layers = []
for alt in (8, 15):
    for side, tag in ((1, "A"), (-1, "B")):
        # offset both endpoints by 12 m perpendicular to the canal
        s2 = meters_to_latlon(s_la, s_lo, side * 12 * uy, side * 12 * ux)
        e2 = meters_to_latlon(e_la, e_lo, side * 12 * uy, side * 12 * ux)
        seg = [(s2[1], s2[0], alt), (e2[1], e2[0], alt)]
        if side < 0: seg = seg[::-1]
        canal_layers.append((f"Canal {tag} {alt}m (±12m)", seg))

STYLES = {"grid": "ffff00ff", "orbit": "ffffff00", "canal": "ffff00ff",  # placeholder
          "g": "ff00ffff", "o": "ffffaa00", "c": "ffff55ff"}

def kml():
    L = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
         '<name>Water-town 360 capture (master)</name>',
         '<description>Grid + landmark orbit + canal. HOME is placeholder — move it to '
         'the real site; the whole plan follows. Fly phases in order; exclude takeoff/landing.</description>',
         '<Style id="g"><LineStyle><color>ff00ffff</color><width>3</width></LineStyle></Style>',
         '<Style id="o"><LineStyle><color>ffffaa00</color><width>3</width></LineStyle></Style>',
         '<Style id="c"><LineStyle><color>ffff55ff</color><width>4</width></LineStyle></Style>',
         '<Style id="home"><IconStyle><color>ff0000ff</color><scale>1.3</scale></IconStyle></Style>']
    # home marker
    L += ['<Placemark><name>HOME / Takeoff (move me to real site)</name><styleUrl>#home</styleUrl>',
          f'<Point><coordinates>{HOME[1]:.7f},{HOME[0]:.7f},0</coordinates></Point></Placemark>']
    def folder(title, layers, sid):
        out = [f'<Folder><name>{title}</name>']
        for nm, coords in layers:
            out += ['<Placemark>', f'<name>{nm}</name>', f'<styleUrl>#{sid}</styleUrl>',
                    '<LineString><altitudeMode>relativeToGround</altitudeMode><tessellate>1</tessellate>',
                    '<coordinates>']
            out += [f'{lo:.7f},{la:.7f},{al:.1f}' for lo, la, al in coords]
            out += ['</coordinates></LineString></Placemark>']
        out += ['</Folder>']
        return out
    L += folder("Phase 1 — Grid (overall, ~36min)", grid_layers, "g")
    L += folder("Phase 2 — Landmark orbits (R=15m, ~4min each)", orbit_layers, "o")
    L += folder("Phase 3 — Canal sweep (~10min)", canal_layers, "c")
    L += ['</Document></kml>']
    return "\n".join(L)

with open(OUT, "w") as f:
    f.write(kml())
nseg = len(grid_layers) + len(orbit_layers) + len(canal_layers)
print(f"wrote {OUT}: 3 folders, {nseg} labeled segments "
      f"(grid {len(grid_layers)} / orbit {len(orbit_layers)} / canal {len(canal_layers)}) + HOME marker")
