# 3DGS-Friendly Flight Plans

Sample waypoint files for the **Antigravity A1 8K 360°** panoramic drone.

See `../FLIGHT_PLANNING.md` for the full design rationale.

## Files

| File | Mode | Description |
|---|---|---|
| `generate_waypoints.py` | — | Generator: produce KML + CSV from CLI args |
| `mode_A_orbital.kml` | A | Hand-edited template for orbital around a subject |
| `replan_026_orbital.*` | A | Example: replace failed scene_026 vertical-ascent flight |
| `replan_027_grid.*` | B | Example: replace failed scene_027 city flyover |
| `replan_028_linear.*` | C | Example: replace failed scene_028 viewpoint sweep |

## Usage

```bash
# Mode A: orbital around a single subject (sculpture, small building)
python generate_waypoints.py orbital \
    --center LAT,LON --radius 15 \
    --altitudes 5,10,15 --speed 1.5 \
    --output my_subject

# Mode B: grid scan over a large area
python generate_waypoints.py grid \
    --center LAT,LON --size 100x60 \
    --altitudes 30,45 --spacing 20 --speed 2.5 \
    --output my_area

# Mode C: linear sweep (street, boardwalk, façade)
python generate_waypoints.py linear \
    --start LAT,LON --end LAT,LON \
    --offset 8 --altitudes 5,15 --speed 1.5 \
    --output my_corridor
```

## Importing into Antigravity APP

1. Connect drone to app, open mission planning view.
2. Import the `.kml` file. Most drone apps support standard KML.
3. If KML isn't supported directly, manually enter waypoints from the `.csv` table.
4. Verify on the satellite map that:
   - Waypoints are over the intended area
   - Altitudes are referenced to ground (AGL), not sea level (MSL)
   - Speed is set to the value in the CSV
5. **Take-off and landing points must be set OUTSIDE the scan area** (≥ 30m away). Do not include them in the mission waypoints — they should be manual.

## Required pre-flight modifications

Edit the `--center`, `--start`, `--end` coordinates to match your actual site (use Google Maps / 高德地图 to get lat,lon).

The sample uses Shanghai 浦东新区 coordinates (31.2304°N, 121.4737°E) as a placeholder — **do not fly with these unchanged**.
