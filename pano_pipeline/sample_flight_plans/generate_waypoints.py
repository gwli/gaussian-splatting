#!/usr/bin/env python3
"""
Generate flight waypoint files (KML + CSV) for 3DGS-friendly drone capture.

3 flight patterns:
  A) orbital - Triple-altitude orbital around a subject (best for objects/buildings)
  B) grid    - Lawnmower pattern at 2 altitudes (best for large areas)
  C) linear  - Two-way sweep along a linear target (best for streets/boardwalks)

Usage:
    # Mode A: orbital around a subject
    python generate_waypoints.py orbital \
        --center 31.2304,121.4737 --radius 15 \
        --altitudes 5,10,15 --speed 1.5 \
        --output my_subject

    # Mode B: grid scan
    python generate_waypoints.py grid \
        --center 31.2304,121.4737 --size 100x60 \
        --altitudes 30,45 --spacing 20 --speed 2.5 \
        --output my_area

    # Mode C: linear sweep
    python generate_waypoints.py linear \
        --start 31.2300,121.4730 --end 31.2310,121.4740 \
        --offset 8 --altitudes 5,15 --speed 1.5 \
        --output my_boardwalk
"""

import argparse
import math
import os
from datetime import datetime


EARTH_RADIUS = 6378137.0


def meters_to_latlon(center_lat, center_lon, dx_m, dy_m):
    """Convert local meter offsets to (lat, lon) given a center point."""
    dlat = dy_m / EARTH_RADIUS * (180 / math.pi)
    dlon = dx_m / (EARTH_RADIUS * math.cos(math.radians(center_lat))) * (180 / math.pi)
    return center_lat + dlat, center_lon + dlon


def write_kml(path, name, description, layers):
    """layers = [(layer_name, [(lon, lat, alt), ...]), ...]"""
    body = []
    body.append('<?xml version="1.0" encoding="UTF-8"?>')
    body.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    body.append('  <Document>')
    body.append(f'    <name>{name}</name>')
    body.append(f'    <description>{description}</description>')
    body.append('    <Style id="path">')
    body.append('      <LineStyle><color>ff00ffff</color><width>3</width></LineStyle>')
    body.append('    </Style>')

    for layer_name, coords in layers:
        body.append('    <Placemark>')
        body.append(f'      <name>{layer_name}</name>')
        body.append('      <styleUrl>#path</styleUrl>')
        body.append('      <LineString>')
        body.append('        <altitudeMode>relativeToGround</altitudeMode>')
        body.append('        <tessellate>1</tessellate>')
        body.append('        <coordinates>')
        for lon, lat, alt in coords:
            body.append(f'          {lon:.7f},{lat:.7f},{alt:.1f}')
        body.append('        </coordinates>')
        body.append('      </LineString>')
        body.append('    </Placemark>')

    body.append('  </Document>')
    body.append('</kml>')
    with open(path, 'w') as f:
        f.write('\n'.join(body))


def write_csv(path, header, rows):
    """Simple csv writer (waypoint table for human review / manual entry)."""
    with open(path, 'w') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(str(c) for c in r) + '\n')


def gen_orbital(args):
    """Mode A: 3-layer orbital around subject."""
    center_lat, center_lon = args.center
    radius = args.radius
    altitudes = args.altitudes
    n_points = args.points_per_orbit

    layers = []
    csv_rows = []
    for alt in altitudes:
        coords = []
        for i in range(n_points + 1):  # +1 to close loop
            angle = 2 * math.pi * i / n_points
            dx = radius * math.cos(angle)
            dy = radius * math.sin(angle)
            lat, lon = meters_to_latlon(center_lat, center_lon, dx, dy)
            coords.append((lon, lat, alt))
            csv_rows.append([f'L{alt}m', i, lat, lon, alt, args.speed])
        layers.append((f'Layer @ {alt}m AGL', coords))

    desc = (f'3-Layer Orbital Pattern (Mode A).\n'
            f'Center: {center_lat:.6f},{center_lon:.6f}\n'
            f'Radius: {radius}m | Altitudes: {altitudes} | Speed: {args.speed}m/s\n'
            f'Generated: {datetime.now().isoformat()}')

    write_kml(f'{args.output}.kml', f'Orbital Scan - {args.output}', desc, layers)
    write_csv(f'{args.output}.csv',
              ['layer', 'idx', 'lat', 'lon', 'alt_m', 'speed_mps'],
              csv_rows)
    print(f'Generated {args.output}.kml and {args.output}.csv')
    print(f'  {len(altitudes)} layers x {n_points+1} waypoints = {len(csv_rows)} total')
    est_time = sum(2 * math.pi * radius / args.speed for _ in altitudes) / 60
    print(f'  Est. flight time: {est_time:.1f} min')


def gen_grid(args):
    """Mode B: Lawnmower grid scan, 2 altitudes."""
    center_lat, center_lon = args.center
    w_m, h_m = args.size
    spacing = args.spacing
    altitudes = args.altitudes

    n_rows = int(h_m / spacing) + 1

    layers = []
    csv_rows = []
    for alt in altitudes:
        coords = []
        for row in range(n_rows):
            y = -h_m/2 + row * spacing
            x_start, x_end = (-w_m/2, w_m/2) if row % 2 == 0 else (w_m/2, -w_m/2)
            # Add start point
            lat, lon = meters_to_latlon(center_lat, center_lon, x_start, y)
            coords.append((lon, lat, alt))
            csv_rows.append([f'L{alt}m', f'row{row}_start', lat, lon, alt, args.speed])
            # Add end point
            lat, lon = meters_to_latlon(center_lat, center_lon, x_end, y)
            coords.append((lon, lat, alt))
            csv_rows.append([f'L{alt}m', f'row{row}_end', lat, lon, alt, args.speed])
        layers.append((f'Grid @ {alt}m AGL', coords))

    desc = (f'Lawnmower Grid Scan (Mode B).\n'
            f'Center: {center_lat:.6f},{center_lon:.6f}\n'
            f'Area: {w_m}x{h_m}m | Spacing: {spacing}m | Altitudes: {altitudes} | Speed: {args.speed}m/s\n'
            f'Generated: {datetime.now().isoformat()}')

    write_kml(f'{args.output}.kml', f'Grid Scan - {args.output}', desc, layers)
    write_csv(f'{args.output}.csv',
              ['layer', 'waypoint', 'lat', 'lon', 'alt_m', 'speed_mps'],
              csv_rows)
    print(f'Generated {args.output}.kml and {args.output}.csv')
    print(f'  {len(altitudes)} altitudes x {n_rows} rows = {len(csv_rows)} waypoints')
    total_dist = sum(n_rows * w_m + (n_rows-1) * spacing for _ in altitudes)
    est_time = total_dist / args.speed / 60
    print(f'  Est. flight time: {est_time:.1f} min')


def gen_linear(args):
    """Mode C: Two-way linear sweep with parallel offset."""
    start_lat, start_lon = args.start
    end_lat, end_lon = args.end
    offset = args.offset
    altitudes = args.altitudes

    # Compute perpendicular direction (in local meters)
    dlat = (end_lat - start_lat) * EARTH_RADIUS * math.pi / 180
    dlon = (end_lon - start_lon) * EARTH_RADIUS * math.cos(math.radians(start_lat)) * math.pi / 180
    length = math.sqrt(dlat**2 + dlon**2)
    # Unit perpendicular vector
    if length > 0:
        ux, uy = -dlat/length, dlon/length  # 90° rotation
    else:
        ux, uy = 1, 0

    layers = []
    csv_rows = []
    pass_count = 0
    for alt in altitudes:
        for side, direction in [(1, 'A'), (-1, 'B')]:
            # Move start/end perpendicular by offset
            dx_perp = ux * offset * side
            dy_perp = uy * offset * side
            s_lat, s_lon = meters_to_latlon(start_lat, start_lon, dx_perp, dy_perp)
            e_lat, e_lon = meters_to_latlon(end_lat, end_lon, dx_perp, dy_perp)
            if side == -1:  # Reverse direction on B side
                s_lat, s_lon, e_lat, e_lon = e_lat, e_lon, s_lat, s_lon

            coords = [(s_lon, s_lat, alt), (e_lon, e_lat, alt)]
            layers.append((f'Sweep {direction} @ {alt}m', coords))
            csv_rows.append([f'L{alt}m_{direction}', 'start', s_lat, s_lon, alt, args.speed])
            csv_rows.append([f'L{alt}m_{direction}', 'end', e_lat, e_lon, alt, args.speed])
            pass_count += 1

    desc = (f'Two-Way Linear Sweep (Mode C).\n'
            f'Start: {start_lat:.6f},{start_lon:.6f}\n'
            f'End:   {end_lat:.6f},{end_lon:.6f}\n'
            f'Length: {length:.1f}m | Offset: {offset}m | Altitudes: {altitudes} | Speed: {args.speed}m/s\n'
            f'Generated: {datetime.now().isoformat()}')

    write_kml(f'{args.output}.kml', f'Linear Sweep - {args.output}', desc, layers)
    write_csv(f'{args.output}.csv',
              ['layer', 'waypoint', 'lat', 'lon', 'alt_m', 'speed_mps'],
              csv_rows)
    print(f'Generated {args.output}.kml and {args.output}.csv')
    print(f'  {pass_count} passes ({len(altitudes)} altitudes x 2 sides) | length {length:.1f}m')
    est_time = pass_count * length / args.speed / 60
    print(f'  Est. flight time: {est_time:.1f} min')


def parse_coords(s):
    """Parse 'lat,lon' string."""
    parts = s.split(',')
    return float(parts[0]), float(parts[1])


def parse_size(s):
    """Parse 'wxh' string."""
    parts = s.lower().split('x')
    return float(parts[0]), float(parts[1])


def parse_list(s):
    return [float(x) for x in s.split(',')]


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)

    pa = sub.add_parser('orbital', help='Mode A: 3-layer orbital around subject')
    pa.add_argument('--center', type=parse_coords, required=True, help='lat,lon')
    pa.add_argument('--radius', type=float, default=15, help='meters (default 15)')
    pa.add_argument('--altitudes', type=parse_list, default=[5, 10, 15], help='comma-sep meters')
    pa.add_argument('--speed', type=float, default=1.5, help='m/s')
    pa.add_argument('--points-per-orbit', type=int, default=24)
    pa.add_argument('--output', default='orbital', help='output filename prefix')

    pb = sub.add_parser('grid', help='Mode B: lawnmower grid scan')
    pb.add_argument('--center', type=parse_coords, required=True, help='lat,lon')
    pb.add_argument('--size', type=parse_size, required=True, help='WxH in meters')
    pb.add_argument('--spacing', type=float, default=20, help='row spacing meters')
    pb.add_argument('--altitudes', type=parse_list, default=[30, 45])
    pb.add_argument('--speed', type=float, default=2.5)
    pb.add_argument('--output', default='grid')

    pc = sub.add_parser('linear', help='Mode C: linear two-way sweep')
    pc.add_argument('--start', type=parse_coords, required=True, help='lat,lon')
    pc.add_argument('--end', type=parse_coords, required=True, help='lat,lon')
    pc.add_argument('--offset', type=float, default=8, help='perpendicular offset meters')
    pc.add_argument('--altitudes', type=parse_list, default=[5, 15])
    pc.add_argument('--speed', type=float, default=1.5)
    pc.add_argument('--output', default='linear')

    args = parser.parse_args()

    if args.mode == 'orbital':
        gen_orbital(args)
    elif args.mode == 'grid':
        gen_grid(args)
    elif args.mode == 'linear':
        gen_linear(args)


if __name__ == '__main__':
    main()
