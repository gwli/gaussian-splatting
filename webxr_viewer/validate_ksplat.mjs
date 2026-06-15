// Headless validation: parse a .ksplat with the SAME library the WebXR viewer
// uses (KSplatLoader -> SplatBuffer) and report header/count/SH-degree + decoded
// centers/color. A clean parse with the expected splat count == the viewer loads it.
//
// Usage (needs the GaussianSplats3D build present):
//   docker run --rm -v $PWD/webxr_viewer/GaussianSplats3D:/gs3d -v <ply_dir>:/in:ro \
//     -w /gs3d node:20 node /gs3d/../validate_ksplat.mjs /in/point_cloud.ksplat
// (a copy lives at GaussianSplats3D/util/ with the ../build import path.)
import fs from 'fs';
globalThis.window = globalThis; // lib uses window.setTimeout; node has setTimeout globally
import { KSplatLoader } from './GaussianSplats3D/build/gaussian-splats-3d.module.js';

const path = process.argv[2];
const buf = fs.readFileSync(path);
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const sb = await KSplatLoader.loadFromFileData(ab);
const n = sb.getSplatCount();
// sanity: read center + color of first & last splat (exercises the data sections)
const col0 = [], cObj0 = {}, cObjN = {};
sb.getSplatColor(0, { set(...a) { col0.push(...a); } });
sb.getSplatCenter(0, cObj0);          // writes .x/.y/.z
sb.getSplatCenter(n - 1, cObjN);
const c0 = [cObj0.x, cObj0.y, cObj0.z], cN = [cObjN.x, cObjN.y, cObjN.z];
console.log(JSON.stringify({
  file: path,
  bytes: buf.length,
  splatCount: n,
  maxSplatCount: sb.getMaxSplatCount(),
  minSHDegree: sb.getMinSphericalHarmonicsDegree(),
  firstCenter: c0.map(v => +v.toFixed(3)),
  firstColorRGBA: col0,
  lastCenter: cN.map(v => +v.toFixed(3)),
  // eslint-disable-next-line
  parsed_ok: n > 0 && isFinite(c0[0]) && isFinite(cN[0]),
}, null, 2));
