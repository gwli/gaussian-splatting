#!/usr/bin/env python3
"""Generalized hybrid GPS+IMU pose builder for any A1 scene.
Auto-parses the trailer offset table (validated: 10B entries chain contiguously),
decodes GPS (id7) + quats (id14), applies the mount-constant basis A,B
(quat_basis_AB.npy, solved on 023 — same rigid mount for all scenes), snaps
per-frame tilt to the horizon fit, writes pano_cams_<scene>_gps3.json + metric init.

  gps_hybrid_scene.py --scene scene_023hf --insv <file> --dur <sec> [--n 240]
"""
import argparse, struct, os, math, json
import numpy as np
from PIL import Image

MAGIC=b"8db42d694ccc418790edff439fe026bf"
ap=argparse.ArgumentParser()
ap.add_argument("--scene",required=True); ap.add_argument("--insv",required=True)
ap.add_argument("--dur",type=float,required=True); ap.add_argument("--n",type=int,default=240)
ap.add_argument("--root",default="/w")
a=ap.parse_args(); ROOT=a.root

S=os.path.getsize(a.insv); f=open(a.insv,"rb")
f.seek(S-72); hdr=f.read(72); assert hdr[40:72]==MAGIC
ES=struct.unpack("<I",hdr[32:36])[0]; xs=S-ES
f.seek(S-72-4096); tail=f.read(4096)
cand={}
for off in range(len(tail)-10):
    rid,fmt=tail[off],tail[off+1]
    sz,ro=struct.unpack("<II",tail[off+2:off+10])
    if 1<=rid<=18 and fmt<=2 and 0<sz<ES and ro+sz<=ES:
        cand.setdefault(rid,[]).append((ro,sz))
bounds=set()
for lst in cand.values():
    for ro,sz in lst: bounds.add(ro); bounds.add(ro+sz)
tab={}
for rid,lst in cand.items():
    pick=max(lst,key=lambda e:((e[0]+e[1]) in bounds)+(e[0] in bounds))
    tab[rid]=(xs+pick[0],pick[1])
assert 7 in tab and 14 in tab, f"records found: {sorted(tab)}"

off,ln=tab[7]; f.seek(off); d=f.read(ln); G=[]
for i in range(0,len(d)-52,53):
    ts=struct.unpack_from("<Q",d,i)[0]+struct.unpack_from("<H",d,i+8)[0]/1000
    if chr(d[i+10])!='A': continue
    lat=struct.unpack_from("<d",d,i+11)[0]; lon=struct.unpack_from("<d",d,i+20)[0]
    if chr(d[i+19])=='S': lat=-abs(lat)
    if chr(d[i+28])=='W': lon=-abs(lon)
    G.append((ts,lat,lon,struct.unpack_from("<d",d,i+45)[0]))
G.sort(); assert len(G)>10, "too few GPS fixes"
t0,lat0,lon0,alt0=G[0]
mlat=111320.0; mlon=111320.0*math.cos(math.radians(lat0))
grt=np.array([g[0]-t0 for g in G])
gE=np.array([(g[2]-lon0)*mlon for g in G]); gN=np.array([(g[1]-lat0)*mlat for g in G]); gU=np.array([g[3]-alt0 for g in G])

off,ln=tab[14]; f.seek(off); d=f.read(ln); QT,QQ=[],[]
for i in range(0,len(d)-35,36):
    QT.append(struct.unpack_from("<Q",d,i)[0]/1000.0); QQ.append(struct.unpack_from("<ffff",d,i+20))
QT=np.array(QT); QT-=QT[0]; QQ=np.array(QQ)

def q2R(q):
    x,y,z,w=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
A,B=np.load(f"{ROOT}/p3_pano/quat_basis_AB.npy")

def horizon_up(p,sw=1024,sh=512):
    g=np.asarray(Image.open(p).convert("L").resize((sw,sh)),np.float32)
    lons=(np.arange(sw)+0.5)/sw*2*np.pi-np.pi
    r0,r1=int(0.12*sh),int(0.75*sh); dpx=4
    grad=g[r0+dpx:r1-dpx]-g[r0+2*dpx:r1]
    rows=np.argmax(grad,axis=0)+r0+dpx; st=np.max(grad,axis=0)
    keep=st>max(8.0,np.percentile(st,40))
    if keep.sum()<20: return None
    lk=lons[keep]; la=np.pi/2-(rows[keep]+0.5)/sh*np.pi
    M=np.stack([np.sin(lk),np.tan(np.clip(la,-1.4,1.4)),np.cos(lk)],1)
    _,_,Vt=np.linalg.svd(M,full_matrices=False); n=Vt[-1]
    return (n if n[1]>=0 else -n)/np.linalg.norm(n)
def rot_ab(x,y):
    x=x/np.linalg.norm(x); y=y/np.linalg.norm(y)
    v=np.cross(x,y); c=float(x@y)
    if np.linalg.norm(v)<1e-9: return np.eye(3) if c>0 else np.diag([1,-1,-1.0])
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+vx+vx@vx*(1/(1+c))

N=a.n; fps=N/a.dur; ft=np.arange(N)/fps
C=np.stack([np.interp(ft,grt,gE),np.interp(ft,grt,gU),np.interp(ft,grt,gN)],1)
qi=np.clip(np.searchsorted(QT,ft),0,len(QQ)-1)
PANO=f"{ROOT}/data/8kpano/scenes/{a.scene}_pano/panoramas"
tmpl=json.load(open(f"{ROOT}/p3_pano/pano_cams_{a.scene}.json"))
byidx={c['idx']:c for c in tmpl['cameras']}
up=np.array([0,1,0.0]); cams=[]; nfit=0
for k in range(N):
    R=A@q2R(QQ[qi[k]])@B
    n=horizon_up(f"{PANO}/pano_{k+1:04d}.jpg")
    if n is not None: R=rot_ab(R@up,n)@R; nfit+=1
    c0=C[k]; src=byidx.get(k+1,{})
    cams.append({"idx":k+1,"image":src.get("image",f"pano_{k+1:04d}.jpg"),
                 "R_wp":R.tolist(),"T":(-R@c0).tolist(),"C":c0.tolist(),
                 "n_crops":src.get("n_crops",1),"ref_yaw":src.get("ref_yaw",0.0),"ref_pitch":src.get("ref_pitch",0.0)})
out=dict(tmpl); out['cameras']=cams
ce=np.array([c['C'] for c in cams]); out['cameras_extent']=float(np.linalg.norm(ce-ce.mean(0),axis=1).max()*1.1)
from plyfile import PlyData,PlyElement
lo,hi=ce.min(0),ce.max(0); n=200000
xyz=np.stack([np.random.uniform(lo[0]-120,hi[0]+120,n),np.random.uniform(lo[1]-60,hi[1]+10,n),
              np.random.uniform(lo[2]-120,hi[2]+120,n)],1).astype(np.float32)
verts=np.empty(n,dtype=[("x","f4"),("y","f4"),("z","f4"),("red","u1"),("green","u1"),("blue","u1")])
verts["x"],verts["y"],verts["z"]=xyz[:,0],xyz[:,1],xyz[:,2]
verts["red"]=verts["green"]=verts["blue"]=160
ply=f"{ROOT}/p3_pano/gps_init_{a.scene}.ply"
PlyData([PlyElement.describe(verts,"vertex")]).write(ply); out['point_cloud']=ply
oj=f"{ROOT}/p3_pano/pano_cams_{a.scene}_gps3.json"
json.dump(out,open(oj,"w"),indent=1)
span=(ce.max(0)-ce.min(0))
print(f"[{a.scene}] gps={len(G)} fits={nfit}/{N} span={np.round(span,1).tolist()}m -> {oj}")
