#!/usr/bin/env python3
"""Hybrid poses (gps3): tilt from per-frame horizon fit (accurate), yaw from the
solved-basis quaternion R_i = A M_i B. Correction: R'_i = R_min(R_i@up -> n_i) @ R_i
(minimal pano-frame rotation snapping predicted up to measured up; preserves yaw).
Frames with failed horizon fit keep R_i. Writes pano_cams_scene_023hf_gps3.json.
"""
import struct, os, math, json
import numpy as np
from PIL import Image

ROOT="/w"
F=f"{ROOT}/data/8kpano/VID_20260326_073432_023.insv"
S=os.path.getsize(F); xs=S-15672335
fh=open(F,"rb")
fh.seek(xs+3537641); d=fh.read(410652)
QT,QQ=[],[]
for i in range(0,len(d)-35,36):
    QT.append(struct.unpack_from("<Q",d,i)[0]/1000.0); QQ.append(struct.unpack_from("<ffff",d,i+20))
QT=np.array(QT); QT-=QT[0]; QQ=np.array(QQ)
fh.seek(xs+7624073); d=fh.read(12614)
G=[]
for i in range(0,len(d)-52,53):
    ts=struct.unpack_from("<Q",d,i)[0]+struct.unpack_from("<H",d,i+8)[0]/1000
    lat=struct.unpack_from("<d",d,i+11)[0]; lon=struct.unpack_from("<d",d,i+20)[0]
    alt=struct.unpack_from("<d",d,i+45)[0]; G.append((ts,lat,lon,alt))
G.sort(); t0,lat0,lon0,alt0=G[0]
mlat=111320.0; mlon=111320.0*math.cos(math.radians(lat0))
grt=np.array([g[0]-t0 for g in G])
gE=np.array([(g[2]-lon0)*mlon for g in G]); gN=np.array([(g[1]-lat0)*mlat for g in G]); gU=np.array([g[3]-alt0 for g in G])

def q2R(q):
    x,y,z,w=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

A,B=np.load(f"{ROOT}/p3_pano/quat_basis_AB.npy")

def horizon_up(img_path, sw=1024, sh=512):
    g=np.asarray(Image.open(img_path).convert("L").resize((sw,sh)),np.float32)
    lons=(np.arange(sw)+0.5)/sw*2*np.pi-np.pi
    r0,r1=int(0.12*sh),int(0.75*sh); dpx=4
    grad=g[r0+dpx:r1-dpx]-g[r0+2*dpx:r1]
    rows=np.argmax(grad,axis=0)+r0+dpx; stren=np.max(grad,axis=0)
    keep=stren>max(8.0,np.percentile(stren,40))
    if keep.sum()<20: return None
    lon_k=lons[keep]; lat_k=np.pi/2-(rows[keep]+0.5)/sh*np.pi
    Am=np.stack([np.sin(lon_k),np.tan(np.clip(lat_k,-1.4,1.4)),np.cos(lon_k)],1)
    _,_,Vt=np.linalg.svd(Am,full_matrices=False); n=Vt[-1]
    if n[1]<0: n=-n
    return n/np.linalg.norm(n)

def rot_a_to_b(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b)
    v=np.cross(a,b); c=float(a@b)
    if np.linalg.norm(v)<1e-9: return np.eye(3) if c>0 else np.diag([1,-1,-1.0])
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+vx+vx@vx*(1/(1+c))

DUR=380.55; N=240; fps=N/DUR
PANO=f"{ROOT}/data/8kpano/scenes/scene_023hf_pano/panoramas"
ft=np.arange(N)/fps
qi=np.clip(np.searchsorted(QT,ft),0,len(QQ)-1)
up=np.array([0,1,0.0])
corr=[]; nfit=0
cams=[]
C=np.stack([np.interp(ft,grt,gE),np.interp(ft,grt,gU),np.interp(ft,grt,gN)],1)
tmpl=json.load(open(f"{ROOT}/p3_pano/pano_cams_scene_023hf.json"))
byidx={c['idx']:c for c in tmpl['cameras']}
for k in range(N):
    R=A@q2R(QQ[qi[k]])@B
    n=horizon_up(f"{PANO}/pano_{k+1:04d}.jpg")
    if n is not None:
        pred=R@up
        ang=math.degrees(math.acos(np.clip(float(pred@n),-1,1)))
        corr.append(ang); nfit+=1
        R=rot_a_to_b(pred,n)@R
    c0=C[k]; T=-R@c0
    src=byidx.get(k+1,{})
    cams.append({"idx":k+1,"image":src.get("image",f"pano_{k+1:04d}.jpg"),
                 "R_wp":R.tolist(),"T":T.tolist(),"C":c0.tolist(),
                 "n_crops":src.get("n_crops",1),"ref_yaw":src.get("ref_yaw",0.0),"ref_pitch":src.get("ref_pitch",0.0)})
print(f"horizon fits: {nfit}/{N}, tilt correction median={np.median(corr):.1f}deg p90={np.percentile(corr,90):.1f}deg")
out=dict(tmpl); out['cameras']=cams
cext=np.array([c['C'] for c in cams]); out['cameras_extent']=float(np.linalg.norm(cext-cext.mean(0),axis=1).max()*1.1)
out['point_cloud']="/w/p3_pano/gps_init_023v2.ply"
json.dump(out,open(f"{ROOT}/p3_pano/pano_cams_scene_023hf_gps3.json","w"),indent=1)
print("wrote pano_cams_scene_023hf_gps3.json")
