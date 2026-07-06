#!/usr/bin/env python3
"""Assumption-free quaternion basis solve.
v_pano = A M(q_i) B v_world. Up-constraint per frame: A M_i u' = n_i where
u'=B@up_world (constant) and n_i = pano-frame up from the §5.2 horizon fit
(sky/ground boundary -> tilted great circle; spread~0.005 validated).
Alternate solve A (polar/Procrustes) and u' (mean). B = minimal rotation
up->u' (remaining world-yaw dof is a harmless global). Writes gps2 json.
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

# --- per-frame pano-frame up n_i from horizon fit (as §5.2) ---
def horizon_up(img_path, sw=1024, sh=512):
    g=np.asarray(Image.open(img_path).convert("L").resize((sw,sh)),np.float32)
    lons=(np.arange(sw)+0.5)/sw*2*np.pi-np.pi
    r0,r1=int(0.12*sh),int(0.75*sh); dpx=4
    grad=g[r0+dpx:r1-dpx]-g[r0+2*dpx:r1]
    rows=np.argmax(grad,axis=0)+r0+dpx; stren=np.max(grad,axis=0)
    keep=stren>max(8.0,np.percentile(stren,40))
    if keep.sum()<20: return None
    lon_k=lons[keep]; lat_k=np.pi/2-(rows[keep]+0.5)/sh*np.pi
    Amat=np.stack([np.sin(lon_k),np.tan(np.clip(lat_k,-1.4,1.4)),np.cos(lon_k)],1)
    _,_,Vt=np.linalg.svd(Amat,full_matrices=False); n=Vt[-1]
    if n[1]<0: n=-n
    return n/np.linalg.norm(n)

DUR=380.55; N=240; fps=N/DUR
PANO=f"{ROOT}/data/8kpano/scenes/scene_023cal_pano/panoramas"
ns={}
for k in range(0,N,2):     # every 2nd frame is plenty
    n=horizon_up(f"{PANO}/pano_{k+1:04d}.jpg")
    if n is not None:
        ns[k]=n
keys=sorted(ns); print(f"horizon normals from {len(keys)} frames")

# --- alternate solve A, u' (with clock-offset sweep) ---
def polar(M):
    U,_,Vt=np.linalg.svd(M); R=U@Vt
    if np.linalg.det(R)<0: U[:,-1]*=-1; R=U@Vt
    return R
up=np.array([0,1,0.0])
def solve(dt):
    Ms={k:q2R(QQ[np.clip(np.searchsorted(QT,k/fps+dt),0,len(QQ)-1)]) for k in keys}
    u=up.copy(); A=np.eye(3)
    for it in range(120):
        A=polar(sum(np.outer(ns[k],Ms[k]@u) for k in keys))
        v=sum(Ms[k].T@A.T@ns[k] for k in keys); u=v/np.linalg.norm(v)
    res=np.array([math.degrees(math.acos(np.clip(float(ns[k]@(A@Ms[k]@u)),-1,1))) for k in keys])
    return float(np.median(res)),A,u,res
best=None
for dt in np.arange(-3.0,3.01,0.5):
    m,A_,u_,res_=solve(dt)
    print(f"  dt={dt:+.1f}s -> up residual median={m:.2f}deg")
    if best is None or m<best[0]: best=(m,dt,A_,u_,res_)
m,DT,A,u,res=best
print(f"BEST dt={DT:+.1f}s: median={m:.2f}deg p90={np.percentile(res,90):.2f}deg")
qi=np.clip(np.searchsorted(QT,np.arange(N)/fps+DT),0,len(QQ)-1)
# B: minimal rotation up->u
c=np.cross(up,u); dt=float(up@u)
if np.linalg.norm(c)<1e-8: B=np.eye(3) if dt>0 else np.diag([1,-1,-1.0])
else:
    vx=np.array([[0,-c[2],c[1]],[c[2],0,-c[0]],[-c[1],c[0],0]])
    B=np.eye(3)+vx+vx@vx*(1/(1+dt))
B=B.T if False else B
# note: need B s.t. B@up=u -> rotation up->u is exactly this
print("check B@up=u:",np.round(B@up-u,4).tolist())

# --- write json + init ---
ft=np.arange(N)/fps
C=np.stack([np.interp(ft,grt,gE),np.interp(ft,grt,gU),np.interp(ft,grt,gN)],1)  # world=(E,U,N), y=up
tmpl=json.load(open(f"{ROOT}/p3_pano/pano_cams_scene_023cal.json"))
byidx={c['idx']:c for c in tmpl['cameras']}
cams=[]
for k in range(N):
    Rwp=A@q2R(QQ[qi[k]])@B
    c0=C[k]; T=-Rwp@c0
    src=byidx.get(k+1,{})
    cams.append({"idx":k+1,"image":src.get("image",f"pano_{k+1:04d}.jpg"),
                 "R_wp":Rwp.tolist(),"T":T.tolist(),"C":c0.tolist(),
                 "n_crops":src.get("n_crops",1),"ref_yaw":src.get("ref_yaw",0.0),"ref_pitch":src.get("ref_pitch",0.0)})
out=dict(tmpl); out['cameras']=cams
cext=np.array([c['C'] for c in cams]); out['cameras_extent']=float(np.linalg.norm(cext-cext.mean(0),axis=1).max()*1.1)
from plyfile import PlyData,PlyElement
lo,hi=cext.min(0),cext.max(0); n=200000
xyz=np.stack([np.random.uniform(lo[0]-120,hi[0]+120,n),
              np.random.uniform(lo[1]-60,hi[1]+10,n),
              np.random.uniform(lo[2]-120,hi[2]+120,n)],1).astype(np.float32)
verts=np.empty(n,dtype=[("x","f4"),("y","f4"),("z","f4"),("red","u1"),("green","u1"),("blue","u1")])
verts["x"],verts["y"],verts["z"]=xyz[:,0],xyz[:,1],xyz[:,2]
verts["red"]=verts["green"]=verts["blue"]=160
PlyData([PlyElement.describe(verts,"vertex")]).write(f"{ROOT}/p3_pano/gps_init_023cal.ply")
out['point_cloud']="/w/p3_pano/gps_init_023cal.ply"
json.dump(out,open(f"{ROOT}/p3_pano/pano_cams_scene_023cal_gps2.json","w"),indent=1)
np.save(f"{ROOT}/p3_pano/quat_basis_AB_cal.npy",np.stack([A,B]))
span=(cext.max(0)-cext.min(0))
print(f"A=\n{np.round(A,3)}\nu'={np.round(u,3).tolist()}")
print(f"wrote pano_cams_scene_023cal_gps2.json span={np.round(span,1).tolist()}m extent={out['cameras_extent']:.1f}")
