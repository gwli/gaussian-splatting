#!/usr/bin/env python3
"""Solve R_wp = A * R(q) * B with TWO constant rotations (world-frame A, body/
mount-frame B) via alternating orthogonal Procrustes. Targets: on fast GPS
segments the true pano pose ~ pure yaw at the GPS motion bearing (level flight,
forward along track). Validates residual angles + up-preservation on ALL frames,
then writes corrected pano_cams json + metric init ply.
"""
import struct, os, math, json
import numpy as np

ROOT="/w"
F=f"{ROOT}/data/8kpano/VID_20260326_073432_023.insv"
S=os.path.getsize(F); xs=S-15672335
fh=open(F,"rb")
fh.seek(xs+3537641); d=fh.read(410652)
QT,QQ=[],[]
for i in range(0,len(d)-35,36):
    QT.append(struct.unpack_from("<Q",d,i)[0]/1000.0)
    QQ.append(struct.unpack_from("<ffff",d,i+20))
QT=np.array(QT); QT-=QT[0]; QQ=np.array(QQ)
fh.seek(xs+7624073); d=fh.read(12614)
G=[]
for i in range(0,len(d)-52,53):
    ts=struct.unpack_from("<Q",d,i)[0]+struct.unpack_from("<H",d,i+8)[0]/1000
    lat=struct.unpack_from("<d",d,i+11)[0]; lon=struct.unpack_from("<d",d,i+20)[0]
    alt=struct.unpack_from("<d",d,i+45)[0]
    G.append((ts,lat,lon,alt))
G.sort(); t0=G[0][0]; lat0,lon0,alt0=G[0][1],G[0][2],G[0][3]
mlat=111320.0; mlon=111320.0*math.cos(math.radians(lat0))
grt=np.array([g[0]-t0 for g in G])
gE=np.array([(g[2]-lon0)*mlon for g in G]); gN=np.array([(g[1]-lat0)*mlat for g in G]); gU=np.array([g[3]-alt0 for g in G])
dE,dN=np.diff(gE),np.diff(gN); spd=np.hypot(dE,dN)/np.diff(grt)
bear=np.arctan2(dE,dN)   # rad, compass from North, ENU

def q2R(q,order="xyzw"):
    if order=="xyzw": x,y,z,w=q
    else: w,x,y,z=q
    n=math.sqrt(x*x+y*y+z*z+w*w)+1e-9; x,y,z,w=x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def Ry(psi):  # yaw about +Y (up), forward +Z toward bearing psi (E,N mapped x,?): fwd=[sin,0,cos]->(E,U,N)? our world axes = (E,N,U)?
    # world axes: x=E, y=N, z=U?? NO: our pano convention has y=up. world=(E,U?,...)
    pass

# Define world axes consistent with pano convention: x=East, y=Up, z=North.
# GPS ENU -> world: (E,N,U) -> (E,U,N)  i.e. C_world = [E, U, N]
def bearing_R(psi):  # camera forward (+Z) points at compass bearing psi
    c,s=math.cos(psi),math.sin(psi)
    # forward = [sin(psi) (E), 0 (up), cos(psi) (N)]; up=[0,1,0]; right=fwd x up? build R_pw (cam->world) with cols right,up,fwd
    fwd=np.array([s,0,c]); up=np.array([0,1,0.0]); right=np.cross(up,fwd)
    Rpw=np.stack([right,up,fwd],1)
    return Rpw.T   # world->pano

# fast samples -> targets
sel=np.where(spd>2.0)[0]
DUR=380.55; N=240; fps=N/DUR
Ms=[]; Ys=[]
for j in sel:
    t=(grt[j]+grt[j+1])/2
    k=int(np.clip(round(t*fps),0,N-1))
    qk=QQ[np.clip(np.searchsorted(QT,t),0,len(QQ)-1)]
    Ms.append(q2R(qk)); Ys.append(bearing_R(bear[j]))
print(f"targets from {len(Ms)} fast segments")

def polar(M):
    U,_,Vt=np.linalg.svd(M); R=U@Vt
    if np.linalg.det(R)<0: U[:,-1]*=-1; R=U@Vt
    return R
A=np.eye(3); B=np.eye(3)
for it in range(60):
    A=polar(sum(Y@B.T@M.T for M,Y in zip(Ms,Ys)))
    B=polar(sum(M.T@A.T@Y for M,Y in zip(Ms,Ys)))
res=[math.degrees(math.acos(np.clip((np.trace(Ys[i].T@(A@Ms[i]@B))-1)/2,-1,1))) for i in range(len(Ms))]
res=np.array(res)
print(f"Procrustes residual: median={np.median(res):.1f}deg p90={np.percentile(res,90):.1f}deg")
# up-preservation across ALL frames with solved A,B
ft=np.arange(N)/fps; up=np.array([0,1,0.0])
qi=np.clip(np.searchsorted(QT,ft),0,len(QQ)-1)
uperr=[math.degrees(math.acos(np.clip(float(up@(A@q2R(QQ[qi[k]])@B@up)),-1,1))) for k in range(N)]
print(f"up tilt over all frames: median={np.median(uperr):.1f}deg p90={np.percentile(uperr,90):.1f}deg (expect ~banked-flight few-deg..15deg)")

# write corrected cams json
C=np.stack([np.interp(ft,grt,gE),np.interp(ft,grt,gU),np.interp(ft,grt,gN)],1)   # world=(E,U,N)
tmpl=json.load(open(f"{ROOT}/p3_pano/pano_cams_scene_023hf.json"))
byidx={c['idx']:c for c in tmpl['cameras']}
cams=[]
for k in range(N):
    Rwp=A@q2R(QQ[qi[k]])@B
    c=C[k]; T=-Rwp@c
    src=byidx.get(k+1,{})
    cams.append({"idx":k+1,"image":src.get("image",f"pano_{k+1:04d}.jpg"),
                 "R_wp":Rwp.tolist(),"T":T.tolist(),"C":c.tolist(),
                 "n_crops":src.get("n_crops",1),"ref_yaw":src.get("ref_yaw",0.0),"ref_pitch":src.get("ref_pitch",0.0)})
out=dict(tmpl); out['cameras']=cams
cext=np.array([c['C'] for c in cams]); out['cameras_extent']=float(np.linalg.norm(cext-cext.mean(0),axis=1).max()*1.1)
# metric init ply
from plyfile import PlyData,PlyElement
lo,hi=cext.min(0),cext.max(0); n=200000
xyz=np.stack([np.random.uniform(lo[0]-120,hi[0]+120,n),
              np.random.uniform(lo[1]-60,hi[1]+10,n),
              np.random.uniform(lo[2]-120,hi[2]+120,n)],1).astype(np.float32)
verts=np.empty(n,dtype=[("x","f4"),("y","f4"),("z","f4"),("red","u1"),("green","u1"),("blue","u1")])
verts["x"],verts["y"],verts["z"]=xyz[:,0],xyz[:,1],xyz[:,2]
verts["red"]=verts["green"]=verts["blue"]=160
PlyData([PlyElement.describe(verts,"vertex")]).write(f"{ROOT}/p3_pano/gps_init_023v2.ply")
out['point_cloud']=f"/w/p3_pano/gps_init_023v2.ply"
json.dump(out,open(f"{ROOT}/p3_pano/pano_cams_scene_023hf_gps2.json","w"),indent=1)
np.save(f"{ROOT}/p3_pano/quat_basis_AB.npy",np.stack([A,B]))
print(f"A=\n{np.round(A,3)}\nB=\n{np.round(B,3)}")
print(f"wrote pano_cams_scene_023hf_gps2.json  span={np.round(cext.ptp(0),1).tolist()}m")
