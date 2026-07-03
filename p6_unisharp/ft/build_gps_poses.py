import struct,os,json,math,numpy as np
ROOT="/w"
F=f"{ROOT}/data/8kpano/VID_20260326_073432_023.insv"; S=os.path.getsize(F)
ES=15672335; xs=S-ES; f=open(F,"rb")
# ---- GPS (id7 @7624073,12614,53B): ts,lat,lon,alt,track ----
f.seek(xs+7624073); d=f.read(12614); GP=[]
for i in range(0,len(d)-52,53):
    ts=struct.unpack_from("<Q",d,i)[0]+struct.unpack_from("<H",d,i+8)[0]/1000
    lat=struct.unpack_from("<d",d,i+11)[0]; lon=struct.unpack_from("<d",d,i+20)[0]
    trk=struct.unpack_from("<d",d,i+37)[0]; alt=struct.unpack_from("<d",d,i+45)[0]
    GP.append((ts,lat,lon,alt,trk))
GP.sort(); gt0=GP[0][0]; lat0,lon0,alt0=GP[0][1],GP[0][2],GP[0][3]
mlat=111320.0; mlon=111320.0*math.cos(math.radians(lat0))
grt=np.array([g[0]-gt0 for g in GP])
gE=np.array([(g[2]-lon0)*mlon for g in GP]); gN=np.array([(g[1]-lat0)*mlat for g in GP])
gU=np.array([g[3]-alt0 for g in GP]); gTrk=np.array([g[4] for g in GP])
# ---- quats (id14 @3537641,410652,36B: u64 ts + 3 f32 + 4 f32 quat) ----
f.seek(xs+3537641); d=f.read(410652); QT=[]; QQ=[]
for i in range(0,len(d)-35,36):
    ts=struct.unpack_from("<Q",d,i)[0]/1000.0
    q=struct.unpack_from("<ffff",d,i+20)   # last 4 f32
    QT.append(ts); QQ.append(q)
QT=np.array(QT); QQ=np.array(QQ); QT-=QT[0]
print(f"GPS rel range {grt[-1]:.1f}s ({len(GP)} samp) | quat rel range {QT[-1]:.1f}s ({len(QQ)} samp)")
# ---- frame times ----
DUR=380.55; N=240; fps=N/DUR; ft=np.array([k/fps for k in range(N)])
E=np.interp(ft,grt,gE); Nn=np.interp(ft,grt,gN); U=np.interp(ft,grt,gU)
C=np.stack([E,Nn,U],1)
qi=np.clip(np.searchsorted(QT,ft),0,len(QT)-1)
Q=QQ[qi]  # (240,4) per frame
def quat2R(q,order):
    if order=="xyzw": x,y,z,w=q
    else: w,x,y,z=q
    n=math.sqrt(x*x+y*y+z*z+w*w)+1e-9; x,y,z,w=x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
# ---- validate: heading from R vs GPS track, over moving frames ----
mv=np.linalg.norm(np.diff(C,axis=0),axis=1); moving=np.where(mv>0.5)[0]
def bearing_from_R(R,direction):
    Rw = R.T if direction=="bw" else R           # ensure world->body -> body axis in world = R.T? we test both
    # camera/body forward assumed +Z (or -Z); take body Z axis expressed in world = Rww[:,2]
    Rww = R if direction=="wb" else R.T
    fwd = Rww[:,2]  # body +Z in world
    return math.degrees(math.atan2(fwd[0],fwd[1]))%360  # ENU: atan2(E,N)=compass bearing
best=None
for order in ("xyzw","wxyz"):
  for direction in ("wb","bw"):
    for axis in (2,0,1):
      hd=[]
      for k in moving:
        R=quat2R(Q[k],order); Rww=R if direction=="wb" else R.T
        fwd=Rww[:,axis]; hd.append(math.degrees(math.atan2(fwd[0],fwd[1]))%360)
      hd=np.array(hd); trk=gTrk[np.clip(np.searchsorted(grt,ft[moving]),0,len(grt)-1)]
      # circular correlation via cos of diff
      diff=np.radians(hd-trk); score=np.cos(diff).mean()
      if best is None or score>best[0]: best=(score,order,direction,axis)
print(f"best heading match: cos={best[0]:.3f} order={best[1]} dir={best[2]} axis={best[3]}")
sc,order,direction,axis=best
# ---- build pano_cams_gps.json ----
tmpl=json.load(open(f"{ROOT}/p3_pano/pano_cams_scene_023hf.json"))
byidx={c['idx']:c for c in tmpl['cameras']}
cams=[]
for k in range(N):
    R=quat2R(Q[k],order); Rwp=R if direction=="wb" else R.T
    c=C[k]; T=(-Rwp@c)
    src=byidx.get(k+1,{})
    cams.append({"idx":k+1,"image":src.get("image",f"pano_{k+1:04d}.jpg"),
                 "R_wp":Rwp.tolist(),"T":T.tolist(),"C":c.tolist(),
                 "n_crops":src.get("n_crops",1),"ref_yaw":src.get("ref_yaw",0.0),"ref_pitch":src.get("ref_pitch",0.0)})
out=dict(tmpl); out['cameras']=cams
cext=np.array([c['C'] for c in cams]); out['cameras_extent']=float(np.linalg.norm(cext-cext.mean(0),axis=1).max()*1.1)
json.dump(out,open(f"{ROOT}/p3_pano/pano_cams_scene_023hf_gps.json","w"),indent=1)
print(f"wrote pano_cams_scene_023hf_gps.json  centers span {cext.ptp(0).round(1).tolist()} m  extent={out['cameras_extent']:.1f}")
