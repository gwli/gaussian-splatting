#!/usr/bin/env python3
"""Solve the A1 quaternion basis: find signed axis-permutation P (24 proper
rotations), component order and transpose such that R_wp = P R(q) P^T is
~pure-yaw for the (mostly level-flying) drone: it must keep the up axis fixed
(small roll/pitch, like the ~8 deg §5.2 residual). Then check yaw against GPS
track over fast frames to resolve heading sign. Prints ranked candidates.
"""
import struct, os, math, itertools
import numpy as np

ROOT="/w"
F=f"{ROOT}/data/8kpano/VID_20260326_073432_023.insv"
S=os.path.getsize(F); xs=S-15672335
fh=open(F,"rb")
# quats
fh.seek(xs+3537641); d=fh.read(410652)
QT,QQ=[],[]
for i in range(0,len(d)-35,36):
    QT.append(struct.unpack_from("<Q",d,i)[0]/1000.0)
    QQ.append(struct.unpack_from("<ffff",d,i+20))
QT=np.array(QT)-QT[0] if QT else QT; QQ=np.array(QQ)
# gps for track
fh.seek(xs+7624073); d=fh.read(12614)
G=[]
for i in range(0,len(d)-52,53):
    ts=struct.unpack_from("<Q",d,i)[0]+struct.unpack_from("<H",d,i+8)[0]/1000
    lat=struct.unpack_from("<d",d,i+11)[0]; lon=struct.unpack_from("<d",d,i+20)[0]
    G.append((ts,lat,lon))
G.sort(); t0=G[0][0]; lat0,lon0=G[0][1],G[0][2]
mlat=111320.0; mlon=111320.0*math.cos(math.radians(lat0))
grt=np.array([g[0]-t0 for g in G])
gE=np.array([(g[2]-lon0)*mlon for g in G]); gN=np.array([(g[1]-lat0)*mlat for g in G])
dE,dN=np.diff(gE),np.diff(gN); spd=np.hypot(dE,dN)/np.diff(grt)
bear=np.degrees(np.arctan2(dE,dN))%360   # motion bearing (compass, ENU)
fast=np.where(spd>2.0)[0]
print(f"fast gps segs (>2m/s): {len(fast)}")

DUR=380.55; N=240; fps=N/DUR
ft=np.arange(N)/fps
qi=np.clip(np.searchsorted(QT,ft),0,len(QQ)-1)

def q2R(q,order):
    if order=="xyzw": x,y,z,w=q
    else: w,x,y,z=q
    n=math.sqrt(x*x+y*y+z*z+w*w)+1e-9; x,y,z,w=x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

# all 24 proper signed permutation matrices
perms=[]
for p in itertools.permutations(range(3)):
    for s in itertools.product((1,-1),repeat=3):
        M=np.zeros((3,3))
        for r,(c,sg) in enumerate(zip(p,s)): M[r,c]=sg
        if abs(np.linalg.det(M)-1)<1e-6: perms.append(M)
print(f"{len(perms)} proper perms")

up=np.array([0,1,0.0])
frames=np.arange(0,N,5)
Rs={o:[q2R(QQ[qi[k]],o) for k in frames] for o in ("xyzw","wxyz")}
res=[]
for o in ("xyzw","wxyz"):
    for tr in (0,1):
        base=[R.T if tr else R for R in Rs[o]]
        for pi,P in enumerate(perms):
            # up-preservation error of R_wp = P R P^T  (want R_wp @ up ~ up)
            errs=[abs(1.0-float(up@(P@R@P.T@up))) for R in base]
            res.append((float(np.mean(errs)),o,tr,pi))
res.sort()
print("top-6 by up-preservation (want ~0, cos(8deg)~0.99 -> err~0.01):")
for e,o,tr,pi in res[:6]:
    print(f"  err={e:.4f} order={o} T={tr} perm#{pi}\n   P={perms[pi].astype(int).tolist()}")

# for top-3, check yaw vs GPS track on fast frames
def yaw_of(Rwp):
    # camera forward in world = column of R_wp^T; pano forward = +Z
    fwd=Rwp.T@np.array([0,0,1.0])
    return math.degrees(math.atan2(fwd[0],fwd[1]))%360
print("\nheading check (fast frames):")
gps_bear_t=grt[:-1][fast]; gps_bear=bear[fast]
for e,o,tr,pi in res[:3]:
    P=perms[pi]
    hd=[]
    for tsel in gps_bear_t:
        k=int(np.clip(round(tsel*fps),0,N-1))
        R=q2R(QQ[qi[k]],o); R=R.T if tr else R
        hd.append(yaw_of(P@R@P.T))
    hd=np.array(hd)
    for sign,lab in ((1,"+"),(-1,"-")):
        c=float(np.cos(np.radians(sign*hd-gps_bear)).mean())
        print(f"  perm#{pi} {o} T={tr}: cos({lab}yaw vs track)={c:.3f}")
