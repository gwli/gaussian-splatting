#!/usr/bin/env python3
"""Judge LingBot-Map poses on 023 against GPS ground truth (same Umeyama Sim3
test that measured VGGT at 102m median residual)."""
import struct, os, math, numpy as np
F="/w/data/8kpano/VID_20260326_073432_023.insv"; S=os.path.getsize(F); xs=S-15672335
f=open(F,"rb"); f.seek(xs+7624073); d=f.read(12614); G=[]
for i in range(0,len(d)-52,53):
    ts=struct.unpack_from("<Q",d,i)[0]+struct.unpack_from("<H",d,i+8)[0]/1000
    G.append((ts,struct.unpack_from("<d",d,i+11)[0],struct.unpack_from("<d",d,i+20)[0],struct.unpack_from("<d",d,i+45)[0]))
G.sort(); t0,lat0,lon0,alt0=G[0]
mlat=111320.0; mlon=111320.0*math.cos(math.radians(lat0))
grt=np.array([g[0]-t0 for g in G])
gE=np.array([(g[2]-lon0)*mlon for g in G]); gN=np.array([(g[1]-lat0)*mlat for g in G]); gU=np.array([g[3]-alt0 for g in G])
N=240; ft=np.arange(N)/(N/380.55)
ENU=np.stack([np.interp(ft,grt,gE),np.interp(ft,grt,gN),np.interp(ft,grt,gU)],1)

ex=np.load("/w/p6_unisharp/lbmap/lb_023.npz")["extrinsic"]  # (N,3,4) c2w
ex=ex.reshape(-1,3,4)[:N]
C=ex[:,:,3]
def umeyama(src,dst):
    ms,md=src.mean(0),dst.mean(0); s,dd=src-ms,dst-md
    cov=dd.T@s/len(src); U,D,Vt=np.linalg.svd(cov); Sg=np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt)<0: Sg[2,2]=-1
    R=U@Sg@Vt; sc=np.trace(np.diag(D)@Sg)/(s**2).sum()*len(src)
    return sc,R,md-sc*R@ms
sc,R,t=umeyama(C,ENU)
res=np.linalg.norm((sc*(R@C.T).T+t)-ENU,axis=1)
print(f"LingBot-Map n={len(C)} | scale={sc:.2f} (unit~{sc:.1f}m)")
print(f"Sim3 residual: median={np.median(res):.1f}m mean={res.mean():.1f}m p90={np.percentile(res,90):.1f}m")
print(f"reference: VGGT median=101.9m | GPS truth extent 370x358x108m")
