#!/usr/bin/env python3
"""Photometric R_rig solve in exact COLMAP convention (up-SfM unreliable).
R_rig = Rz(g) @ Rx(180+a) @ Rz(b); coarse g scan then fine 3-dof; score = ring
correlation of down vs up reprojections averaged over 4 frames. Updates rig023.npz.
"""
import math, numpy as np, torch, torch.nn.functional as Fn
from PIL import Image

ROOT="/w"; dev="cuda" if torch.cuda.is_available() else "cpu"
z=np.load(f"{ROOT}/p3_pano/rig023.npz"); cal_d, cal_u = z["cal_d"], z["cal_u"]
DOWN=f"{ROOT}/data/8kpano/scenes/fish023/images"; UP=f"{ROOT}/data/8kpano/scenes/fish023/images_up"
TH_MAX=math.radians(100.0); H,W=256,512
vv,uu=torch.meshgrid(torch.arange(H,device=dev),torch.arange(W,device=dev),indexing="ij")
lon=(uu+0.5)/W*2*math.pi-math.pi; lat=math.pi/2-(vv+0.5)/H*math.pi
d_erp=torch.stack([torch.cos(lat)*torch.sin(lon),torch.sin(lat),torch.cos(lat)*torch.cos(lon)],-1)
R_e2c=torch.tensor([[1,0,0],[0,0,1],[0,-1,0]],dtype=torch.float32,device=dev)
d_dn=torch.einsum("ij,hwj->hwi",R_e2c,d_erp)
def proj(d,cal):
    fx,fy,cx,cy,k1,k2,k3,k4=cal
    x,y,zc=d[...,0],d[...,1],d[...,2]
    hyp=torch.sqrt(x*x+y*y).clamp(min=1e-9); th=torch.atan2(hyp,zc); t2=th*th
    r=th*(1+k1*t2+k2*t2**2+k3*t2**3+k4*t2**4)
    u=fx*r*x/hyp+cx; v=fy*r*y/hyp+cy
    return torch.stack([u/(1920-1)*2-1,v/(1920-1)*2-1],-1),th
gd,thd=proj(d_dn,cal_d)
ring=(thd>math.radians(80))&(thd<math.radians(100))
def load(p): return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),np.float32)).permute(2,0,1)[None].to(dev)/255
KS=(40,110,170,220)
FD=[Fn.grid_sample(load(f"{DOWN}/f_{k:04d}.jpg"),gd[None],align_corners=True)[0] for k in KS]
FU_raw=[load(f"{UP}/f_{k:04d}.jpg") for k in KS]
def R3(g,a,b):
    cg,sg=math.cos(g),math.sin(g); ca,sa=math.cos(math.pi+a),math.sin(math.pi+a); cb,sb=math.cos(b),math.sin(b)
    Rz1=torch.tensor([[cg,-sg,0],[sg,cg,0],[0,0,1]],dtype=torch.float32,device=dev)
    Rx=torch.tensor([[1,0,0],[0,ca,-sa],[0,sa,ca]],dtype=torch.float32,device=dev)
    Rz2=torch.tensor([[cb,-sb,0],[sb,cb,0],[0,0,1]],dtype=torch.float32,device=dev)
    return Rz1@Rx@Rz2
def score(Rr):
    d_up=torch.einsum("ij,hwj->hwi",Rr,d_dn)
    gu,thu=proj(d_up,cal_u); m=ring&(thu<TH_MAX)
    if m.sum()<400: return -9
    tot=0
    for fd,fu in zip(FD,FU_raw):
        b=Fn.grid_sample(fu,gu[None],align_corners=True)[0]
        x=fd[:,m]-fd[:,m].mean(); y=b[:,m]-b[:,m].mean()
        tot+=float((x*y).sum()/(x.norm()*y.norm()+1e-9))
    return tot/len(FD)
best=(-9,0,0,0)
for gdeg in range(0,360,3):
    s=score(R3(math.radians(gdeg),0,0))
    if s>best[0]: best=(s,gdeg,0,0)
print(f"coarse: corr={best[0]:.3f} g={best[1]}")
b0=best
for gdeg in np.arange(b0[1]-4,b0[1]+4.1,1.0):
    for a in np.arange(-12,12.1,2.0):
        for bb in np.arange(-12,12.1,2.0):
            s=score(R3(math.radians(gdeg),math.radians(a),math.radians(bb)))
            if s>best[0]: best=(s,gdeg,a,bb)
print(f"fine: corr={best[0]:.3f} g={best[1]:.1f} a={best[2]:.1f} b={best[3]:.1f}")
b1=best
for gdeg in np.arange(b1[1]-1,b1[1]+1.01,0.5):
    for a in np.arange(b1[2]-2,b1[2]+2.01,0.5):
        for bb in np.arange(b1[3]-2,b1[3]+2.01,0.5):
            s=score(R3(math.radians(gdeg),math.radians(a),math.radians(bb)))
            if s>best[0]: best=(s,gdeg,a,bb)
print(f"ultra: corr={best[0]:.3f} g={best[1]:.2f} a={best[2]:.2f} b={best[3]:.2f}")
Rr=R3(math.radians(best[1]),math.radians(best[2]),math.radians(best[3])).cpu().numpy()
np.savez(f"{ROOT}/p3_pano/rig023.npz",R_rig=Rr,cal_d=cal_d,cal_u=cal_u)
print("rig023.npz updated (photometric R_rig)")
