#!/usr/bin/env python3
"""Orientation gate: de-rotate one ERP with the GPS-quaternion R_wp under 4
conventions (xyzw/wxyz x R/R.T). The correct one yields an upright ERP with a
level, centered horizon (like §5.2 gravity-leveling). Writes a 4-row montage.
"""
import struct, os, math, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image

ROOT="/w"
F_INSV=f"{ROOT}/data/8kpano/VID_20260326_073432_023.insv"
S=os.path.getsize(F_INSV); xs=S-15672335
f=open(F_INSV,"rb"); f.seek(xs+3537641); d=f.read(410652)
QT,QQ=[],[]
for i in range(0,len(d)-35,36):
    QT.append(struct.unpack_from("<Q",d,i)[0]/1000.0)
    QQ.append(struct.unpack_from("<ffff",d,i+20))
QT=np.array(QT); QQ=np.array(QQ); QT-=QT[0]

K=120  # frame pano_0120
DUR=380.55; fps=240/DUR; t=K/fps
q=QQ[np.clip(np.searchsorted(QT,t),0,len(QQ)-1)]
print("quat sample:",np.round(q,3),"|q|=",round(float(np.linalg.norm(q)),3))

def q2R(q,order):
    if order=="xyzw": x,y,z,w=q
    else: w,x,y,z=q
    n=math.sqrt(x*x+y*y+z*z+w*w)+1e-9;x,y,z,w=x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

dev="cuda" if torch.cuda.is_available() else "cpu"
H,W=512,1024
vv,uu=torch.meshgrid(torch.arange(H,device=dev),torch.arange(W,device=dev),indexing="ij")
lon=(uu+0.5)/W*2*np.pi-np.pi; lat=np.pi/2-(vv+0.5)/H*np.pi
cl=torch.cos(lat)
dw=torch.stack([cl*torch.sin(lon),torch.sin(lat),cl*torch.cos(lon)],-1)

img=Image.open(f"{ROOT}/data/8kpano/scenes/scene_023hf_pano/panoramas/pano_0120.jpg").convert("RGB").resize((W,H))
t_img=torch.from_numpy(np.asarray(img,np.float32)).permute(2,0,1)[None].to(dev)/255.0

rows=[np.asarray(img)]  # row0 = original
labels=["original"]
for order in ("xyzw","wxyz"):
    for tr in (False,True):
        R=q2R(q,order); Rwp=R.T if tr else R
        Rt=torch.tensor(Rwp,dtype=torch.float32,device=dev)
        dp=torch.einsum("ij,hwj->hwi",Rt,dw)
        lon_p=torch.atan2(dp[...,0],dp[...,2]); lat_p=torch.asin(dp[...,1].clamp(-1+1e-6,1-1e-6))
        u=torch.remainder((lon_p+np.pi)/(2*np.pi)*W,W); v=(np.pi/2-lat_p)/np.pi*H
        grid=torch.stack([u/(W-1)*2-1,v/(H-1)*2-1],-1)[None]
        out=F.grid_sample(t_img,grid,mode="bilinear",padding_mode="border",align_corners=True)
        arr=(out[0].permute(1,2,0).clamp(0,1)*255).byte().cpu().numpy()
        rows.append(arr); labels.append(f"{order}{' R.T' if tr else ' R'}")
mont=np.concatenate(rows,0)
from PIL import ImageDraw
im=Image.fromarray(mont); dr=ImageDraw.Draw(im)
for i,lab in enumerate(labels):
    dr.text((6,i*H+6),lab,fill=(255,255,0))
    dr.line([(0,i*H+H//2),(W,i*H+H//2)],fill=(255,0,0),width=1)  # equator marker
im.save(f"{ROOT}/p6_unisharp/ft/quat_gate.jpg",quality=90)
print("saved quat_gate.jpg rows:",labels)
