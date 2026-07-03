import numpy as np
from plyfile import PlyData
from PIL import Image
def load(p):
    v=PlyData.read(p)["vertex"]; names=v.data.dtype.names
    xyz=np.stack([v["x"],v["y"],v["z"]],1).astype(np.float32)
    if "f_dc_0" in names:
        c=np.stack([v["f_dc_0"],v["f_dc_1"],v["f_dc_2"]],1).astype(np.float32)
        col=np.clip(0.5+0.2820948*c,0,1)
    elif "red" in names:
        col=np.stack([v["red"],v["green"],v["blue"]],1).astype(np.float32)/255
    else: col=np.full((len(xyz),3),0.5,np.float32)
    op=1/(1+np.exp(-v["opacity"])) if "opacity" in names else np.ones(len(xyz))
    return xyz,col,op
def topdown(p,out,res=1000,rad=2):
    xyz,col,op=load(p)
    m=op>0.1; xyz,col=xyz[m],col[m]
    # brighten: normalize color percentiles
    lo_c,hi_c=np.percentile(col,3),np.percentile(col,97); col=np.clip((col-lo_c)/(hi_c-lo_c+1e-6),0,1)
    E,N,U=xyz[:,0],xyz[:,1],xyz[:,2]
    lo=np.percentile(xyz,1,0); hi=np.percentile(xyz,99,0)
    xi=np.clip(((E-lo[0])/(hi[0]-lo[0]+1e-9)*(res-1)).astype(int),0,res-1)
    yi=np.clip(((hi[1]-N)/(hi[1]-lo[1]+1e-9)*(res-1)).astype(int),0,res-1)
    img=np.zeros((res,res,3),np.float32); zb=np.full((res,res),-1e9)
    order=np.argsort(U)
    for i in order:
        x0,x1=max(0,xi[i]-rad),min(res,xi[i]+rad+1); y0,y1=max(0,yi[i]-rad),min(res,yi[i]+rad+1)
        img[y0:y1,x0:x1]=col[i]
    Image.fromarray((img*255).astype(np.uint8)).save(out)
    print(f"saved {out}  span E={hi[0]-lo[0]:.0f}m N={hi[1]-lo[1]:.0f}m  npts={len(xyz)}")
topdown("/w/data/8kpano/scenes/scene_023hf_pano/output_gps/point_cloud/iteration_7000/point_cloud.ply","/w/p6_unisharp/ft/topdown_gps.png")
