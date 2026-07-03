import numpy as np
from plyfile import PlyData
for tag,P in [("GPS(023hf)","/w/data/8kpano/scenes/scene_023hf_pano/output_gps/point_cloud/iteration_7000/point_cloud.ply"),
              ("VGGT(023hf)","/w/data/8kpano/scenes/scene_023hf_pano/output_pano/point_cloud/iteration_15000/point_cloud.ply")]:
    try:
        v=PlyData.read(P)["vertex"]; xyz=np.stack([v["x"],v["y"],v["z"]],1)
        p5,p95=np.percentile(xyz,5,0),np.percentile(xyz,95,0)
        print(f"{tag:14} N={len(xyz):7d}  bbox(5-95%)={np.round(p95-p5,1).tolist()} m")
    except Exception as e:
        print(tag,"ERR",e)
