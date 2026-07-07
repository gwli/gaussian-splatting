import json, math
import numpy as np
def centers_obs(p):
    C={}; O={}
    for ln in open(p):
        if ln.startswith('#') or not ln.strip(): continue
        f=ln.split()
        if len(f)>=10 and f[9].endswith('.jpg'):
            w,x,y,z=map(float,f[1:5]); t=np.array(list(map(float,f[5:8])))
            n=math.sqrt(w*w+x*x+y*y+z*z); w,x,y,z=w/n,x/n,y/n,z/n
            R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
            cur=f[9]; C[cur]=-R.T@t; O[cur]=0
        elif len(f)>3:
            O[cur]=sum(1 for i in range(2,len(f),3) if f[i]!='-1')
    return C,O
C,O=centers_obs('/w/data/8kpano/scenes/fish023d/ba4/images.txt')
ns=sorted(C); P=np.stack([C[n] for n in ns])
z=np.zeros(len(ns)); z[1:-1]=np.linalg.norm(P[2:]-2*P[1:-1]+P[:-2],axis=1)*39.76
good={int(n.split('_')[1].split('.')[0]) for i,n in enumerate(ns) if O[n]>=400 and z[i]<=0.25}
m2=json.load(open('/w/p3_pano/pano_cams_scene_023rigd.json'))
m2['cameras']=[c for c in m2['cameras'] if c['idx'] in good]
json.dump(m2,open('/w/p3_pano/pano_cams_scene_023rigd3.json','w'),indent=1)
print('hard-filtered cameras:',len(m2['cameras']))
