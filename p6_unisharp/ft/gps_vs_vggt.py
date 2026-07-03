import struct,os,json,math,numpy as np
F="data/8kpano/VID_20260326_073432_023.insv"; S=os.path.getsize(F)
extra_size=15672335; extra_start=S-extra_size
# --- decode GPS (id7 @ off 7624073, 12614B, 53B items) ---
f=open(F,"rb"); f.seek(extra_start+7624073); d=f.read(12614)
G=[]
for i in range(0,len(d)-52,53):
    ts=struct.unpack_from("<Q",d,i)[0]+struct.unpack_from("<H",d,i+8)[0]/1000.0
    lat=struct.unpack_from("<d",d,i+11)[0]; lon=struct.unpack_from("<d",d,i+20)[0]
    alt=struct.unpack_from("<d",d,i+45)[0]
    G.append((ts,lat,lon,alt))
G.sort(); t0=G[0][0]; lat0=G[0][1]; lon0=G[0][2]; alt0=G[0][3]
mlat=111320.0; mlon=111320.0*math.cos(math.radians(lat0))
gt=np.array([g[0]-t0 for g in G])
gE=np.array([(g[2]-lon0)*mlon for g in G]); gN=np.array([(g[1]-lat0)*mlat for g in G]); gU=np.array([g[3]-alt0 for g in G])
DUR=380.55; N=240; fps=N/DUR
# frame times
ft=np.array([(k)/fps for k in range(N)])   # pano_0001->t~0
def interp(x): return np.interp(ft,gt,x)
ENU=np.stack([interp(gE),interp(gN),interp(gU)],1)   # (240,3) GPS metric per frame
# --- VGGT centers ---
cams={c['idx']:c for c in json.load(open('p3_pano/pano_cams_scene_023hf.json'))['cameras']}
C=np.array([cams[k]['C'] for k in range(1,N+1)])       # (240,3)
# --- Umeyama sim3: C -> ENU ---
def umeyama(src,dst):
    mu_s=src.mean(0); mu_d=dst.mean(0); s=src-mu_s; d=dst-mu_d
    cov=d.T@s/len(src); U,D,Vt=np.linalg.svd(cov); Sgn=np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt)<0: Sgn[2,2]=-1
    R=U@Sgn@Vt; var=(s**2).sum()/len(src); scale=np.trace(np.diag(D)@Sgn)/var
    t=mu_d-scale*R@mu_s; return scale,R,t
sc,R,t=umeyama(C,ENU)
Cn=(sc*(R@C.T).T+t)
res=np.linalg.norm(Cn-ENU,axis=1)
def ext(a): return a.max(0)-a.min(0)
print(f"GPS ENU extent (m):   E={ext(ENU)[0]:.0f} N={ext(ENU)[1]:.0f} U={ext(ENU)[2]:.0f}")
print(f"VGGT C extent (units):{ext(C)[0]:.3f} {ext(C)[1]:.3f} {ext(C)[2]:.3f}")
print(f"Umeyama scale (VGGT->m) = {sc:.1f}   (VGGT unit ~= {sc:.1f} m)")
print(f"Sim3 fit residual: median={np.median(res):.1f}m  mean={res.mean():.1f}m  p90={np.percentile(res,90):.1f}m")
print(f"GPS trajectory total path: {np.linalg.norm(np.diff(ENU,axis=0),axis=1).sum():.0f} m")
print(f"=> if residual << extent(360m): VGGT shape OK, only mis-scaled. if ~extent: VGGT broke geometry.")
