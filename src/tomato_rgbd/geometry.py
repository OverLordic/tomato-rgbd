"""Visible canopy geometry, in camera coordinates relative to estimated tray plane."""
import cv2,numpy as np
from tomato_rgbd.plane import estimate_plane

FEATURES=['plane_ok','plane_inlier_fraction','plane_coverage','plane_residual_mm','projected_area_cm2','height_q95_cm','height_q99_cm','envelope_volume_cm3','plant_valid_fraction']

def backproject(depth,calibration,step=1):
 d=depth[::step,::step];v,u=np.indices(d.shape);u=u*step;v=v*step;z=d.astype(np.float64)*.001
 ci=calibration['color_int'];x=(u-ci['ppx'])*z/ci['fx'];y=(v-ci['ppy'])*z/ci['fy']
 return np.stack((x,y,z),axis=-1),d>0

def grid_features(points,plane,cell=.005):
 """Orthogonal projection to plane, occupied cells and max-height envelope."""
 a,b,c=plane;normal=np.array([a,b,-1.]);normal/=np.linalg.norm(normal)
 origin=np.array([0.,0.,c]);axis=np.array([1.,0.,a]);axis/=np.linalg.norm(axis);second=np.cross(normal,axis);second/=np.linalg.norm(second)
 rel=points-origin; heights=rel@normal;uv=np.column_stack((rel@axis,rel@second));valid=(heights>=.005)&(heights<1.5)
 heights=heights[valid];uv=uv[valid]
 if len(heights)<20:return (np.nan,)*4
 grid=np.floor(uv/cell).astype(np.int64);_,inv=np.unique(grid,axis=0,return_inverse=True);maxh=np.zeros(inv.max()+1);np.maximum.at(maxh,inv,heights)
 return len(maxh)*cell**2*1e4,np.quantile(heights,.95)*100,np.quantile(heights,.99)*100,maxh.sum()*cell**2*1e6

def scene(rgb_path,depth_path,calibration,step=8):
 bgr=cv2.imread(str(rgb_path),cv2.IMREAD_COLOR);d=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
 if bgr is None or d is None or d.dtype!=np.uint16 or d.shape!=bgr.shape[:2]:raise ValueError('Invalid RGB-D')
 xyz,valid=backproject(d,calibration,step);color=bgr[::step,::step];hsv=cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
 yy,xx=np.indices(valid.shape);u=xx*step;v=yy*step;roi=(u>=800)&(u<3040)&(v>=380)&(v<1900)
 b,g,r=cv2.split(color.astype(np.float32)/255);exg=(2*g-r-b)/(r+g+b+1e-6)
 tray=(hsv[:,:,0]>=5)&(hsv[:,:,0]<=28)&(hsv[:,:,1]>100)&(hsv[:,:,2]>70)&roi&valid&(exg<=.08)
 plant=(exg>.08)&(g>.08)&roi
 candidates=xyz[tray];fit=estimate_plane(candidates)
 result=dict.fromkeys(FEATURES,np.nan);result['plane_ok']=0;result['plant_valid_fraction']=float((plant&valid).sum()/max(plant.sum(),1))
 info={'xyz':xyz,'color':color,'valid':valid,'tray':tray,'plant':plant,'fit':fit,'roi':roi}
 if fit is None:return result,info
 coef,frac,resid=fit;a,b,c=coef;inliers=tray & (np.abs(xyz[:,:,2]-(a*xyz[:,:,0]+b*xyz[:,:,1]+c))<.008)
 # Spread measured relative to ROI as occupied 8x8 image blocks, not proof of correct surface.
 blocks=np.unique(np.column_stack((yy[inliers]//30,xx[inliers]//40)),axis=0)
 coverage=len(blocks);points=xyz[plant&valid]
 area,q95,q99,volume=grid_features(points,coef)
 result.update(plane_ok=1,plane_inlier_fraction=frac,plane_coverage=coverage,plane_residual_mm=resid*1000,projected_area_cm2=area,height_q95_cm=q95,height_q99_cm=q99,envelope_volume_cm3=volume)
 info['inliers']=inliers
 return result,info
