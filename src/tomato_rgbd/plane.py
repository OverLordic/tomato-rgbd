"""Robust tray-plane estimate from aligned RGB-D; not the substrate surface."""
from pathlib import Path
import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression

PLANE_FEATURES=['tray_plane_valid','tray_inlier_fraction','tray_residual_mm','height_above_tray_q90_cm','height_above_tray_q98_cm']

def estimate_plane(xyz, random_state=42):
    """Fit Z=aX+bY+c in meters, using camera coordinates and robust sampling."""
    if len(xyz)<100:
        return None
    model=RANSACRegressor(estimator=LinearRegression(),min_samples=3,residual_threshold=.008,
                          max_trials=100,random_state=random_state)
    try:model.fit(xyz[:,:2],xyz[:,2])
    except ValueError:return None
    inliers=model.inlier_mask_
    if inliers.sum()<100 or inliers.mean()<.5:return None
    a,b=model.estimator_.coef_;c=model.estimator_.intercept_
    residual=np.median(np.abs(model.predict(xyz[inliers,:2])-xyz[inliers,2]))
    return np.array([a,b,c]),float(inliers.mean()),float(residual)

def plane_features(rgb_path,depth_path,calibration):
    bgr=cv2.imread(str(rgb_path),cv2.IMREAD_COLOR);d=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
    if bgr is None or d is None or d.dtype!=np.uint16 or d.shape!=bgr.shape[:2]:raise ValueError('Invalid RGB-D')
    # Stride on original pixels preserves intrinsic coordinates and depth values.
    step=8;bgr=bgr[::step,::step];d=d[::step,::step];hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    yy,xx=np.indices(d.shape);u=xx*step;v=yy*step;z=d.astype(np.float64)*.001
    ci=calibration['color_int'];x=(u-ci['ppx'])*z/ci['fx'];y=(v-ci['ppy'])*z/ci['fy']
    roi=(u>=800)&(u<3040)&(v>=380)&(v<1900)
    orange=(hsv[:,:,0]>=5)&(hsv[:,:,0]<=28)&(hsv[:,:,1]>100)&(hsv[:,:,2]>70)&roi&(d>0)
    # Exclude green plant material from candidate tray pixels.
    b,g,r=cv2.split(bgr.astype(np.float32)/255);exg=(2*g-r-b)/(r+g+b+1e-6)
    orange &= exg<=.08
    xyz=np.column_stack((x[orange],y[orange],z[orange]));fit=estimate_plane(xyz)
    if fit is None:return dict(zip(PLANE_FEATURES,[0,np.nan,np.nan,np.nan,np.nan]))
    coef,fraction,residual=fit;a,b,c=coef
    green=(exg>.08)&(g>.08)&roi&(d>0)
    # Signed perpendicular distance from tray plane in centimeters, positive toward camera.
    above=(a*x+b*y+c-z)/np.sqrt(a*a+b*b+1)*100
    heights=above[green];heights=heights[(heights>=0)&(heights<150)]
    if len(heights)<20:q90=q98=np.nan
    else:q90,q98=np.quantile(heights,[.9,.98])
    return dict(zip(PLANE_FEATURES,[1,fraction,residual*1000,q90,q98]))
