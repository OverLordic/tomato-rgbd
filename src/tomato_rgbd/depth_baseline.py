"""Experimental mask: combine green ExG with depth foreground against local tray."""
import cv2,numpy as np,pandas as pd
from tomato_rgbd.baseline import FEATURES,train_cv
from tomato_rgbd.data import load_development_manifest

def extract_features(rgb_path,depth_path):
 bgr=cv2.imread(str(rgb_path),cv2.IMREAD_COLOR);depth=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
 if bgr is None or depth is None or depth.dtype!=np.uint16 or depth.shape!=bgr.shape[:2]:raise ValueError('Invalid images')
 bgr=cv2.resize(bgr,(960,540),interpolation=cv2.INTER_AREA);depth=cv2.resize(depth,(960,540),interpolation=cv2.INTER_NEAREST)
 b,g,r=cv2.split(bgr.astype(np.float32)/255);exg=(2*g-r-b)/(r+g+b+1e-6)
 roi=np.zeros((540,960),np.uint8);roi[95:475,200:760]=1
 green=((exg>.08)&(g>.08)&(roi>0)).astype(np.uint8)
 green=cv2.morphologyEx(green,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
 hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
 # Find orange tray depth independently in each image, using color only.
 orange=((hsv[:,:,0]>=5)&(hsv[:,:,0]<=28)&(hsv[:,:,1]>100)&(hsv[:,:,2]>70)&(roi>0)&(depth>0))
 tray=depth[orange]
 if len(tray)>=100:
  reference=float(np.median(tray)); foreground=((depth>0)&(depth<reference-35)&(roi>0)).astype(np.uint8)
  foreground=cv2.morphologyEx(foreground,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
 else: foreground=np.zeros_like(green)
 plant=np.maximum(green,foreground)
 red=(((hsv[:,:,0]<10)|(hsv[:,:,0]>170))&(hsv[:,:,1]>120)&(hsv[:,:,2]>60)&(roi>0)).astype(np.uint8)
 red*=cv2.dilate(plant,np.ones((21,21),np.uint8))
 n,_,stats,_=cv2.connectedComponentsWithStats(red,8);components=sum(int(stats[i,cv2.CC_STAT_AREA])>=8 for i in range(1,n))
 ys,xs=np.nonzero(plant);area=float(roi.sum());width=(xs.max()-xs.min()+1)/960 if xs.size else 0.;height=(ys.max()-ys.min()+1)/540 if ys.size else 0.
 hull=cv2.contourArea(cv2.convexHull(np.column_stack((xs,ys)).astype(np.int32)))/area if xs.size>=3 else 0.
 values=depth[(plant>0)&(depth>0)].astype(np.float32)*.001;q=np.quantile(values,[.1,.25,.5,.75,.9]) if values.size else [np.nan]*5
 return dict(zip(FEATURES,map(float,[plant.sum()/area,width,height,hull,red.sum()/area,components,len(values)/max(int(plant.sum()),1),q[0],q[2],q[4],q[3]-q[1]]))),plant,red

def build_features(root):
 out=[]
 for i,row in enumerate(load_development_manifest(root),1):
  f,_,_=extract_features(root/row['rgb_path'],root/row['depth_path']);out.append({'plant_id':row['plant_id'],**f})
  if i%50==0 or i==383:print('Improved mask:',i,'/ 383',flush=True)
 return pd.DataFrame(out)
