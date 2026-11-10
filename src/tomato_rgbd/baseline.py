"""Simple RGB-D features and strictly out-of-fold regression."""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from tomato_rgbd.data import load_development_manifest, load_fold_dataframes
from tomato_rgbd.metrics import TARGET_NAMES, rmsre

FEATURES = ['green_fraction','green_bbox_width','green_bbox_height','green_hull_fraction',
            'red_fraction_near_green','red_components','depth_valid_fraction',
            'depth_q10_m','depth_q50_m','depth_q90_m','depth_iqr_m']


def extract_features(rgb_path, depth_path):
    """Fixed thresholds on a 960x540 frame; no labels or learned parameters."""
    bgr=cv2.imread(str(rgb_path),cv2.IMREAD_COLOR)
    depth=cv2.imread(str(depth_path),cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None or depth.dtype != np.uint16 or depth.shape != bgr.shape[:2]:
        raise ValueError('Missing/misaligned RGB and uint16 depth')
    bgr=cv2.resize(bgr,(960,540),interpolation=cv2.INTER_AREA)
    depth=cv2.resize(depth,(960,540),interpolation=cv2.INTER_NEAREST)
    b,g,r=cv2.split(bgr.astype(np.float32)/255.0)
    total=r+g+b+1e-6
    exg=(2*g-r-b)/total
    # Exclude the outer scene: table, cables and person can otherwise be green/red.
    roi=np.zeros((540,960),np.uint8); roi[95:475,200:760]=1
    green=((exg>0.08)&(g>0.08)&(roi>0)).astype(np.uint8)
    green=cv2.morphologyEx(green,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    red=(((hsv[:,:,0]<10)|(hsv[:,:,0]>170))&(hsv[:,:,1]>120)&(hsv[:,:,2]>60)&(roi>0)).astype(np.uint8)
    near=cv2.dilate(green,np.ones((21,21),np.uint8))
    red=red*near
    n,_,stats,_=cv2.connectedComponentsWithStats(red,8)
    components=sum(int(stats[i,cv2.CC_STAT_AREA])>=8 for i in range(1,n))
    ys,xs=np.nonzero(green); area=float(roi.sum())
    width=float((xs.max()-xs.min()+1)/960) if xs.size else 0.
    height=float((ys.max()-ys.min()+1)/540) if ys.size else 0.
    hull=0.
    if xs.size>=3:
        points=np.column_stack((xs,ys)).astype(np.int32)
        hull=float(cv2.contourArea(cv2.convexHull(points)))/area
    values=depth[(green>0)&(depth>0)].astype(np.float32)*0.001
    q=np.quantile(values,[.1,.25,.5,.75,.9]) if values.size else [np.nan]*5
    result=[green.sum()/area,width,height,hull,red.sum()/area,components,
            len(values)/max(int(green.sum()),1),q[0],q[2],q[4],q[3]-q[1]]
    return dict(zip(FEATURES,map(float,result))),green,red


def build_features(root, progress=print):
    rows=load_development_manifest(root); result=[]
    for i,row in enumerate(rows,1):
        features,_,_=extract_features(root/row['rgb_path'],root/row['depth_path'])
        result.append({'plant_id':row['plant_id'],**features})
        if progress and (i%50==0 or i==len(rows)):progress(f'Features: {i}/{len(rows)}',flush=True)
    return pd.DataFrame(result)


def train_cv(root, features):
    if features.plant_id.duplicated().any() or set(features.columns)!=set(['plant_id']+FEATURES):
        raise ValueError('Feature columns or IDs are invalid')
    models={'constant':None,'ridge':lambda: make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=10.0)),
            'boosting':lambda:make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=7,l2_regularization=1.0,learning_rate=.05,random_state=42))}
    predictions=[]
    for fold in range(5):
        tr,va=load_fold_dataframes(root,fold)
        tr=tr.merge(features,on='plant_id',validate='one_to_one');va=va.merge(features,on='plant_id',validate='one_to_one')
        if len(tr)+len(va)!=383:raise ValueError('Missing feature rows')
        for name,factory in models.items():
            pred=np.zeros((len(va),4))
            for j,target in enumerate(TARGET_NAMES):
                mask=tr[target].notna();y=tr.loc[mask,target].to_numpy(float)
                if name=='constant': pred[:,j]=np.mean(y)
                else:
                    model=factory();model.fit(tr.loc[mask,FEATURES],y);pred[:,j]=model.predict(va[FEATURES])
            pred=np.maximum(pred,0)  # fixed rule for physically nonnegative targets
            for i,plant in enumerate(va.plant_id):
                predictions.append({'plant_id':plant,'fold':fold,'model':name,**{t:pred[i,j] for j,t in enumerate(TARGET_NAMES)}})
    oof=pd.DataFrame(predictions)
    truth=pd.DataFrame(load_development_manifest(root))[['plant_id',*TARGET_NAMES]]
    scores={}
    for model in models:
        joined=truth.merge(oof[oof.model==model],on='plant_id',validate='one_to_one',suffixes=('_true','_pred'))
        if len(joined)!=383:raise ValueError('OOF coverage error')
        score=rmsre(joined[[t+'_true' for t in TARGET_NAMES]].to_numpy(),joined[[t+'_pred' for t in TARGET_NAMES]].to_numpy())
        scores[model]={'mean':score.mean,'per_target':score.per_target,'n_valid':score.n_valid}
    return oof,scores
