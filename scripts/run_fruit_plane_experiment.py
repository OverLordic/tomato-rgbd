from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from tomato_rgbd.data import load_development_manifest,load_fold_dataframes,read_json
from tomato_rgbd.baseline import FEATURES
from tomato_rgbd.plane import plane_features,PLANE_FEATURES
rows=load_development_manifest(root);cal=read_json(root/'data/raw/single_plant_cam/oak-d-s2-poe_conf.json')
out=[]
for i,r in enumerate(rows,1):
 out.append({'plant_id':r['plant_id'],**plane_features(root/r['rgb_path'],root/r['depth_path'],cal)})
 if i%50==0:print('Planes',i,'/383',flush=True)
plane=pd.DataFrame(out);plane.to_csv(root/'data/processed/tray_plane_features_development.csv',index=False)
base=pd.read_csv(root/'data/processed/baseline_features_development.csv');feat=base.merge(plane,on='plant_id',validate='one_to_one')
models=['mse','weighted','positive_only','two_stage','weighted_plane'];allpred=[]
for fold in range(5):
 train,val=load_fold_dataframes(root,fold);train=train.merge(feat,on='plant_id',validate='one_to_one');val=val.merge(feat,on='plant_id',validate='one_to_one')
 for name in models:
  cols=FEATURES+(PLANE_FEATURES if name=='weighted_plane' else [])
  imp=SimpleImputer(strategy='median',keep_empty_features=True);x=imp.fit_transform(train[cols]);v=imp.transform(val[cols]);y=train.number_of_red_fruits.to_numpy(float);valid=np.isfinite(y);x=x[valid];y=y[valid]
  reg=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=7,l2_regularization=1.,learning_rate=.05,random_state=42)
  if name=='mse':reg.fit(x,y);p=reg.predict(v)
  elif name in ('weighted','weighted_plane'):
   w=1/(1+y)**2;reg.fit(x,y,sample_weight=w/w.mean());p=reg.predict(v)
  elif name=='positive_only':
   pos=y>0;reg.fit(x[pos],y[pos]);p=reg.predict(v)
  else:
   clf=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,l2_regularization=1.,learning_rate=.05,random_state=42)
   clf.fit(x,(y>0).astype(int));pos=y>0;reg.fit(x[pos],y[pos]);p=clf.predict_proba(v)[:,1]*np.maximum(reg.predict(v),0)
  for id,true,pred in zip(val.plant_id,val.number_of_red_fruits,np.maximum(p,0)):allpred.append(dict(plant_id=id,fold=fold,model=name,truth=true,prediction=pred))
pred=pd.DataFrame(allpred);pred.to_csv(root/'reports/fruit_weighted_oof.csv',index=False)
report={}
for name,g in pred.groupby('model',sort=False):
 err=g.prediction-g.truth;zero=g.truth==0;pos=g.truth>0
 report[name]={'rmsre':float(np.sqrt(np.mean((err/(g.truth+1))**2))),'mae_all':float(np.mean(abs(err))),'mae_zero':float(np.mean(abs(err[zero]))),'mae_positive':float(np.mean(abs(err[pos]))),'mean_prediction_zero':float(g.prediction[zero].mean()),'n_zero':int(zero.sum()),'n_positive':int(pos.sum())}
report['plane_diagnostics']={'valid':int(plane.tray_plane_valid.sum()),'total':len(plane),'median_inlier_fraction':float(plane.tray_inlier_fraction.median()),'median_residual_mm':float(plane.tray_residual_mm.median()),'note':'tray plane only; not calibrated substrate plane'}
(root/'reports/fruit_weighted_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
