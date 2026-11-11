"""Leave one variety out on development only; parameters fixed in advance."""
from pathlib import Path
import os,sys,random,json,importlib.util
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('TORCH_HOME',str(ROOT/'data/torch_cache'));sys.path.insert(0,str(ROOT/'src'))
from tomato_rgbd.data import load_development_manifest
from tomato_rgbd.metrics import TARGET_NAMES,rmsre
from tomato_rgbd.baseline import FEATURES
from tomato_rgbd.geometry import FEATURES as GEO
spec=importlib.util.spec_from_file_location('rgb_training',ROOT/'scripts/train_resnet18_rgb.py');rgb=importlib.util.module_from_spec(spec);spec.loader.exec_module(rgb)

def main():
 rows=pd.DataFrame(load_development_manifest(ROOT));assert len(rows)==383 and rows.plant_id.is_unique
 images={}
 for i,r in enumerate(rows.itertuples(),1):
  images[r.plant_id]=rgb.make_image(ROOT/r.rgb_path)
  if i%100==0:print('Prepared:',i,'/383',flush=True)
 features=pd.read_csv(ROOT/'data/processed/baseline_features_development.csv').merge(pd.read_csv(ROOT/'data/processed/geometry_development.csv'),on='plant_id',validate='one_to_one')
 assert set(features.plant_id)==set(rows.plant_id)
 device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');predictions=[]
 for v in 'ABCD':
  tr=rows[rows.variety!=v].reset_index(drop=True);te=rows[rows.variety==v].reset_index(drop=True)
  assert not set(tr.plant_id)&set(te.plant_id) and set(tr.variety)==set('ABCD')-{v}
  train=tr.merge(features,on='plant_id',validate='one_to_one');test=te.merge(features,on='plant_id',validate='one_to_one');cols=FEATURES+GEO;p=np.zeros((len(te),4))
  for k,t in enumerate(TARGET_NAMES):
   valid=train[t].notna();model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=7,l2_regularization=1.,learning_rate=.05,random_state=42));model.fit(train.loc[valid,cols],train.loc[valid,t]);p[:,k]=np.maximum(model.predict(test[cols]),0)
  for i,id in enumerate(test.plant_id):predictions.append({'plant_id':id,'variety':v,'model':'baseline_geometry',**{t:float(p[i,k]) for k,t in enumerate(TARGET_NAMES)}})
  # Eight epochs = median of previously observed best epochs [18,6,9,8,4]; fixed before LOVO.
  random.seed(42);np.random.seed(42);torch.manual_seed(42)
  model=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1);model.fc=nn.Linear(model.fc.in_features,4);model=model.to(device)
  for par in model.parameters():par.requires_grad=False
  for par in list(model.fc.parameters())+list(model.layer4.parameters()):par.requires_grad=True
  optim=torch.optim.AdamW([{'params':model.fc.parameters(),'lr':1e-3},{'params':model.layer4.parameters(),'lr':1e-4}],weight_decay=1e-4)
  loader=DataLoader(rgb.PlantDataset(tr,images,True),batch_size=16,shuffle=True,num_workers=0)
  for epoch in range(8):
   model.train()
   for module in model.modules():
    if isinstance(module,nn.BatchNorm2d):module.eval()
   for x,y,mask,_ in loader:
    x,y,mask=x.to(device),y.to(device),mask.to(device);optim.zero_grad(set_to_none=True);loss=rgb.loss_fn(model(x),y,mask);loss.backward();optim.step()
   print('Variety',v,'epoch',epoch+1,'/8',flush=True)
  model.eval();loader=DataLoader(rgb.PlantDataset(te,images,False),batch_size=16,shuffle=False,num_workers=0)
  with torch.no_grad():
   for x,y,mask,ids in loader:
    vals=torch.expm1(model(x.to(device)).clamp(0,12)).cpu().numpy()
    for id,pred in zip(ids,vals):predictions.append({'plant_id':id,'variety':v,'model':'resnet18_rgb',**{t:float(pred[k]) for k,t in enumerate(TARGET_NAMES)}})
  folder=ROOT/'checkpoints/lovo_rgb';folder.mkdir(parents=True,exist_ok=True);torch.save({'variety':v,'epochs':8,'state':model.state_dict()},folder/f'{v}.pt')
 out=pd.DataFrame(predictions);out.to_csv(ROOT/'reports/lovo_predictions.csv',index=False);report={}
 for name in ['baseline_geometry','resnet18_rgb']:
  report[name]={}
  for v in 'ABCD':
   group=out[(out.model==name)&(out.variety==v)];truth=rows[rows.variety==v][['plant_id',*TARGET_NAMES]];j=truth.merge(group,on='plant_id',validate='one_to_one',suffixes=('_true','_pred'));assert len(j)==len(truth)
   score=rmsre(j[[t+'_true' for t in TARGET_NAMES]].to_numpy(),j[[t+'_pred' for t in TARGET_NAMES]].to_numpy());y=j.number_of_red_fruits_true;pp=j.number_of_red_fruits_pred;mask=y>0
   report[name][v]={'n':len(j),'mean':score.mean,'per_target':score.per_target,'n_valid':score.n_valid,'fruit_mae_zero':float(abs(pp[y==0]-y[y==0]).mean()) if (y==0).any() else None,'fruit_mae_positive':float(abs(pp[mask]-y[mask]).mean()) if mask.any() else None}
 (ROOT/'reports/lovo_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
