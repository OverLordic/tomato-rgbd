from pathlib import Path
import sys,json,cv2,numpy as np
root=Path('/Users/alex/Desktop/tomato-rgbd');sys.path.insert(0,str(root/'src'));
from tomato_rgbd.data import load_development_manifest
from tomato_rgbd.baseline import extract_features as old
from tomato_rgbd.depth_baseline import extract_features as new
rows={r['plant_id']:r for r in load_development_manifest(root)}
ids=['C_07c66fd4','A_98ec0fae','B_417b35df','B_8b40f43a','C_84683099','D_4144ce28']
canvas=np.full((len(ids)*210,960,3),255,np.uint8)
for i,id in enumerate(ids):
 row=rows[id];rgb=cv2.imread(str(root/row['rgb_path']));_,a,ar=old(root/row['rgb_path'],root/row['depth_path']);_,b,br=new(root/row['rgb_path'],root/row['depth_path'])
 imgs=[rgb]
 for green,red in [(a,ar),(b,br)]:
  mask=np.zeros((540,960,3),np.uint8);mask[green>0]=(0,210,0);mask[red>0]=(0,0,255);imgs.append(mask)
 for n,img in enumerate(imgs):canvas[i*210:i*210+180,n*320:n*320+320]=cv2.resize(img,(320,180),interpolation=cv2.INTER_AREA if n==0 else cv2.INTER_NEAREST)
 cv2.putText(canvas,id+'  RGB / old / depth+green',(5,i*210+201),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),1)
cv2.imwrite('/Users/alex/Desktop/tomato-rgbd/reports/mask_comparison.png',canvas)
