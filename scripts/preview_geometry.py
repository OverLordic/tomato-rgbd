from pathlib import Path
import sys,cv2,numpy as np,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tomato_rgbd.data import load_development_manifest,read_json
from tomato_rgbd.geometry import scene
rows=load_development_manifest(ROOT);cal=read_json(ROOT/'data/raw/single_plant_cam/oak-d-s2-poe_conf.json');ids=['A_04ade032','B_0328ab97','C_07c66fd4','D_028f38e4'];canvas=np.full((len(ids)*270,1440,3),255,np.uint8);diag={}
for i,id in enumerate(ids):
 row=next(r for r in rows if r['plant_id']==id);result,info=scene(ROOT/row['rgb_path'],ROOT/row['depth_path'],cal);diag[id]={k:(None if isinstance(v,float) and np.isnan(v) else v) for k,v in result.items()}
 color=info['color'].copy();over=color.copy();over[info['tray']]=[255,0,255]
 if 'inliers' in info:over[info['inliers']]=[0,255,255]
 plant=info['plant'];over[plant]=[0,190,0]
 images=[cv2.resize(color,(360,203)),cv2.resize(over,(360,203))]
 xyz=info['xyz'];valid=info['valid']&info['roi'];points=xyz[valid];col=color[valid];stride=max(len(points)//50000,1);points=points[::stride];col=col[::stride]
 # Fixed physical axes to make scale across plants comparable: top X/Y, side X/Z.
 for side in (False,True):
  img=np.full((203,360,3),245,np.uint8);px=((points[:,0]+.6)/1.2*359).astype(int)
  py=(((points[:,2]-.3)/1.7 if side else (points[:,1]+.45)/.9)*202).astype(int)
  ok=(px>=0)&(px<360)&(py>=0)&(py<203);img[py[ok],px[ok]]=col[ok];images.append(img)
 y=i*270
 for n,img in enumerate(images):canvas[y:y+203,n*360:(n+1)*360]=img
 label=id+' RGB | tray candidates magenta, inliers yellow, plant green | cloud XY | cloud XZ'
 cv2.putText(canvas,label,(5,y+222),cv2.FONT_HERSHEY_SIMPLEX,.44,(0,0,0),1)
 cv2.putText(canvas,'Plane valid: '+str(result['plane_ok'])+' | coverage blocks: '+str(result['plane_coverage']),(5,y+244),cv2.FONT_HERSHEY_SIMPLEX,.44,(0,0,0),1)
cv2.imwrite(str(ROOT/'reports/geometry_scene_check.png'),canvas);(ROOT/'reports/geometry_scene_diagnostics.json').write_text(json.dumps(diag,indent=2))
