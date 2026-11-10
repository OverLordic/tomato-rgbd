from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tomato_rgbd.depth_baseline import build_features
from tomato_rgbd.baseline import train_cv

def main():
 features=build_features(ROOT);features.to_csv(ROOT/'data/processed/depth_mask_features_development.csv',index=False)
 oof,scores=train_cv(ROOT,features);oof.to_csv(ROOT/'reports/depth_mask_oof.csv',index=False)
 (ROOT/'reports/depth_mask_scores.json').write_text(json.dumps(scores,indent=2)+'\n')
 print(json.dumps(scores,indent=2))
if __name__=='__main__':main()
