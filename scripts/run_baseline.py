from pathlib import Path
import json,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tomato_rgbd.baseline import build_features,train_cv,FEATURES

def main():
    features=build_features(ROOT)
    features.to_csv(ROOT/'data/processed/baseline_features_development.csv',index=False)
    oof,scores=train_cv(ROOT,features)
    oof.to_csv(ROOT/'reports/baseline_oof.csv',index=False)
    report={'scope':'development only; fixed 5-fold OOF; holdout untouched','feature_columns':FEATURES,'scores':scores,
            'limitations':['Threshold masks are crude; red tray/person may contaminate','Depth quantiles are camera distances, not plant height','Mean constant is not optimized for RMSRE','Hyperparameters are initial fixed choices']}
    (ROOT/'reports/baseline_scores.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(scores,indent=2))
if __name__=='__main__':main()
