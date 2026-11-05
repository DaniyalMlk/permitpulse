"""Train and export the frozen HistGB teammate-base business-day quantile models."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, r2_score
from threadpoolctl import threadpool_limits
from model_features import load_all, make_features, encode, CATEGORICAL

PARAMS = dict(max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=30,
              learning_rate=0.05, max_iter=135, early_stopping=False, random_state=42)


def calculate(y, ps):
    pred = np.sort(np.column_stack([ps[.1], ps[.5], ps[.9]]), axis=1)
    p10, p50, p90 = pred.T
    return dict(count=len(y), p50_mae=float(mean_absolute_error(y,p50)),
                p50_r2=float(r2_score(y,p50)),
                p90_pinball=float(mean_pinball_loss(y,p90,alpha=.9)),
                coverage_pct=float(100*np.mean((y>=p10)&(y<=p90))),
                mean_interval_width=float(np.mean(p90-p10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',default='data/dob_now_clean_columns.parquet')
    ap.add_argument('--output',default='models')
    args=ap.parse_args()
    target=Path(args.output); target.mkdir(parents=True,exist_ok=True)
    _, data, work_cols = load_all(Path(args.data))
    data = data.loc[data.filing_date.dt.year.between(2021,2025)].copy()
    # Freeze historical study split. The 2025 data is NOT used for training.
    tr = data.filing_date.dt.year.between(2021,2023)
    va = data.filing_date.dt.year.eq(2024)
    te = data.filing_date.dt.year.eq(2025)
    feat = make_features(data,work_cols)
    prep, Xtr, Xva, catmask = encode(feat.loc[tr],feat.loc[va],CATEGORICAL)
    Xte = prep.transform(feat.loc[te]).astype('float32')
    ytr = data.loc[tr,'approval_business_days'].to_numpy()
    yva = data.loc[va,'approval_business_days'].to_numpy()
    yte = data.loc[te,'approval_business_days'].to_numpy()
    models={}
    valpred={}; testpred={}
    for q in (.1,.5,.9):
        m=HistGradientBoostingRegressor(loss='quantile',quantile=q,**PARAMS,
                                         categorical_features=catmask)
        with threadpool_limits(limits=6):
            m.fit(Xtr,ytr)
            valpred[q]=np.maximum(m.predict(Xva),0)
            testpred[q]=np.maximum(m.predict(Xte),0)
        models[q]=m
        print('Trained P'+str(int(q*100)),flush=True)
    # Historical metrics recorded for monitoring; do not tune using 2025 test here.
    metrics={}
    for label, mask, yy, pp in [
        ('2024_validation',va,yva,valpred),('2025_reused_holdout',te,yte,testpred)
    ]:
        smask=data.loc[mask,'filing_review_type'].eq('Standard Plan Examination').to_numpy()
        metrics[label]={'all':calculate(yy,pp),
                         'standard_plan':calculate(yy[smask],{q: v[smask] for q,v in pp.items()})}
    joblib.dump(dict(models=models,preprocessor=prep,feature_columns=list(feat.columns),
                     work_columns=work_cols,categorical_features=CATEGORICAL,
                     target='approval_business_days',training_years='2021-2023',
                     holiday_calendar='US NY state observed; DOB office closures may differ',
                     trained_at=datetime.now(timezone.utc).isoformat()), target/'permitpulse_histgb.joblib')
    (target/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
    print('Exported:',target/'permitpulse_histgb.joblib')

if __name__=='__main__': main()
