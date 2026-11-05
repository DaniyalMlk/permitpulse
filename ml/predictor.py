"""Inference on the same engineered features as the training notebook."""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from model_features import BASE_CAT, EXTRA_CAT, NUM_RAW, make_features

class PermitPulsePredictor:
    def __init__(self, path='models/permitpulse_histgb.joblib'):
        self.bundle = joblib.load(path)

    def predict(self, request):
        request = dict(request)
        for field in ('filing_date','filing_review_type','job_type'):
            if not request.get(field):
                raise ValueError(f'Missing required field: {field}')
        day = pd.to_datetime(request['filing_date'], errors='coerce')
        if pd.isna(day):
            raise ValueError('Invalid filing_date: use YYYY-MM-DD')
        if not (2021 <= day.year <= 2035):
            raise ValueError('filing_date outside supported range 2021–2035')
        req = pd.DataFrame([request])
        for c in (BASE_CAT + [x for x in EXTRA_CAT if x != 'filing_kind'] + NUM_RAW
                  + self.bundle['work_columns'] + ['job_filing_number']):
            if c not in req: req[c] = np.nan
        req['filing_date'] = pd.to_datetime(req['filing_date'])
        feat = make_features(req,self.bundle['work_columns'])
        # Training fitted columns and preprocessing are saved with the model.
        feat = feat.reindex(columns=self.bundle['feature_columns'])
        x = self.bundle['preprocessor'].transform(feat).astype('float32')
        vals = sorted([max(0.,float(self.bundle['models'][q].predict(x)[0]))
                       for q in (.1,.5,.9)])
        notes=[]
        if not request.get('filing_kind') and not request.get('job_filing_number'):
            notes.append('Missing filing_kind/job_filing_number: a high-importance input is unknown; quality may degrade.')
        if not request.get('borough'):
            notes.append('Borough not supplied.')
        return dict(unit='business_days',p10=round(vals[0],2),p50=round(vals[1],2),
                    p90=round(vals[2],2),
                    target_definition='Business days from filing to approval; excludes filing day',
                    notes=notes)
