"""Train per-sensor Ridge models on METR-LA. Run: python train_forecast.py --data metr-la.h5 --out ../models"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='metr-la.h5'); ap.add_argument('--out',default='../models'); ap.add_argument('--lags',type=int,default=12); ap.add_argument('--alpha',type=float,default=1.0); args=ap.parse_args()
    os.makedirs(args.out,exist_ok=True)
    df=pd.read_hdf(args.data).sort_index()
    if not isinstance(df.index,pd.DatetimeIndex): raise ValueError('Expected timestamp index in HDF5 data')
    df=df.apply(pd.to_numeric,errors='coerce').interpolate(limit_direction='both')
    values=df.to_numpy(dtype=np.float32); n,sensors=values.shape; L=args.lags
    # Chronological split: 70% train, 15% validation, 15% test. No random shuffling.
    train_end=int(n*.70); val_end=int(n*.85)
    models={}; all_true=[]; all_pred=[]; val_true=[]; val_pred=[]
    for j,name in enumerate(df.columns.astype(str)):
        series=values[:,j]
        X=np.stack([series[i-L:i] for i in range(L,n)])
        y=series[L:]
        # X/y row corresponds to timestamp index i; preserve chronological split by target time.
        target_idx=np.arange(L,n)
        tr=target_idx<train_end; va=(target_idx>=train_end)&(target_idx<val_end); te=target_idx>=val_end
        model=Ridge(alpha=args.alpha)
        model.fit(X[tr],y[tr]); models[name]=model
        if va.any(): val_pred.extend(model.predict(X[va]).tolist()); val_true.extend(y[va].tolist())
        if te.any(): all_pred.extend(model.predict(X[te]).tolist()); all_true.extend(y[te].tolist())
    # Aggregate per-sensor 1-step predictions, plus persistence baseline (last value).
    # Recompute persistence on same test targets using each series' previous reading.
    persistence=[]
    for j in range(sensors):
        series=values[:,j]; persistence.extend(series[np.arange(L,n)[np.arange(L,n)>=val_end]-1].tolist())
    mae=float(mean_absolute_error(all_true,all_pred)); rmse=float(np.sqrt(mean_squared_error(all_true,all_pred)))
    p_mae=float(mean_absolute_error(all_true,persistence)); p_rmse=float(np.sqrt(mean_squared_error(all_true,persistence)))
    bundle={'models':models,'sensor_ids':[str(c) for c in df.columns], 'lags':L,'sampling_minutes':5,'trained_rows':train_end,'validation_end_row':val_end,'data_start':str(df.index[0]),'data_end':str(df.index[-1]),'model_type':'per-sensor Ridge regression, 1-step-ahead','note':'Historical speed forecasting baseline; not a live traffic feed.'}
    joblib.dump(bundle,os.path.join(args.out,'metr_la_ridge.joblib'),compress=3)
    metrics={'dataset_shape':[int(n),int(sensors)],'sampling_minutes':5,'lags':L,'forecast_horizon_minutes':5,'split':{'train':'first 70%','validation':'next 15%','test':'final 15%','chronological':True},'test_mae':mae,'test_rmse':rmse,'persistence_baseline_mae':p_mae,'persistence_baseline_rmse':p_rmse,'improvement_over_persistence_mae_percent':float((p_mae-mae)/p_mae*100) if p_mae else None,'records_start':str(df.index[0]),'records_end':str(df.index[-1]),'sensor_count':int(sensors),'trained_model_count':len(models)}
    with open(os.path.join(args.out,'metrics.json'),'w') as f: json.dump(metrics,f,indent=2)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
