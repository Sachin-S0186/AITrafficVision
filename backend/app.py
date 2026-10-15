import os, joblib, numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH=os.getenv('MODEL_PATH',os.path.join(os.path.dirname(__file__),'..','models','metr_la_ridge.joblib'))
app=FastAPI(title='TrafficVision AI API',version='0.1.0',description='METR-LA traffic-speed forecasting baseline')
origins=[x.strip() for x in os.getenv('CORS_ORIGINS','http://localhost:3000,http://127.0.0.1:5500').split(',') if x.strip()]
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=False,allow_methods=['GET','POST'],allow_headers=['*'])
_bundle=None
def bundle():
 global _bundle
 if _bundle is None:
  if not os.path.exists(MODEL_PATH): raise HTTPException(503,'Forecast model is not installed yet. Run the training script and configure MODEL_PATH.')
  _bundle=joblib.load(MODEL_PATH)
 return _bundle
class ForecastRequest(BaseModel):
 sensor_id: str = Field(description='Sensor ID from METR-LA, e.g. 773869')
 recent_speeds: list[float] = Field(min_length=12, max_length=12, description='Exactly 12 speed values, oldest to newest, sampled 5 minutes apart')
 steps: int = Field(default=1, ge=1, le=6, description='Forecast steps of 5 minutes each')
@app.get('/health')
def health(): return {'status':'ok','model_loaded':_bundle is not None,'service':'TrafficVision AI'}
@app.get('/sensors')
def sensors():
 b=bundle(); return {'sensor_ids':b['sensor_ids'],'lags':b['lags'],'sampling_minutes':b['sampling_minutes']}
@app.post('/forecast')
def forecast(req:ForecastRequest):
 b=bundle(); sid=str(req.sensor_id)
 if sid not in b['models']: raise HTTPException(404,f'Unknown sensor_id. Use GET /sensors for valid IDs.')
 seq=np.asarray(req.recent_speeds,dtype=float)
 if not np.isfinite(seq).all(): raise HTTPException(422,'recent_speeds must contain finite numeric values')
 model=b['models'][sid]; preds=[]; window=seq.copy()
 for _ in range(req.steps):
  y=float(model.predict(window.reshape(1,-1))[0]); preds.append(y); window=np.r_[window[1:],y]
 return {'sensor_id':sid,'forecast_horizon_minutes':[5*(i+1) for i in range(req.steps)],'predicted_speeds':preds,'units':'same as METR-LA source values; verify source conventions before operational use','model_type':b['model_type'],'warning':'Research prototype. Forecasts are not a live traffic feed or traffic-control advice.'}
