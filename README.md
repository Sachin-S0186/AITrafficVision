# TrafficVision AI

A student-project starter for **historical traffic-speed forecasting** using METR-LA. The current image-based vehicle detection workflow is intentionally separate and is not yet included in this runnable forecasting starter.

## Project status
- METR-LA HDF5 file inspected: 34,272 timestamps × 207 sensors, five-minute sampling.
- Forecast training script: included.
- Forecast API and frontend: included.
- Training metrics: generate by running the script; do not claim model performance before reviewing them.
- Live deployment: not performed; requires your GitHub and hosting accounts.

## 1. Train the model
Use Python 3.10 or 3.11. From this folder:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt pandas tables
python training/train_forecast.py --data training/metr-la.h5 --out models
```

Place your downloaded `metr-la.h5` at `training/metr-la.h5` (the dataset is excluded from the deployment ZIP and Git). The script trains one Ridge regression model per sensor, using the previous 12 readings (60 minutes) to forecast the next five-minute reading. It uses chronological 70%/15%/15% train/validation/test partitions and writes `models/metr_la_ridge.joblib` plus `models/metrics.json`. The evaluation aggregates test predictions and compares them with a persistence baseline. Review the metrics before describing model quality.

## 2. Run API locally

```bash
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8000
```

If running from the repository root, the default model path resolves to `models/metr_la_ridge.joblib`. Visit `http://127.0.0.1:8000/docs`; use `GET /sensors`, `GET /health`, and `POST /forecast`.

`POST /forecast` JSON example (use exactly 12 actual consecutive readings for a sensor):

```json
{"sensor_id":"773869","recent_speeds":[64.4,62.7,61.8,60.5,61.2,62.0,63.1,64.0,63.6,62.8,63.4,64.0],"steps":3}
```

These example values are illustrative, not a verified observation sequence. Forecast values use the same units/conventions as the source data; confirm these before interpretation.

## 3. Deploy backend

Deploy the repository using the included root `Dockerfile` on a Docker-capable Python host. Ensure the trained artifact `models/metr_la_ridge.joblib` is available to the service (prefer object storage or a private release/artifact store for larger deployments). Set:

- `MODEL_PATH` to the deployed artifact path.
- `CORS_ORIGINS` to the exact Vercel site origin, e.g. `https://your-project.vercel.app`.

The Dockerfile starts Uvicorn on the host-provided `PORT`. Do not expose debug mode. Keep the HDF5 source data out of the backend image after training.

## 4. Deploy frontend to Vercel

1. Create a GitHub repository and push this project, excluding dataset/model artifacts if using `.gitignore`.
2. In Vercel, import the repository.
3. Set **Root Directory** to `frontend` and deploy as a static site (no build command; output directory `.`).
4. Open the deployed page and paste the backend base URL into the API URL field.
5. Ensure backend `CORS_ORIGINS` includes the exact deployed frontend origin; test the forecast form.

Alternative: deploy the root repository as a static site if Vercel's project configuration is set to serve `frontend/index.html`; using `frontend` as root is simplest.

## Important limitations
- METR-LA is historical data, not a live sensor stream. This app forecasts from user-provided readings.
- One-step Ridge regression with recursive multi-step forecasts is a transparent baseline, not a state-of-the-art model.
- The frontend example readings are placeholders; replace them with actual consecutive sensor values for meaningful predictions.
- Congestion labels cannot be inferred reliably from speed without a documented threshold/calibration dataset.
- Current image-based vehicle detection requires a separate annotated image/video dataset and model integration.
- Do not use for live traffic control or safety-critical decisions.
