
# Hydraulics API (Reservoir Routing & Spillway Discharge)

FastAPI microservice that reproduces the logic from a legacy VB module to compute
reservoir routing (level–storage curve) with one or two spillways (free, with needles, or controlled)
plus an intake/outlet (obra de toma).

## Run locally

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Run test 
```bash
python -m pytest -vv -rP
```

Open docs at: http://localhost:8000/docs

## Docker

```bash
docker build -t hydraulics-api .
docker run --rm -p 8000:8000 hydraulics-api
```

## Endpoints

- `POST /simulate` — JSON payload (see `app/schemas.py:SimulationRequest`)
- `POST /simulate/files` — multipart upload of CSVs (ELCAP, INFLOW, optional LVC1/LVC2/ELS1/ELS2/TOMA/ELAGUJ1/ELAGUJ2)

**CSV formats** (headers required):

- `ELCAP.csv` — `elevation_m,volume_m3` (monotonic by volume)
- `INFLOW.csv` — `t_hours,q_m3s` (monotonic time; constant dt recommended)
- `LVC1.csv` — `elevation_m,q_m3s` (policy for spillway 1, if `CONTROLADO`)
- `LVC2.csv` — `elevation_m,q_m3s` (policy for spillway 2, if `CONTROLADO`)
- `ELS1.csv` — `elevation_m,factor` (launder/submergence factor for spillway 1)
- `ELS2.csv` — `elevation_m,factor` (launder/submergence factor for spillway 2)
- `TOMA.csv` — `q_m3s,elevation_m` (discharge vs water level for intake, used if `intake.mode="TABLE"`)
- `ELAGUJ1.csv` — `length_m,crest_m` (segments for spillway 1 in mode `CON_AGUJAS`)
- `ELAGUJ2.csv` — `length_m,crest_m` (segments for spillway 2 in mode `CON_AGUJAS`)

## PHP standalone script

A lightweight PHP port of the simulation engine is available at `php/simulate.php`.
Place this file on a PHP-enabled server and send a `POST` request with the same
JSON payload accepted by the FastAPI `/simulate` endpoint. The script executes
the hydraulic routing and persists the inputs, time series and summary metrics
into the MySQL schema defined in this repository.

## Notes

- Units: elevation in meters, volume in m³, flows in m³/s, time in hours.
- The algorithm follows the legacy structure: linear interpolation on curves, weir equation `Q=C L H^(3/2)`, optional
  velocity-head feedback, slope correction `CT`, needles summation, and trapezoidal storage routing.
- Tolerances and safeguards are configurable in the JSON request.
- The service returns time series and summary KPIs similar to the VB routine.
