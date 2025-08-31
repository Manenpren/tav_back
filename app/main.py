
from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict
import io
import pandas as pd
from fastapi.exceptions import RequestValidationError
from fastapi import Request
import traceback
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    SimulationRequest, SpillwayConfig, IntakeConfig,
    ElevationVolumePoint, ElevationQPoint, ElevationFactorPoint,
    SegmentNeedle, HydroPoint
)
from .engine import simulate

app = FastAPI(title="Hydraulics API", version="1.0.0")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # o ["*"] si solo es local y sin credenciales
    allow_credentials=False,    # True solo si usas cookies/autenticación de navegador
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Responde con TODO el detalle de validación
    return JSONResponse(
        status_code=422,
        content={"detail": "validation_error", "errors": exc.errors(), "body": str(await request.body())}
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Responde con el traceback (para depurar rápido)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "internal_error",
            "error": str(exc),
            "trace": traceback.format_exc(),
        },
    )

@app.post("/simulate")
def simulate_json(req: SimulationRequest):
    try:
        results, summary = simulate(req)
        return {
            "summary": summary.__dict__,
            "timeseries": [r.__dict__ for r in results]
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

# ---- File-based endpoint ----

def _read_csv_optional(file: Optional[UploadFile]) -> Optional[pd.DataFrame]:
    if file is None:
        return None
    file.file.seek(0)  # <- rebobinar SIEMPRE antes de leer
    df = pd.read_csv(file.file, sep=None, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

@app.post("/simulate/files")
async def simulate_files(
    # Required
    elcap: UploadFile = File(..., description="ELCAP.csv with columns elevation_m,volume_m3"),
    inflow: UploadFile = File(..., description="INFLOW.csv with columns t_hours,q_m3s"),

    # Spillway 1 basics
    sp1_mode: str = Form("LIBRE"),
    sp1_crest_m: float = Form(...),
    sp1_design_head_m: float = Form(...),
    sp1_length_m: float = Form(...),
    sp1_approach_depth_m: float = Form(0.0),
    sp1_auto_coefficient: bool = Form(True),
    sp1_discharge_coefficient: Optional[float] = Form(None),
    sp1_slope_correction_enabled: bool = Form(False),
    sp1_slope_variant: int = Form(1),
    sp1_els_enabled: bool = Form(False),

    # Spillway 2 (optional)
    sp2_present: bool = Form(False),
    sp2_mode: str = Form("LIBRE"),
    sp2_crest_m: Optional[float] = Form(None),
    sp2_design_head_m: Optional[float] = Form(None),
    sp2_length_m: Optional[float] = Form(None),
    sp2_approach_depth_m: float = Form(0.0),
    sp2_auto_coefficient: bool = Form(True),
    sp2_discharge_coefficient: Optional[float] = Form(None),
    sp2_slope_correction_enabled: bool = Form(False),
    sp2_slope_variant: int = Form(1),
    sp2_els_enabled: bool = Form(False),

    # Optional CSVs
    lvc1: UploadFile = File(None),
    lvc2: UploadFile = File(None),
    els1: UploadFile = File(None),
    els2: UploadFile = File(None),
    toma: UploadFile = File(None),
    elaguj1: UploadFile = File(None),
    elaguj2: UploadFile = File(None),

    # Intake
    intake_mode: str = Form("OFF"),
    intake_q_constant_m3s: float = Form(0.0),

    # Global options
    gravity: float = Form(9.81),
    tolerance: float = Form(1e-5),
    max_iter_step: int = Form(1000),
    velocity_head_enabled: bool = Form(False),
    dt_hours: Optional[float] = Form(None),
    initial_level_m: Optional[float] = Form(None),
    initial_volume_m3: Optional[float] = Form(None),
    drain_tail: bool = Form(True),
    tail_margin_m: float = Form(0.02),
    tail_hours_limit: Optional[float] = Form(None),
):
    try:
        # ---- DEBUG: tamaños de archivo ----
        elcap.file.seek(0); elcap_bytes = len(elcap.file.read()); elcap.file.seek(0)
        inflow.file.seek(0); inflow_bytes = len(inflow.file.read()); inflow.file.seek(0)
        print(f"[DEBUG] bytes -> elcap={elcap_bytes}, inflow={inflow_bytes}")

        # ---- Lee CSVs (helper rebobina y normaliza headers a lower) ----
        df_elcap = _read_csv_optional(elcap)
        df_inflow = _read_csv_optional(inflow)
        if df_elcap is None or df_elcap.empty:
            raise ValueError("ELCAP es requerido y no puede estar vacío.")
        if df_inflow is None or df_inflow.empty:
            raise ValueError("INFLOW es requerido y no puede estar vacío.")
        print(f"[DEBUG] cols -> elcap={list(df_elcap.columns)}, inflow={list(df_inflow.columns)}")

        # ---- Mapeo columnas ELCAP ----
        cols_elcap = set(df_elcap.columns)
        if {"elevation_m", "volume_m3"} <= cols_elcap:
            e_col = "elevation_m"; v_col = "volume_m3"
        elif {"elevación", "volumen"} <= cols_elcap or {"elevacion", "volumen"} <= cols_elcap:
            e_col = "elevación" if "elevación" in df_elcap.columns else "elevacion"; v_col = "volumen"
        elif {"elevación", "capacidad"} <= cols_elcap or {"elevacion", "capacidad"} <= cols_elcap:
            e_col = "elevación" if "elevación" in df_elcap.columns else "elevacion"; v_col = "capacidad"
        else:
            raise ValueError(
                f"Encabezados de ELCAP no válidos. Usa 'elevation_m,volume_m3', "
                f"'Elevación,Volumen' o 'Elevación,Capacidad'. Recibí: {list(df_elcap.columns)}"
            )

        # ✅ Construir ELCAP y luego validar
        elev_volume = [
            ElevationVolumePoint(elevation_m=float(r[e_col]), volume_m3=float(r[v_col]))
            for _, r in df_elcap.iterrows()
        ]
        if len(elev_volume) < 2:
            raise ValueError("ELCAP debe tener al menos 2 filas (elevación–volumen).")

        # ---- Mapeo columnas INFLOW ----
        if {"t_hours", "q_m3s"} <= set(df_inflow.columns):
            t_col = "t_hours"; q_col = "q_m3s"
        elif {"tiempo", "caudal"} <= set(df_inflow.columns):
            t_col = "tiempo"; q_col = "caudal"
        else:
            raise ValueError("INFLOW: usa 't_hours,q_m3s' o 'Tiempo,Caudal'.")

        # ✅ Construir INFLOW y luego validar
        inflow_pts = [
            HydroPoint(t_hours=float(r[t_col]), q_m3s=float(r[q_col]))
            for _, r in df_inflow.iterrows()
        ]
        if len(inflow_pts) < 2:
            raise ValueError("INFLOW debe tener al menos 2 filas (tiempo–caudal).")

        # ---- CSVs opcionales ----
        policy1 = None
        if lvc1 is not None:
            df = _read_csv_optional(lvc1)
            if df is not None and not df.empty:
                policy1 = [ElevationQPoint(elevation_m=float(r["elevation_m"]), q_m3s=float(r["q_m3s"])) for _, r in df.iterrows()]

        els_curve1 = None
        if els1 is not None:
            df = _read_csv_optional(els1)
            if df is not None and not df.empty:
                els_curve1 = [ElevationFactorPoint(elevation_m=float(r["elevation_m"]), factor=float(r["factor"])) for _, r in df.iterrows()]

        needles1 = None
        if elaguj1 is not None:
            df = _read_csv_optional(elaguj1)
            if df is not None and not df.empty:
                needles1 = [SegmentNeedle(length_m=float(r["length_m"]), crest_m=float(r["crest_m"])) for _, r in df.iterrows()]

        sp1 = SpillwayConfig(
            mode=sp1_mode,
            crest_m=sp1_crest_m,
            design_head_m=sp1_design_head_m,
            length_m=sp1_length_m,
            approach_depth_m=sp1_approach_depth_m,
            discharge_coefficient=sp1_discharge_coefficient,
            auto_coefficient=sp1_auto_coefficient,
            slope_correction_enabled=sp1_slope_correction_enabled,
            slope_variant=int(sp1_slope_variant),
            els_enabled=sp1_els_enabled,
            els_curve=els_curve1,
            needles=needles1,
            policy_q=policy1
        )

        # ---- Vertedor 2 opcional ----
        sp2 = None
        if sp2_present:
            # Validación mínima: si está presente, estos campos deben existir
            missing = []
            if sp2_crest_m is None: missing.append("sp2_crest_m")
            if sp2_design_head_m is None: missing.append("sp2_design_head_m")
            if sp2_length_m is None: missing.append("sp2_length_m")
            if missing:
                raise ValueError(f"Faltan parámetros para spillway2: {', '.join(missing)}")

            policy2 = None
            if lvc2 is not None:
                df = _read_csv_optional(lvc2)
                if df is not None and not df.empty:
                    policy2 = [ElevationQPoint(elevation_m=float(r["elevation_m"]), q_m3s=float(r["q_m3s"])) for _, r in df.iterrows()]

            els_curve2 = None
            if els2 is not None:
                df = _read_csv_optional(els2)
                if df is not None and not df.empty:
                    els_curve2 = [ElevationFactorPoint(elevation_m=float(r["elevation_m"]), factor=float(r["factor"])) for _, r in df.iterrows()]

            needles2 = None
            if elaguj2 is not None:
                df = _read_csv_optional(elaguj2)
                if df is not None and not df.empty:
                    needles2 = [SegmentNeedle(length_m=float(r["length_m"]), crest_m=float(r["crest_m"])) for _, r in df.iterrows()]

            sp2 = SpillwayConfig(
                mode=sp2_mode,
                crest_m=float(sp2_crest_m),
                design_head_m=float(sp2_design_head_m),
                length_m=float(sp2_length_m),
                approach_depth_m=sp2_approach_depth_m,
                discharge_coefficient=sp2_discharge_coefficient,
                auto_coefficient=sp2_auto_coefficient,
                slope_correction_enabled=sp2_slope_correction_enabled,
                slope_variant=int(sp2_slope_variant),
                els_enabled=sp2_els_enabled,
                els_curve=els_curve2,
                needles=needles2,
                policy_q=policy2
            )

        # ---- Intake opcional ----
        intake_curve = None
        if toma is not None:
            df = _read_csv_optional(toma)
            if df is not None and not df.empty:
                intake_curve = [ElevationQPoint(elevation_m=float(r["elevation_m"]), q_m3s=float(r["q_m3s"])) for _, r in df.iterrows()]

        intake = IntakeConfig(
            mode=intake_mode,
            q_constant_m3s=float(intake_q_constant_m3s),
            q_vs_level=intake_curve
        )

        # ---- dt_hours: inferir si no viene y validar uniformidad ----
        _dt = dt_hours
        if _dt is None and len(inflow_pts) >= 2:
            infer = inflow_pts[1].t_hours - inflow_pts[0].t_hours
            if infer == 0:
                raise ValueError("dt_hours=0; revisa columna de tiempo.")
            if not all(abs((inflow_pts[i+1].t_hours - inflow_pts[i].t_hours) - infer) < 1e-9
                       for i in range(len(inflow_pts)-1)):
                raise ValueError("El intervalo de tiempo en INFLOW no es uniforme.")
            _dt = float(infer)

        # ---- Construye request y simula ----
        req = SimulationRequest(
            gravity=gravity, tolerance=tolerance, max_iter_step=int(max_iter_step),
            elev_volume=elev_volume, inflow=inflow_pts, dt_hours=_dt,
            spillway1=sp1, spillway2=sp2,
            intake=intake, velocity_head_enabled=velocity_head_enabled,
            initial_level_m=initial_level_m, initial_volume_m3=initial_volume_m3,
            drain_tail=drain_tail, tail_margin_m=tail_margin_m, tail_hours_limit=tail_hours_limit
        )

        print("[DEBUG] Simulando con request:", req)

        results, summary = simulate(req)

        # ---- Respuesta OK ----
        return {
            "summary": summary.__dict__,
            "timeseries": [r.__dict__ for r in results]
        }

    except Exception as e:
        err = traceback.format_exc()
        print("[ERROR /simulate/files]", str(e))
        print(err)
        raise HTTPException(status_code=422, detail=f"{type(e).__name__}: {e}")


@app.post("/simulate/files/debug")
async def simulate_files_debug(
    elcap: UploadFile = File(...),
    inflow: UploadFile = File(...),
):
    def _read(u: UploadFile):
        u.file.seek(0)
        raw = u.file.read()
        u.file.seek(0)
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
        df.columns = [str(c).strip().lower() for c in df.columns]
        return {"size": len(raw), "columns": list(df.columns), "head": df.head(3).to_dict(orient="records")}

    try:
        elcap_info = _read(elcap)
        inflow_info = _read(inflow)
        return {
            "ok": True,
            "elcap": elcap_info,
            "inflow": inflow_info,
        }
    except Exception as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})

@app.post("/simulate/files/echo")
async def simulate_files_echo(request: Request):
    """
    Devuelve TODO lo que llegó en el multipart, sin procesar.
    Útil para depurar el front: nombres de campos, valores y archivos.
    """
    form = await request.form()

    fields = {}
    files = {}

    for key, value in form.multi_items():
        # UploadFile => archivo; str => campo normal
        if hasattr(value, "filename"):
            # Es un archivo (UploadFile)
            fileobj = value.file
            # medir tamaño sin “comerse” el stream
            pos = fileobj.tell()
            fileobj.seek(0, 2)               # al final
            size = fileobj.tell()
            fileobj.seek(pos)                 # vuelve donde estaba

            files[key] = {
                "filename": value.filename,
                "content_type": value.content_type,
                "size_bytes": size,
            }
        else:
            fields[key] = value

    result = {
        "ok": True,
        "fields": fields,
        "files": files,
    }

    # 👇 Esto lo imprime en consola del servidor
    print("Simulate Echo ->", result)            

    return {
        "ok": True,
        "fields": fields,
        "files": files,
    }
