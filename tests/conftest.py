# tests/conftest.py
import io
import pandas as pd
import pytest
from starlette.testclient import TestClient

# 👇 AJUSTA ESTA LÍNEA AL PATH REAL DE TU APP
from app.main import app  # p.ej. myapp.api == archivo donde está tu FastAPI (el código que pegaste)

@pytest.fixture(scope="session")
def client():
    return TestClient(app)

@pytest.fixture
def elcap_csv_bytes():
    # Elevación–Volumen simple y estrictamente creciente
    df = pd.DataFrame({
        "elevation_m": [1000.0, 1001.0, 1002.5, 1004.0],
        "volume_m3":   [0.0,    5_000.0, 20_000.0, 50_000.0],
    })
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    bio.seek(0)
    return bio.read()

@pytest.fixture
def inflow_uniform_csv_bytes():
    df = pd.DataFrame({
        "t_hours": [0.0, 1.0, 2.0, 3.0, 4.0],
        "q_m3s":   [0.0, 5.0, 10.0, 5.0, 0.0],
    })
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    bio.seek(0)
    return bio.read()

@pytest.fixture
def inflow_nonuniform_csv_bytes():
    df = pd.DataFrame({
        "t_hours": [0.0, 0.5, 2.0, 3.0],  # paso irregular (0.5, 1.5, 1.0)
        "q_m3s":   [2.0,  2.0, 2.0, 2.0],
    })
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    bio.seek(0)
    return bio.read()
