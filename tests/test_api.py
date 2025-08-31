# tests/test_api.py
import io
import json
import pytest

def _mp(fieldname: str, content: bytes, filename: str):
    return (fieldname, (filename, io.BytesIO(content), "text/csv"))

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}

def test_files_debug_returns_parsed_metadata(client, elcap_csv_bytes, inflow_uniform_csv_bytes):
    files = [
        _mp("elcap", elcap_csv_bytes, "ELCAP.csv"),
        _mp("inflow", inflow_uniform_csv_bytes, "INFLOW.csv"),
    ]
    r = client.post("/simulate/files/debug", files=files)
    assert r.status_code == 200
    js = r.json()
    assert js["ok"] is True
    assert set(js["elcap"]["columns"]) == {"elevation_m", "volume_m3"}
    assert set(js["inflow"]["columns"]) == {"t_hours", "q_m3s"}
    assert len(js["elcap"]["head"]) > 0
    assert len(js["inflow"]["head"]) > 0

def test_files_echo_lists_fields_and_files(client, elcap_csv_bytes, inflow_uniform_csv_bytes):
    files = [
        _mp("elcap", elcap_csv_bytes, "ELCAP.csv"),
        _mp("inflow", inflow_uniform_csv_bytes, "INFLOW.csv"),
    ]
    data = {
        "sp1_mode": "LIBRE",
        "sp1_crest_m": "1000.5",
        "sp1_design_head_m": "0.5",
        "sp1_length_m": "10.0",
    }
    r = client.post("/simulate/files/echo", data=data, files=files)
    assert r.status_code == 200
    js = r.json()
    assert js["ok"] is True
    # Verifica que se capturen ambos tipos
    assert "sp1_mode" in js["fields"]
    assert "elcap" in js["files"] and "inflow" in js["files"]
    assert js["files"]["elcap"]["filename"] == "ELCAP.csv"

def test_simulate_files_happy_path_minimal(client, elcap_csv_bytes, inflow_uniform_csv_bytes):
    files = [
        _mp("elcap", elcap_csv_bytes, "ELCAP.csv"),
        _mp("inflow", inflow_uniform_csv_bytes, "INFLOW.csv"),
    ]
    data = {
        "sp1_mode": "LIBRE",
        "sp1_crest_m": "1000.5",
        "sp1_design_head_m": "0.3",
        "sp1_length_m": "12.0",
        "sp1_approach_depth_m": "0.0",
        "sp1_auto_coefficient": "true",
        "sp1_slope_correction_enabled": "false",
        "sp1_els_enabled": "false",
        "intake_mode": "OFF",
        "gravity": "9.81",
        "velocity_head_enabled": "false",
        "drain_tail": "true",
        "tail_margin_m": "0.02",
    }
    r = client.post("/simulate/files", data=data, files=files)
    assert r.status_code == 200, r.text
    js = r.json()
    assert "summary" in js and "timeseries" in js
    assert isinstance(js["timeseries"], list)
    # Debe haber al menos len(inflow)-1 pasos (más cola si drena)
    assert len(js["timeseries"]) >= 4

def test_simulate_files_invalid_headers_422(client, inflow_uniform_csv_bytes):
    # ELCAP con headers incorrectos
    bad_elcap = b"alt,vol\n1000,0\n1001,1000\n"
    files = [
        _mp("elcap", bad_elcap, "ELCAP.csv"),
        _mp("inflow", inflow_uniform_csv_bytes, "INFLOW.csv"),
    ]
    data = {
        "sp1_mode": "LIBRE",
        "sp1_crest_m": "1000.5",
        "sp1_design_head_m": "0.3",
        "sp1_length_m": "12.0",
    }
    r = client.post("/simulate/files", data=data, files=files)
    assert r.status_code == 422
    js = r.json()
    assert "detail" in js

def test_simulate_files_nonuniform_inflow_422(client, elcap_csv_bytes, inflow_nonuniform_csv_bytes):
    files = [
        _mp("elcap", elcap_csv_bytes, "ELCAP.csv"),
        _mp("inflow", inflow_nonuniform_csv_bytes, "INFLOW.csv"),
    ]
    data = {
        "sp1_mode": "LIBRE",
        "sp1_crest_m": "1000.1",
        "sp1_design_head_m": "0.3",
        "sp1_length_m": "8.0",
    }
    r = client.post("/simulate/files", data=data, files=files)
    assert r.status_code == 422
    assert "El intervalo de tiempo" in r.json()["detail"] or "uniforme" in r.json()["detail"]
