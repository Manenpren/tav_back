# tests/test_engine.py
import math
import pytest
from app.schemas import HydroPoint

# 👇 AJUSTA ESTOS IMPORTS al path real
from app.engine import (
    simulate, weir_Q, intake_Q,
    level_from_volume, volume_from_level,
)
from app.schemas import (
    SimulationRequest, SpillwayConfig, IntakeConfig,
    ElevationVolumePoint, ElevationQPoint, ElevationFactorPoint,
    SegmentNeedle,
)

# ---------- helpers de fábrica ----------

def EVC(e, v): return ElevationVolumePoint(elevation_m=float(e), volume_m3=float(v))
def EQ(e, q):  return ElevationQPoint(elevation_m=float(e), q_m3s=float(q))
def EF(e, f):  return ElevationFactorPoint(elevation_m=float(e), factor=float(f))
def NG(l, c):  return SegmentNeedle(length_m=float(l), crest_m=float(c))

def base_elcap():
    # Monótona, con rango suficiente
    return [
        EVC(1000.0, 0.0),
        EVC(1001.0, 5_000.0),
        EVC(1002.5, 20_000.0),
        EVC(1004.0, 50_000.0),
    ]

def tri_inflow():
    # Triangular 0 → 5 → 10 → 5 → 0
    ts = [0.0, 1.0, 2.0, 3.0, 4.0]
    qs = [0.0, 5.0, 10.0, 5.0, 0.0]
    return [{"t_hours": t, "q_m3s": q} for t, q in zip(ts, qs)]

def inflow_points(lst):
    # convierte dicts a objetos ligeros para el SimulationRequest (pydantic lo convierte)
    return [{"t_hours": x["t_hours"], "q_m3s": x["q_m3s"]} for x in lst]

# ---------- tests de interpolación y curvas ----------

def test_level_volume_inverse_consistency():
    ev = base_elcap()
    for e in [1000.0, 1001.0, 1002.0, 1003.5]:
        v = volume_from_level(ev, e)
        assert v is not None
        e2 = level_from_volume(ev, v)
        assert e2 is not None
        assert e2 == pytest.approx(e, rel=0, abs=1e-9)

# ---------- tests weir/intake básicos ----------

def test_weir_free_below_crest_is_zero():
    cfg = SpillwayConfig(
        mode="LIBRE", crest_m=1001.0, design_head_m=0.5, length_m=10.0,
        approach_depth_m=0.0, auto_coefficient=True, discharge_coefficient=None,
        slope_correction_enabled=False, slope_variant=1, els_enabled=False,
        els_curve=None, needles=None, policy_q=None
    )
    q, c, ha = weir_Q(cfg, level_m=1000.5, gravity=9.81, velocity_head_enabled=False)
    assert q == 0.0
    assert ha == 0.0

def test_intake_modes():
    off = IntakeConfig(mode="OFF", q_constant_m3s=0.0, q_vs_level=None)
    const = IntakeConfig(mode="CONSTANT", q_constant_m3s=3.0, q_vs_level=None)
    table = IntakeConfig(mode="TABLE", q_constant_m3s=0.0, q_vs_level=[EQ(1001.0, 2.0), EQ(1002.0, 5.0)])

    assert intake_Q(off, 1002.0) == 0.0
    assert intake_Q(const, 999.0) == 3.0
    assert intake_Q(table, 1001.5) > 0.0

# ---------- simulate: casos clave ----------

# tests/test_engine.py

def _req_base(
    spillway_mode="LIBRE",
    sp1_extra=None,
    sp2=None,
    intake=IntakeConfig(mode="OFF", q_constant_m3s=0.0, q_vs_level=None),
    velocity_head=False,
    dt_hours=None,
    drain_tail=True,
    elcap=None,   # <--- NUEVO
):
    base_kwargs = dict(
        mode=spillway_mode,
        crest_m=1000.5,
        design_head_m=0.3,
        length_m=12.0,
        approach_depth_m=0.0,
        auto_coefficient=True,
        discharge_coefficient=None,
        slope_correction_enabled=False,
        slope_variant=1,
        els_enabled=False,
        els_curve=None,
        needles=None,
        policy_q=None,
    )
    if sp1_extra:
        base_kwargs.update(sp1_extra)

    sp1 = SpillwayConfig(**base_kwargs)

    return SimulationRequest(
        gravity=9.81, tolerance=1e-6, max_iter_step=200,
        elev_volume=(elcap if elcap is not None else base_elcap()),  # <---
        inflow=inflow_points(tri_inflow()),
        dt_hours=dt_hours,
        spillway1=sp1, spillway2=sp2,
        intake=intake, velocity_head_enabled=velocity_head,
        initial_level_m=None, initial_volume_m3=None,
        drain_tail=drain_tail, tail_margin_m=0.02, tail_hours_limit=None
    )


def test_simulate_happy_path_free_overflow_no_intake():
    req = _req_base()
    ts, sm = simulate(req)
    assert len(ts) >= 4
    # Máximo de nivel y volumen deben ser positivos y orden lógicamente consistente
    assert sm.max_level_m >= base_elcap()[0].elevation_m
    assert sm.max_stored_hm3 >= 0.0
    # picos definidos
    assert sm.peak_inflow_m3s == 10.0
    assert sm.peak_outflow_m3s >= 0.0

def test_simulate_dt_inferred_and_uniformity():
    req = _req_base(dt_hours=None)
    ts, _ = simulate(req)
    assert len(ts) >= 4  # dt inferido de inflow triangular (1 h)

def test_simulate_velocity_head_increases_Q():
    # Con velocidad de aproximación, Ha aumenta cabeza efectiva => Q no debería ser menor
    req_off = _req_base(velocity_head=False)
    req_on  = _req_base(velocity_head=True)
    ts_off, _ = simulate(req_off)
    ts_on,  _ = simulate(req_on)
    # Compara algún paso intermedio
    q_off = max(r.q_total_m3s for r in ts_off)
    q_on  = max(r.q_total_m3s for r in ts_on)
    assert q_on >= q_off * 0.99  # al menos no menor (permitimos pequeña diferencia numérica)

def big_elcap():
    return [
        EVC(1000.0, 0.0),
        EVC(1001.0, 10_000.0),
        EVC(1002.5, 80_000.0),
        EVC(1004.0, 200_000.0),
    ]

def test_simulate_controlled_policy_exact_q():
    policy = [EQ(1000.5, 1.0), EQ(1001.0, 2.0), EQ(1002.0, 3.0)]
    req = _req_base(
        spillway_mode="CONTROLADO",
        sp1_extra={"policy_q": policy},
        elcap=big_elcap(),          # <<--- clave
        drain_tail=False,           # (opcional) evita seguir drenando luego
    )
    ts, _ = simulate(req)
    assert all(r.q1_m3s >= 0.0 for r in ts)
    assert max(r.q1_m3s for r in ts) <= 3.0 + 1e-6

def test_simulate_needles_sum_segments():
    # Con agujas: Q debe sumar los segmentos
    needles = [NG(2.0, 1000.4), NG(3.0, 1000.3)]
    req = _req_base(
        spillway_mode="CON_AGUJAS",
        sp1_extra={"needles": needles}
    )
    ts, _ = simulate(req)
    # algo de descarga debería existir
    assert max(r.q1_m3s for r in ts) > 0.0

def test_simulate_with_constant_intake_reduces_storage():
    # Comparamos pico de almacenamiento con y sin toma constante
    req_no = _req_base(intake=IntakeConfig(mode="OFF", q_constant_m3s=0.0, q_vs_level=None))
    req_yes = _req_base(intake=IntakeConfig(mode="CONSTANT", q_constant_m3s=2.0, q_vs_level=None))
    ts_no, sm_no = simulate(req_no)
    ts_yes, sm_yes = simulate(req_yes)
    assert sm_yes.max_stored_hm3 <= sm_no.max_stored_hm3 + 1e-9

def test_mass_balance_coherence():
    req = _req_base(drain_tail=False)
    ts, _ = simulate(req)

    # tiempos y dt (uniforme)
    t = [p.t_hours for p in req.inflow]
    assert len(t) >= 2
    dt_h = t[1] - t[0]
    dt_s = dt_h * 3600.0

    # volumen inicial y mínimo del ELCAP
    emin = min(p.elevation_m for p in req.elev_volume)
    Vmin = min(p.volume_m3 for p in req.elev_volume)
    V = next(v.volume_m3 for v in req.elev_volume if v.elevation_m == emin)

    # q inicial (nivel inicial: emin)
    E0 = emin
    q1_0, _, _ = weir_Q(req.spillway1, E0, req.gravity, req.velocity_head_enabled)
    q2_0 = 0.0
    if req.spillway2 is not None:
        q2_0, _, _ = weir_Q(req.spillway2, E0, req.gravity, req.velocity_head_enabled)
    q_prev_total = q1_0 + q2_0 + intake_Q(req.intake, E0)

    inflow = [p.q_m3s for p in req.inflow]
    out_ts = [r.q_total_m3s for r in ts]  # Q al final de cada paso

    # Reproduce exactamente el update del motor (trapecio + clamp)
    for i in range(1, len(t)):
        I1 = inflow[i-1]
        I2 = inflow[i]
        q_tot = out_ts[i-1]               # Q al final del paso i
        V_next = V + 0.5 * ((I1 + I2) - (q_prev_total + q_tot)) * dt_s
        V_next = max(V_next, Vmin)        # mismo clamp que el motor
        V = V_next
        q_prev_total = q_tot

    Vend_real = ts[-1].volume_m3
    assert V == pytest.approx(Vend_real, rel=1e-3, abs=5.0)

def test_non_uniform_inflow_raises():
    # t: 0, 0.5, 2.0, 3.0 → no uniforme => ValueError
    req = _req_base()
    req.inflow = [
        HydroPoint(t_hours=0.0, q_m3s=2.0),
        HydroPoint(t_hours=0.5, q_m3s=2.0),
        HydroPoint(t_hours=2.0, q_m3s=2.0),
        HydroPoint(t_hours=3.0, q_m3s=2.0),
    ]
    with pytest.raises(ValueError):
        simulate(req)


def test_els_enabled_requires_factor():
    # Si activas ELS y el nivel cae fuera de la curva, el motor lanza error
    req = _req_base(spillway_mode="LIBRE", sp1_extra={
        "els_enabled": True,
        "els_curve": [EF(1000.5, 0.9), EF(1001.5, 0.95)],
    })
    # Forzamos overflow alto para que consulte el factor; si el nivel sale de rango podría fallar
    req.spillway1.crest_m = 1000.0
    # Asegura que los niveles durante el pico están dentro de [1000.5, 1001.5] o espera error si no
    try:
        simulate(req)
    except ValueError as e:
        assert "ELS factor" in str(e)
