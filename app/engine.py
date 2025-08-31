from __future__ import annotations
from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .schemas import (
    SimulationRequest, SpillwayConfig, IntakeConfig, ElevationVolumePoint,
    ElevationQPoint, ElevationFactorPoint, SegmentNeedle
)

@dataclass
class StepResult:
    t_hours: float
    inflow_m3s: float
    level_m: float
    volume_m3: float
    q1_m3s: float
    q2_m3s: float
    q_intake_m3s: float
    q_total_m3s: float
    coeff1: float
    coeff2: float
    vel_head1: float
    vel_head2: float

@dataclass
class Summary:
    max_level_m: float
    max_stored_hm3: float
    peak_inflow_m3s: float
    t_peak_inflow_h: float
    peak_outflow_m3s: float
    t_peak_outflow_h: float
    peak_q1_m3s: float
    peak_q2_m3s: float
    coeff1_at_peak: float
    coeff2_at_peak: float

# ---------- interpolation helpers (piecewise linear) ----------

def _sort_unique(xs: np.ndarray) -> np.ndarray:
    """Return argsort of unique xs while preserving monotonicity."""
    order = np.argsort(xs)
    xs_sorted = xs[order]
    # Make strictly monotonic by tiny epsilon if equal (avoid /0 in lininterp)
    eps = 1e-12
    for i in range(1, len(xs_sorted)):
        if xs_sorted[i] <= xs_sorted[i-1]:
            xs_sorted[i] = xs_sorted[i-1] + eps
    return order, xs_sorted

def interp_y_from_x(table_x: np.ndarray, table_y: np.ndarray, x: float) -> Optional[float]:
    """Linear interpolation y(x). Returns None if x is out of bounds."""
    if len(table_x) < 2:
        return None
    order, xs = _sort_unique(table_x.copy())
    ys = table_y[order]
    if x < xs.min() or x > xs.max():
        return None
    j = np.searchsorted(xs, x)
    if j == 0:
        return float(ys[0])
    if j >= len(xs):
        return float(ys[-1])
    x1, x2 = xs[j-1], xs[j]
    y1, y2 = ys[j-1], ys[j]
    w = (x - x1) / (x2 - x1)
    return float(y1 + w * (y2 - y1))

# Legacy VB has two directions on ELCAP: elevation<->volume
def level_from_volume(elev_volume: List[ElevationVolumePoint], volume_m3: float) -> Optional[float]:
    vol = np.array([p.volume_m3 for p in elev_volume], dtype=float)
    lev = np.array([p.elevation_m for p in elev_volume], dtype=float)
    return interp_y_from_x(vol, lev, volume_m3)

def volume_from_level(elev_volume: List[ElevationVolumePoint], level_m: float) -> Optional[float]:
    vol = np.array([p.volume_m3 for p in elev_volume], dtype=float)
    lev = np.array([p.elevation_m for p in elev_volume], dtype=float)
    return interp_y_from_x(lev, vol, level_m)

def q_from_level(curve: Optional[List[ElevationQPoint]], level_m: float) -> Optional[float]:
    if not curve:
        return None
    x = np.array([p.elevation_m for p in curve], dtype=float)
    y = np.array([p.q_m3s for p in curve], dtype=float)
    return interp_y_from_x(x, y, level_m)

def factor_from_level(curve: Optional[List[ElevationFactorPoint]], level_m: float) -> Optional[float]:
    if not curve:
        return None
    x = np.array([p.elevation_m for p in curve], dtype=float)
    y = np.array([p.factor for p in curve], dtype=float)
    return interp_y_from_x(x, y, level_m)

# ---------- VB polynomial helpers (ported) ----------

def calccT(P_over_Ho: float, variant: int) -> float:
    """Talud inclinado correction CT(P/Ho) with 3 variants (1..3)."""
    x = P_over_Ho
    if variant == 1:
        return (
            0.0000192915 * x**5 + 0.0000652576 * x**4 - 0.0037167326 * x**3
            + 0.0139414817 * x**2 - 0.020983809 * x + 1.0130887975
        )
    if variant == 2:
        return (
            -0.0007284603 * x**5 + 0.0055058805 * x**4 - 0.0224162021 * x**3
            + 0.0497832864 * x**2 - 0.0633607576 * x + 1.0372571395
        )
    if variant == 3:
        return (
            0.0192355971 * x**6 - 0.1190570309 * x**5 + 0.30854246 * x**4
            - 0.4449669487 * x**3 + 0.4010850905 * x**2 - 0.23510752 * x + 1.0688143858
        )
    return 1.0

def calccn(P_over_Ho: float) -> float:
    """Automatic discharge coefficient vs P/Ho (two polynomials split at 0.4692)."""
    x = P_over_Ho
    if x <= 0.4692:
        return (
            59.3158102091401 * x**5 - 79.563679870761 * x**4 + 40.7822465493219 * x**3
            - 11.3206744409882 * x**2 + 2.51214938629264 * x + 1.70025038012531
        )
    else:
        return (
            0.00664612505920559 * x**5 - 0.0667824975071266 * x**4 + 0.261529474920016 * x**3
            - 0.507105271566364 * x**2 + 0.512670950912138 * x + 1.93997699308661
        )

# ---------- Weir discharge with options ----------

def weir_Q(
    cfg: SpillwayConfig,
    level_m: float,
    gravity: float,
    velocity_head_enabled: bool
) -> Tuple[float, float, float]:
    """
    Compute discharge Q and effective coefficient for a spillway (one of the three modes).
    Returns (Q, coeff_used, velocity_head) where velocity_head (Ha) depends on Q (iterative).
    """
    crest = cfg.crest_m
    L = float(cfg.length_m)
    P = float(cfg.approach_depth_m)
    Ho = max(cfg.design_head_m, 1e-9)

    # Get base coefficient
    if cfg.auto_coefficient or cfg.discharge_coefficient is None:
        base_C = calccn(P / Ho)
    else:
        base_C = float(cfg.discharge_coefficient)

    # Optional slope correction
    if cfg.slope_correction_enabled:
        CT = calccT(P / Ho, cfg.slope_variant)
        base_C *= CT

    # Helper to compute segment Q with optional ELS and vel head loop
    def segment_Q(segment_crest: float, segment_length: float) -> Tuple[float, float]:
        # Iteration for velocity head feedback
        Ha = 0.0
        C_eff = base_C
        if level_m <= segment_crest:
            return 0.0, 0.0

        # Launder/submergence factor
        if cfg.els_enabled and cfg.els_curve:
            fac = factor_from_level(cfg.els_curve, level_m)
            if fac is None:
                # if no factor available, treat as configuration error (equivalent to legacy behavior stopping calc)
                raise ValueError("ELS factor missing for current level (spillway).")
            C_eff *= fac

        # Iterate: Ha depends on Q, which depends on Ha
        for _ in range(100):
            H = max(level_m - segment_crest + Ha, 0.0)
            Q = C_eff * segment_length * (H ** 1.5)
            if velocity_head_enabled and (P + level_m - segment_crest) > 0 and segment_length > 0:
                # Ha = (v^2)/(2g) with v = Q / (L * (P + E - crest))
                v = Q / (segment_length * (P + level_m - segment_crest))
                Ha_new = (v * v) / (2.0 * gravity)
            else:
                Ha_new = 0.0
            if abs(Ha_new - Ha) < 1e-10:
                Ha = Ha_new
                break
            Ha = Ha_new

        return Q, Ha

    if cfg.mode == "CONTROLADO":
        q = q_from_level(cfg.policy_q, level_m)
        if q is None:
            # No policy value at this level -> zero
            return 0.0, 0.0, 0.0
        return float(q), base_C, 0.0

    elif cfg.mode == "CON_AGUJAS":
        if not cfg.needles or len(cfg.needles) == 0:
            return 0.0, base_C, 0.0
        total_q = 0.0
        last_Ha = 0.0
        for seg in cfg.needles:
            qi, hai = segment_Q(seg.crest_m, seg.length_m)
            total_q += qi
            last_Ha = hai
        return total_q, base_C, last_Ha

    else:  # LIBRE
        q, ha = segment_Q(crest, L)
        return q, base_C, ha

# ---------- Intake (obra de toma) ----------

def intake_Q(intake: IntakeConfig, level_m: float) -> float:
    if intake.mode == "OFF":
        return 0.0
    if intake.mode == "CONSTANT":
        return float(intake.q_constant_m3s)
    if intake.mode == "TABLE":
        q = q_from_level(intake.q_vs_level, level_m)
        return float(q) if q is not None else 0.0
    return 0.0

# ---------- Router ----------

def _uniform_dt_from_inflow(times: np.ndarray) -> float:
    dts = np.diff(times)
    if len(dts) == 0:
        return 0.0
    # Check uniformity within 1e-9
    if np.max(np.abs(dts - dts[0])) > 1e-9:
        raise ValueError("Inflow time step dt must be uniform.")
    return float(dts[0])

def simulate(req: SimulationRequest) -> Tuple[List[StepResult], Summary]:
    g = req.gravity

    # Prepare curves (ensure lists)
    elev_volume = list(sorted(req.elev_volume, key=lambda p: p.volume_m3))
    if len(elev_volume) < 2:
        raise ValueError("elev_volume requires at least 2 points.")

    # Inflow
    inflow_pts = list(sorted(req.inflow, key=lambda p: p.t_hours))
    t = np.array([p.t_hours for p in inflow_pts], dtype=float)
    q_in = np.array([p.q_m3s for p in inflow_pts], dtype=float)
    if req.dt_hours is not None:
        dt_h = float(req.dt_hours)
    else:
        dt_h = _uniform_dt_from_inflow(t)
    if dt_h <= 0:
        raise ValueError("Invalid dt_hours.")

    # Initial state
    if req.initial_volume_m3 is not None:
        V = float(req.initial_volume_m3)
        E = level_from_volume(elev_volume, V)
        if E is None:
            raise ValueError("Initial volume is out of ELCAP bounds.")
    elif req.initial_level_m is not None:
        E = float(req.initial_level_m)
        V = volume_from_level(elev_volume, E)
        if V is None:
            raise ValueError("Initial level is out of ELCAP bounds.")
    else:
        # Default to min curve elevation
        E = float(min(p.elevation_m for p in elev_volume))
        V = volume_from_level(elev_volume, E) or float(min(p.volume_m3 for p in elev_volume))

    sp1 = req.spillway1
    sp2 = req.spillway2

    results: List[StepResult] = []

    # Helper to compute outflows for a given level
    def outflows(level: float) -> Tuple[float, float, float, float, float]:
        q1, c1, ha1 = weir_Q(sp1, level, g, req.velocity_head_enabled)
        q2 = 0.0
        c2 = 0.0
        ha2 = 0.0
        if sp2 is not None:
            q2, c2, ha2 = weir_Q(sp2, level, g, req.velocity_head_enabled)
        q_ot = intake_Q(req.intake, level)
        q_tot = q1 + q2 + q_ot
        return q1, q2, q_ot, q_tot, c1 if q1>0 else 0.0, c2 if q2>0 else 0.0, ha1, ha2

    # March over inflow hydrograph
    V_prev = V
    level_init = level_from_volume(elev_volume, V_prev) or 0.0
    q1, q2, q_ot, q_tot, c1, c2, ha1, ha2 = outflows(level_init)
    q_prev_total = q_tot
    t0 = float(t[0])

    for i in range(1, len(t)):
        t1 = float(t[i-1])
        t2 = float(t[i])
        I1 = float(q_in[i-1])
        I2 = float(q_in[i])

        print(f"I1={I1}, I2={I2}, q_prev_total={q_prev_total}, dt_h={dt_h}, V_prev={V_prev}")
        V_guess = V_prev + 0.5 * ((I1 + I2) - 2 * q_prev_total) * (dt_h * 3600.0)
        # Protección para evitar volumen negativo
        V_guess = max(V_guess, min(p.volume_m3 for p in elev_volume))
        # Iterate
        for k in range(req.max_iter_step):
            level = level_from_volume(elev_volume, V_guess)
            if level is None:
                print(f"V_guess fuera de rango: {V_guess}, min={min(p.volume_m3 for p in elev_volume)}, max={max(p.volume_m3 for p in elev_volume)}")
                raise ValueError("Volume went out of ELCAP bounds during iteration.")
            q1, q2, q_ot, q_tot, c1, c2, ha1, ha2 = outflows(level)
            V_next = V_prev + 0.5 * ((I1 + I2) - (q_prev_total + q_tot)) * (dt_h * 3600.0)
            # Protección para evitar volumen negativo
            V_next = max(V_next, min(p.volume_m3 for p in elev_volume))
            if abs(V_next - V_guess) < req.tolerance:
                V_guess = V_next
                break
            V_guess = V_next

        # Commit step
        V_prev = V_guess
        level = level_from_volume(elev_volume, V_prev) or 0.0
        q1, q2, q_ot, q_tot, c1, c2, ha1, ha2 = outflows(level)
        q_prev_total = q_tot

        results.append(StepResult(
            t_hours=t2, inflow_m3s=I2, level_m=level, volume_m3=V_prev,
            q1_m3s=q1, q2_m3s=q2, q_intake_m3s=q_ot, q_total_m3s=q_tot,
            coeff1=c1, coeff2=c2, vel_head1=ha1, vel_head2=ha2
        ))

    # Tail drain
    if req.drain_tail:
        crest_min = sp1.crest_m if sp2 is None else min(sp1.crest_m, sp2.crest_m)
        hours_accum = 0.0
        while True:
            level = level_from_volume(elev_volume, V_prev) or 0.0
            if level <= crest_min + req.tail_margin_m:
                break
            q1, q2, q_ot, q_tot, c1, c2, ha1, ha2 = outflows(level)
            V_next = V_prev - (q_tot) * (dt_h * 3600.0)
            # Prevent negative or falling out of bounds
            if V_next < (min(p.volume_m3 for p in elev_volume)):
                break
            V_prev = V_next
            t0 += dt_h
            hours_accum += dt_h
            results.append(StepResult(
                t_hours=t0, inflow_m3s=0.0, level_m=level, volume_m3=V_prev,
                q1_m3s=q1, q2_m3s=q2, q_intake_m3s=q_ot, q_total_m3s=q_tot,
                coeff1=c1, coeff2=c2, vel_head1=ha1, vel_head2=ha2
            ))
            if req.tail_hours_limit is not None and hours_accum >= req.tail_hours_limit:
                break

    # Summary stats
    max_level = max(r.level_m for r in results) if results else (level_from_volume(elev_volume, V) or 0.0)
    max_vol = max(r.volume_m3 for r in results) if results else V
    hm3 = max_vol / 1e6
    # Peaks
    peak_inflow_idx = int(np.argmax(q_in))
    peak_inflow = float(q_in[peak_inflow_idx])
    t_peak_inflow = float(t[peak_inflow_idx])
    out_tot = np.array([r.q_total_m3s for r in results]) if results else np.array([0.0])
    peak_outflow_idx = int(np.argmax(out_tot))
    peak_outflow = float(out_tot[peak_outflow_idx])
    t_peak_outflow = float(results[peak_outflow_idx].t_hours) if results else float(t[-1])
    peak_q1 = max((r.q1_m3s for r in results), default=0.0)
    peak_q2 = max((r.q2_m3s for r in results), default=0.0)
    coeff1_at_peak = results[peak_outflow_idx].coeff1 if results else 0.0
    coeff2_at_peak = results[peak_outflow_idx].coeff2 if results else 0.0

    summary = Summary(
        max_level_m=max_level,
        max_stored_hm3=hm3,
        peak_inflow_m3s=peak_inflow,
        t_peak_inflow_h=t_peak_inflow,
        peak_outflow_m3s=peak_outflow,
        t_peak_outflow_h=t_peak_outflow,
        peak_q1_m3s=peak_q1,
        peak_q2_m3s=peak_q2,
        coeff1_at_peak=coeff1_at_peak,
        coeff2_at_peak=coeff2_at_peak
    )

    return results, summary
