
from __future__ import annotations
from pydantic import BaseModel, Field, conlist
from typing import List, Optional, Literal

SpillwayMode = Literal["LIBRE", "CON_AGUJAS", "CONTROLADO"]
IntakeMode = Literal["OFF", "CONSTANT", "TABLE"]

class XY(BaseModel):
    x: float
    y: float

class ElevationVolumePoint(BaseModel):
    elevation_m: float
    volume_m3: float

class ElevationQPoint(BaseModel):
    elevation_m: float
    q_m3s: float

class ElevationFactorPoint(BaseModel):
    elevation_m: float
    factor: float

class SegmentNeedle(BaseModel):
    length_m: float
    crest_m: float

class HydroPoint(BaseModel):
    t_hours: float
    q_m3s: float

class SpillwayConfig(BaseModel):
    mode: SpillwayMode = "LIBRE"
    crest_m: float = Field(..., description="Crest elevation (Mo1/Mo2)")
    design_head_m: float = Field(..., description="Ho1/Ho2 (for coefficient corrections)")
    length_m: float = Field(..., description="Crest length L1/L2")
    approach_depth_m: float = Field(0.0, description="P1/P2 used in velocity-head correction equation")
    discharge_coefficient: Optional[float] = Field(None, description="If omitted and `auto_coefficient=True`, it is computed via calccn(P/Ho)")
    auto_coefficient: bool = True
    slope_correction_enabled: bool = False
    slope_variant: Literal[1,2,3] = 1
    els_enabled: bool = False
    els_curve: Optional[List[ElevationFactorPoint]] = None
    needles: Optional[List[SegmentNeedle]] = None
    policy_q: Optional[List[ElevationQPoint]] = None  # For CONTROLADO

class IntakeConfig(BaseModel):
    mode: IntakeMode = "OFF"
    q_constant_m3s: float = 0.0
    q_vs_level: Optional[List[ElevationQPoint]] = None  # NOTE: in legacy VB TOMA csv is (q,elev)

class SimulationRequest(BaseModel):
    gravity: float = 9.81
    tolerance: float = 1e-5
    max_iter_step: int = 1000

    # Level–Storage curve
    elev_volume: List[ElevationVolumePoint]

    # Inflow hydrograph
    inflow: List[HydroPoint]
    dt_hours: Optional[float] = None  # If None, computed from inflow times (must be uniform)

    # Spillways
    spillway1: SpillwayConfig
    spillway2: Optional[SpillwayConfig] = None

    # Intake
    intake: IntakeConfig = IntakeConfig()

    # Velocity head correction
    velocity_head_enabled: bool = False

    # Initial state
    initial_level_m: Optional[float] = None
    initial_volume_m3: Optional[float] = None

    # Continue draining after hydrograph until level <= min crest + margin
    drain_tail: bool = True
    tail_margin_m: float = 0.02
    tail_hours_limit: Optional[float] = None  # if set, limits tail extension
