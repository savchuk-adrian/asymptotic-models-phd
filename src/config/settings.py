from dataclasses import dataclass
from typing import Literal, TypedDict
import numpy as np

class BaseIncidentConfig(TypedDict):
    u_inc_name: str
    d: np.ndarray  

class IncidentFieldConfigTD(BaseIncidentConfig):
    omega: float   
    sigma: float   
    A: float    

class IncidentFieldConfigFD(BaseIncidentConfig):
    omega: complex

@dataclass
class BaseSimulationConfig:
    centers: list[tuple[float, float, float]] 
    geometry: Literal["sphere", "ellipsoid", "ellipsoid_cut"]
    directions: list[tuple[float, float, float]]
    angles: list[float]
    inc_field: BaseIncidentConfig 

@dataclass
class SimulationConfigTD(BaseSimulationConfig):
    inc_field: IncidentFieldConfigTD
    T: float
    dt: float 
    multistep_method: Literal["BDF1", "BDF2", "Trapezoidal"]
    x_0: np.ndarray

@dataclass
class SimulationConfigFD(BaseSimulationConfig):
    inc_field: IncidentFieldConfigFD
    frequency: complex  
    x_0: np.ndarray

@dataclass
class SimulationConfigPlaneTD(SimulationConfigTD):
    limits: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    N_grid: int
    plane: str

@dataclass
class SimulationConfigPlaneFD(SimulationConfigFD):
    limits: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    N_grid: int
    plane: str

