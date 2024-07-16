from botorch.test_functions.multi_objective import (
    BraninCurrin,
    ZDT2,
    DTLZ2,
    DTLZ3,
    VehicleSafety,
    Penicillin,
    WeldedBeam,
    CarSideImpact,
    DiscBrake,
    MW7,
    GMM
) 
from .rover import Rover


BENCHMARKS = {
    "BraninCurrin": BraninCurrin,
    "GMM": GMM,
    "MW7": MW7,
    "ZDT2": ZDT2,
    "DTLZ2": DTLZ2,
    "DTLZ3": DTLZ3,
    "VehicleSafety": VehicleSafety,
    "Penicillin": Penicillin,
    "WeldedBeam": WeldedBeam,
    "CarSideImpact": CarSideImpact,
    "DiscBrake": DiscBrake,
    "Rover": Rover
}

BENCHMARKS_KWS = {
    "BraninCurrin": {},
    "GMM": {},
    "MW7": {},
    "ZDT2": {"dim": 6},
    "DTLZ2": {"dim": 10},
    "DTLZ3": {"dim": 100},
    "VehicleSafety": {},
    "Penicillin": {},
    "WeldedBeam": {},
    "CarSideImpact": {},
    "DiscBrake": {},
    "Rover": {}
}
