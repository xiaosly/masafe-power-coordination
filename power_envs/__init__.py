"""Multi-agent power-system environments with local and global operational constraints.

Paper case labels and environment classes:
    C1  TSODSOEnv        TSO-DSO coordination (3 agents)
    C2  MGVoltageEnv     microgrids in a distribution network (5 agents)
    C3  CarbonMarketEnv  prosumer energy community with carbon caps (12 agents)
"""
from power_envs.c1_tso_dso import TSODSOEnv
from power_envs.c2_microgrids import MGVoltageEnv
from power_envs.c3_carbon import CarbonMarketEnv

CASES = {"C1": TSODSOEnv, "C2": MGVoltageEnv, "C3": CarbonMarketEnv}

__all__ = ["CASES", "TSODSOEnv", "MGVoltageEnv", "CarbonMarketEnv"]
