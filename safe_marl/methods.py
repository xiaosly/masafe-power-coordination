"""Public method names and their training entry points."""

METHODS = ("proposed", "mappo_pf", "mappo_lag", "mappo_vanilla")
LABELS = dict(zip(METHODS, ("Proposed", "MAPPO-PF", "MAPPO-Lag", "MAPPO-Vanilla")))
TRAIN_MODULES = {method: "safe_marl.train" for method in METHODS}
TRAIN_MODULES["mappo_lag"] = "mappo_lag.train"
