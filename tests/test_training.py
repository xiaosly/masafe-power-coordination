import json

import numpy as np
import pytest
from scipy.io import loadmat

from safe_marl.train import main


@pytest.mark.parametrize("method", ("proposed", "mappo_pf", "mappo_lag", "mappo_vanilla"))
def test_short_training_run(tmp_path, method):
    output = tmp_path / "run"
    main(["--case", "C3", "--method", method, "--seed", "1", "--num_env_steps", "24",
          "--n_rollout_threads", "1", "--num_mini_batch", "1", "--output_dir", str(output)])
    metrics = loadmat(output / "training_metrics_C3.mat")
    assert metrics["rewards"].size == 1
    assert np.isfinite(metrics["local_costs_per_agent"]).all()
    assert (output / "models" / "actor_agent11.pt").exists()
    config = json.loads((output / "config.json").read_text())
    assert config["method"] == method and config["num_env_steps"] == 24
    from evaluation.ledger import evaluate
    from evaluation.policy import DeterministicPolicy
    from power_envs import CASES
    env = CASES["C3"]()
    policy = DeterministicPolicy(env, output / "models")
    result = evaluate("C3", 20261005, method, controller=policy)
    assert len(result["steps"]) == 24 and np.isfinite(result["reward"])
