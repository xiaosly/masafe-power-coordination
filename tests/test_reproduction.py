import os
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts import evaluate_all, train_all, plot_training_curves
from safe_marl.methods import METHODS


def test_batch_training_routes_each_method_to_its_implementation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(train_all, "ROOT", tmp_path)
    monkeypatch.setattr(train_all.subprocess, "call", lambda command, **kwargs: calls.append(command) or 0)
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--cases", "C3", "--seeds", "1", "--jobs", "1"])
    with pytest.raises(SystemExit) as done:
        train_all.main()
    assert done.value.code == 0
    assert len(calls) == 4
    modules = {command[command.index("--method") + 1]: command[command.index("-m") + 1] for command in calls}
    assert modules["mappo_lag"] == "mappo_lag.train"
    assert all(modules[method] == "safe_marl.train" for method in METHODS if method != "mappo_lag")
    assert len(list((tmp_path / "results/logs").rglob("*.log"))) == 4


def test_batch_logs_are_separate_for_each_invocation(tmp_path, monkeypatch):
    monkeypatch.setattr(train_all, "ROOT", tmp_path)
    monkeypatch.setattr(train_all.subprocess, "call", lambda command, **kwargs: 0)
    monkeypatch.setattr(sys, "argv", ["train_all.py", "--cases", "C3", "--methods", "proposed", "--seeds", "1"])
    for _ in range(2):
        with pytest.raises(SystemExit) as done:
            train_all.main()
        assert done.value.code == 0
    assert len(list((tmp_path / "results/logs").glob("train_*"))) == 2
    assert len(list((tmp_path / "results/logs").rglob("*.log"))) == 2


@pytest.mark.parametrize("options,labels,mpc_count", [([], set(METHODS), 0),
                                                   (["--pretrained"], {"proposed"}, 0),
                                                   (["--mpc-only"], set(), 2)])
def test_batch_evaluation_labels_and_mpc_selection(tmp_path, monkeypatch, options, labels, mpc_count):
    calls = []
    monkeypatch.setattr(evaluate_all, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_all.subprocess, "call", lambda command, **kwargs: calls.append(command) or 0)
    monkeypatch.setattr(sys, "argv", ["evaluate_all.py", "--cases", "C3", "--seeds", "1", "--jobs", "1", *options])
    with pytest.raises(SystemExit) as done:
        evaluate_all.main()
    assert done.value.code == 0
    rl = [command for command in calls if "evaluation.evaluate_rl" in command]
    assert {command[command.index("--label") + 1] for command in rl} == labels
    assert len([command for command in calls if "baselines.run" in command]) == mpc_count
    for command in rl:
        label = command[command.index("--label") + 1]
        assert Path(command[command.index("--output") + 1]).parent.name == label


def test_paper_curve_archive_has_the_complete_comparison():
    root = Path(__file__).resolve().parents[1]
    path = Path(os.environ.get("MASAFE_CURVE_ARCHIVE", root / "results/paper/training_curves.npz"))
    if not path.is_file():
        pytest.skip("download the paper curves to check the optional release asset")
    with np.load(path, allow_pickle=False) as archive:
        assert len(archive.files) == 180
        for case in ("C1", "C2", "C3"):
            for method in METHODS:
                runs = plot_training_curves.paper_runs(archive, case, method)
                assert len(runs) == 5
                assert all(run["reward"].shape == (20004,) for run in runs)
                assert all(np.isfinite(values).all() for run in runs for values in run.values())
