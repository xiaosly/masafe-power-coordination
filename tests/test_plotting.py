import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.figure2_render import plot


@pytest.mark.parametrize("lengths", [(1, 1, 1, 1), (24, 24, 24, 24), (24, 340, 600, 1200), (302, 411, 501, 607)])
def test_renderer_supports_short_and_different_length_curves(tmp_path, monkeypatch, lengths):
    figures = []
    close = plt.close
    monkeypatch.setattr(plt, "close", lambda figure: figures.append(figure))
    stats = {"C1": {}, "C3": {}}
    for case in stats:
        for method, length in zip(("proposed", "penalty", "unified", "unsafe"), lengths):
            values = np.linspace(1, 2, length if case == "C1" else max(1, length // 2))
            stats[case][method] = {metric: (values, values * 0.9, values * 1.1)
                                  for metric in ("reward", "global_cost", "local_cost")}
    output = tmp_path / "curves"
    plot(stats, output, events=(("C1", "global_cost", "unified", 18684, "conflict"),))
    assert output.with_suffix(".png").stat().st_size > 0
    assert output.with_suffix(".pdf").stat().st_size > 0
    axes = figures[-1].axes
    for axis in axes[:3]:
        assert sorted(len(line.get_xdata()) for line in axis.lines) == sorted(
            (length + (9 if length > 300 else 0)) // (10 if length > 300 else 1) for length in lengths)
    for figure in figures:
        close(figure)
