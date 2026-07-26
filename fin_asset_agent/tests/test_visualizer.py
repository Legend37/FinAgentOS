import os
import tempfile
import pytest

# 在导入 visualizer 前确保 Agg 后端可用
matplotlib = pytest.importorskip("matplotlib")

from sandbox.visualizer import (
    efficient_frontier, allocation_pie, nav_curve, weights_comparison_bar,
)


def _tmp_png():
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    return path


def test_efficient_frontier_basic():
    path = _tmp_png()
    try:
        out = efficient_frontier(
            expected_returns=[0.10, 0.08, 0.15],
            cov_matrix=[[0.04, 0.01, 0.02], [0.01, 0.03, 0.01], [0.02, 0.01, 0.05]],
            n_samples=200,
            highlight_weights=[0.4, 0.3, 0.3],
            save_path=path,
        )
        assert out == path
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000  # PNG 至少几 KB
    finally:
        os.path.exists(path) and os.unlink(path)


def test_efficient_frontier_dimension_error():
    with pytest.raises(ValueError):
        efficient_frontier([0.1, 0.2], [[0.04]], n_samples=10)


def test_allocation_pie():
    path = _tmp_png()
    try:
        out = allocation_pie(["A", "B", "C"], [0.5, 0.3, 0.2], save_path=path)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 500
    finally:
        os.path.exists(path) and os.unlink(path)


def test_allocation_pie_dimension_error():
    with pytest.raises(ValueError):
        allocation_pie(["A"], [0.5, 0.5])


def test_nav_curve():
    nav = [{"date": f"2024-01-{i+1:02d}", "value": 1_000_000 + i * 1000} for i in range(20)]
    path = _tmp_png()
    try:
        out = nav_curve(nav, save_path=path)
        assert os.path.exists(out)
    finally:
        os.path.exists(path) and os.unlink(path)


def test_nav_curve_empty_raises():
    with pytest.raises(ValueError):
        nav_curve([])


def test_weights_comparison_bar():
    path = _tmp_png()
    try:
        out = weights_comparison_bar(
            labels=["茅台", "工行", "比亚迪"],
            base=[0.33, 0.34, 0.33],
            adjusted=[0.40, 0.30, 0.30],
            save_path=path,
        )
        assert os.path.exists(out)
    finally:
        os.path.exists(path) and os.unlink(path)


def test_weights_comparison_dimension_error():
    with pytest.raises(ValueError):
        weights_comparison_bar(["A", "B"], [0.5, 0.5], [0.3])
