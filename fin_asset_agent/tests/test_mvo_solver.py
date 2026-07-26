import pytest
import numpy as np
from sandbox.mvo_solver import MVOSolver


class TestMVOSolver:
    """MVO 求解器单元测试 — 纯确定性计算，不依赖任何外部 API"""

    # ── 权重约束 ──

    def test_weights_sum_to_one(self):
        """输出权重之和必须严格等于 1.0"""
        solver = MVOSolver()
        returns = [0.12, 0.08, 0.15]
        cov = [[0.04, 0.01, 0.02],
               [0.01, 0.03, 0.01],
               [0.02, 0.01, 0.05]]
        result = solver.optimize_portfolio(returns, cov)
        w = result["optimal_weights"]

        assert abs(sum(w) - 1.0) < 1e-5, f"权重和应为 1.0，实际为 {sum(w)}"

    def test_all_weights_in_bounds(self):
        """每项权重必须在 [0, 1] 区间内（做多约束）"""
        solver = MVOSolver()
        returns = [0.10, 0.14, 0.06]
        cov = [[0.05, 0.00, 0.01],
               [0.00, 0.06, 0.02],
               [0.01, 0.02, 0.04]]
        result = solver.optimize_portfolio(returns, cov)
        w = result["optimal_weights"]

        for i, wi in enumerate(w):
            assert 0.0 <= wi <= 1.0, f"资产 {i} 权重 {wi} 超出 [0,1] 范围"

    # ── 最优化有效性 ──

    def test_optimization_reduces_variance(self):
        """MVO 优化后的风险调整收益应优于等权重方案"""
        solver = MVOSolver(risk_score=60, method="mvo")  # risk_aversion = 2.0
        returns = [0.12, 0.08, 0.15]
        cov = [[0.04, 0.01, 0.02],
               [0.01, 0.03, 0.01],
               [0.02, 0.01, 0.05]]
        result = solver.optimize_portfolio(returns, cov)
        opt_weights = np.array(result["optimal_weights"])
        cov_arr = np.array(cov)
        returns_arr = np.array(returns)

        opt_return = opt_weights @ returns_arr
        opt_var = opt_weights.T @ cov_arr @ opt_weights
        opt_utility = opt_return - solver.risk_aversion * opt_var

        equal_weights = np.ones(3) / 3
        eq_return = equal_weights @ returns_arr
        eq_var = equal_weights.T @ cov_arr @ equal_weights
        eq_utility = eq_return - solver.risk_aversion * eq_var

        assert opt_utility >= eq_utility - 1e-9, (
            f"优化后风险调整收益 {opt_utility:.6f} 应不低于等权重 {eq_utility:.6f}"
        )

    def test_returns_minimized_variance(self):
        """返回的 minimized_variance 与实际计算值一致"""
        solver = MVOSolver()
        returns = [0.14, 0.09]
        cov = [[0.06, 0.01],
               [0.01, 0.04]]
        result = solver.optimize_portfolio(returns, cov)
        w = np.array(result["optimal_weights"])
        cov_arr = np.array(cov)

        actual_var = float(w.T @ cov_arr @ w)

        assert abs(result["minimized_variance"] - actual_var) < 1e-8

    # ── 边界情况 ──

    def test_single_asset(self):
        """只有一个资产时，权重必须为 1.0"""
        solver = MVOSolver()
        returns = [0.10]
        cov = [[0.04]]
        result = solver.optimize_portfolio(returns, cov)

        assert result["optimal_weights"] == pytest.approx([1.0])
        assert result["minimized_variance"] == pytest.approx(0.04)

    def test_two_assets_corner(self):
        """两个资产：无风险资产在风险厌恶下占据主导权重"""
        solver = MVOSolver(risk_score=60, method="mvo")  # risk_aversion = 2.0
        returns = [0.05, 0.12]
        cov = [[0.0, 0.0],
               [0.0, 0.10]]
        result = solver.optimize_portfolio(returns, cov)

        # 在风险调整收益框架下，无风险资产权重应占主导（>80%）
        assert result["optimal_weights"][0] > 0.8
        assert abs(sum(result["optimal_weights"]) - 1.0) < 1e-5

    def test_identical_assets(self):
        """完全相同的资产应得到相近的权重"""
        solver = MVOSolver()
        returns = [0.10, 0.10, 0.10]
        cov = [[0.04, 0.02, 0.02],
               [0.02, 0.04, 0.02],
               [0.02, 0.02, 0.04]]
        result = solver.optimize_portfolio(returns, cov)
        w = result["optimal_weights"]

        # 对称输入，权重应近似相等
        for wi in w:
            assert wi == pytest.approx(1 / 3, abs=1e-3)

    # ── 风险偏好 ──

    def test_risk_free_rate_is_configurable(self):
        """无风险利率 + 风险评分均可配置"""
        solver_default = MVOSolver(method="mvo")
        solver_custom = MVOSolver(risk_free_rate=0.05, risk_score=95, method="mvo")

        returns = [0.12, 0.08]
        cov = [[0.05, 0.01], [0.01, 0.04]]

        r1 = solver_default.optimize_portfolio(returns, cov)
        r2 = solver_custom.optimize_portfolio(returns, cov)

        # 进取型(risk_score=95) vs 平衡型(60)，权重分布应有差异
        assert r1["optimal_weights"] != pytest.approx(r2["optimal_weights"])

    # ── 权重维度一致性 ──

    def test_output_length_matches_input(self):
        """输出权重数量与输入资产数量一致"""
        solver = MVOSolver()
        for n in [2, 3, 5]:
            returns = [0.1] * n
            cov = [[0.04 if i == j else 0.01 for j in range(n)] for i in range(n)]
            result = solver.optimize_portfolio(returns, cov)
            assert len(result["optimal_weights"]) == n, f"{n} 资产应返回 {n} 个权重"
