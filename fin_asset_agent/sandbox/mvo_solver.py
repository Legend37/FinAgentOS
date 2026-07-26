import numpy as np
from scipy.optimize import minimize
from typing import List, Dict
from config import mvo as mvo_config


class MVOSolver:
    """多策略投资组合优化器：支持 MVO / L2正则MVO / 风险平价 / 最小方差 / 最大分散度。

    默认使用 risk_parity（风险平价），因其对收益预测误差不敏感，天然分散，
    比纯 MVO 更适合与 LLM 选品结合（LLM 的收益预测精度有限）。
    """

    def __init__(self, risk_free_rate: float = None, risk_score: int = 60,
                 method: str = "risk_parity", l2_lambda: float = 0.5):
        self.rf = risk_free_rate if risk_free_rate is not None else mvo_config.risk_free_rate
        # risk_score 1-100 → risk_aversion: 高分=低厌恶, 低分=高厌恶
        self.risk_aversion = max(0.1, (100 - risk_score) / 20)
        self.method = method.lower().strip()
        self.l2_lambda = l2_lambda  # L2 正则化强度

    def optimize_portfolio(self, expected_returns: List[float],
                           cov_matrix: List[List[float]]) -> Dict[str, float]:
        returns_array = np.array(expected_returns, dtype=float)
        cov_array = np.array(cov_matrix, dtype=float)
        num_assets = len(returns_array)

        if self.method == "mvo":
            weights = self._solve_mvo(returns_array, cov_array, num_assets)
        elif self.method == "mvo_l2":
            weights = self._solve_mvo_l2(returns_array, cov_array, num_assets)
        elif self.method == "min_variance":
            weights = self._solve_min_variance(cov_array, num_assets)
        elif self.method == "max_div":
            weights = self._solve_max_diversification(cov_array, num_assets)
        else:
            # 默认：风险平价（对收益预测最不敏感，最适合 LLM 场景）
            weights = self._solve_risk_parity(cov_array, num_assets)

        port_var = float(weights.T @ cov_array @ weights)
        return {
            "optimal_weights": weights.tolist(),
            "minimized_variance": port_var,
            "method": self.method,
        }

    # ------------------------------------------------------------------
    # 1) 经典 MVO（Markowitz）
    # ------------------------------------------------------------------
    def _solve_mvo(self, returns, cov, n):
        init = np.ones(n) / n

        def objective(w):
            port_ret = w @ returns
            port_var = w.T @ cov @ w
            return -(port_ret - self.risk_aversion * port_var)

        return self._optimize(objective, n)

    # ------------------------------------------------------------------
    # 2) L2 正则化 MVO（防止极端集中）
    # ------------------------------------------------------------------
    def _solve_mvo_l2(self, returns, cov, n):
        init = np.ones(n) / n

        def objective(w):
            port_ret = w @ returns
            port_var = w.T @ cov @ w
            l2_penalty = self.l2_lambda * np.sum((w - 1.0 / n) ** 2)
            return -(port_ret - self.risk_aversion * port_var) + l2_penalty

        return self._optimize(objective, n)

    # ------------------------------------------------------------------
    # 3) 风险平价（Risk Parity）：各资产对组合风险的边际贡献相等
    # ------------------------------------------------------------------
    def _solve_risk_parity(self, cov, n):
        """最小化风险贡献偏离度，同时保持全投资约束。"""
        init = np.ones(n) / n

        def objective(w):
            port_var = w.T @ cov @ w
            if port_var < 1e-12:
                return 1e12
            marginal = cov @ w
            risk_contrib = w * marginal / port_var
            target = 1.0 / n
            return np.sum((risk_contrib - target) ** 2)

        return self._optimize(objective, n, max_iter=500)

    # ------------------------------------------------------------------
    # 4) 最小方差（Minimum Variance）：仅最小化风险
    # ------------------------------------------------------------------
    def _solve_min_variance(self, cov, n):
        init = np.ones(n) / n

        def objective(w):
            return w.T @ cov @ w

        return self._optimize(objective, n)

    # ------------------------------------------------------------------
    # 5) 最大分散度（Maximum Diversification）
    #    目标：max  (w'σ) / sqrt(w'Σw)   →  等价于 min - (w'σ) / sqrt(w'Σw)
    # ------------------------------------------------------------------
    def _solve_max_diversification(self, cov, n):
        vols = np.sqrt(np.diag(cov))
        init = np.ones(n) / n

        def objective(w):
            port_var = w.T @ cov @ w
            if port_var < 1e-12:
                return 1e12
            return -(w @ vols) / np.sqrt(port_var)

        return self._optimize(objective, n)

    # ------------------------------------------------------------------
    # 通用优化器包装
    # ------------------------------------------------------------------
    def _optimize(self, objective, n: int, max_iter: int = 300):
        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0})
        bounds = tuple((0.0, 1.0) for _ in range(n))
        options = {"maxiter": max_iter, "ftol": 1e-9}

        result = minimize(
            objective,
            np.ones(n) / n,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        if result.success:
            w = result.x
            # 清理极小权重（<0.1% 视为噪声置零后重归一化）
            w = np.where(w < 0.001, 0, w)
            if w.sum() > 0:
                w = w / w.sum()
            return w

        # fallback：优化失败时回退等权
        print(f"[MVOSolver] {self.method} 优化失败（{result.message}），回退等权")
        return np.ones(n) / n
