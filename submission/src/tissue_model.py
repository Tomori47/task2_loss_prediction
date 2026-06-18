"""Tissue et al. baseline for loss curve prediction.

Only the Tissue baseline is implemented here. This module does not include Luo
or Multi-Power Law variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


PARAM_NAMES = ("L0", "A", "alpha", "C")


@dataclass
class TissueFitResult:
    """Fitted Tissue parameters and optimization diagnostics."""

    params: dict[str, float]
    lambda_decay: float
    success: bool
    cost: float
    message: str
    nfev: int


def compute_s1(lr: np.ndarray | list[float]) -> np.ndarray:
    """Compute S1(s) = sum_{i=1}^s eta_i."""

    lr_values = _as_1d_float(lr)
    s1 = np.cumsum(lr_values)
    return np.maximum(s1, 1e-12)


def compute_s2(lr: np.ndarray | list[float], lambda_decay: float = 0.99) -> np.ndarray:
    """Compute the Tissue learning-rate annealing feature S2.

    S2(s) = sum_{i=1}^s sum_{k=1}^i (eta_{k-1} - eta_k) * lambda^(i-k)

    For a discrete lr array, this is evaluated with the equivalent recurrence:
    S2[t] = lambda * S2[t - 1] + lr[t - 1] - lr[t].
    """

    lr_values = _as_1d_float(lr)
    if lr_values.size == 0:
        return np.asarray([], dtype=float)

    s2 = np.zeros_like(lr_values, dtype=float)
    for i in range(1, lr_values.size):
        s2[i] = lambda_decay * s2[i - 1] + lr_values[i - 1] - lr_values[i]
    return s2


def tissue_predict(
    lr: np.ndarray | list[float],
    params: dict[str, float] | list[float] | tuple[float, float, float, float],
    lambda_decay: float = 0.99,
) -> np.ndarray:
    """Predict loss from a learning-rate schedule using Tissue parameters."""

    lr_values = _as_1d_float(lr)
    s1 = compute_s1(lr_values)
    s2 = compute_s2(lr_values, lambda_decay=lambda_decay)
    theta = _params_to_array(params)
    return _predict_from_features(s1, s2, theta)


def fit_tissue_model(
    lr: np.ndarray | list[float],
    loss: np.ndarray | list[float],
    lambda_decay: float = 0.99,
) -> TissueFitResult:
    """Fit L0, A, alpha, C on one loss curve."""

    lr_values = _as_1d_float(lr)
    loss_values = _as_1d_float(loss)
    if lr_values.size != loss_values.size:
        raise ValueError("lr and loss must have the same length for Tissue fitting.")

    mask = np.isfinite(lr_values) & np.isfinite(loss_values) & (lr_values >= 0)
    if int(mask.sum()) < 10:
        raise ValueError("Not enough finite points to fit Tissue baseline.")

    fit_lr = lr_values[mask]
    fit_loss = loss_values[mask]
    s1 = compute_s1(fit_lr)
    s2 = compute_s2(fit_lr, lambda_decay=lambda_decay)

    x0 = _initial_params(s1, fit_loss)
    lower, upper = _param_bounds(fit_loss)

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = _predict_from_features(s1, s2, theta)
        if not np.all(np.isfinite(pred)):
            return np.full_like(fit_loss, 1e6, dtype=float)
        return pred - fit_loss

    result = least_squares(
        residual,
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        x_scale=np.asarray([1.0, max(x0[1], 1e-6), 0.3, 1e5], dtype=float),
        max_nfev=3000,
    )

    params = {name: float(value) for name, value in zip(PARAM_NAMES, result.x)}
    return TissueFitResult(
        params=params,
        lambda_decay=float(lambda_decay),
        success=bool(result.success),
        cost=float(result.cost),
        message=str(result.message),
        nfev=int(result.nfev),
    )


def evaluate_prediction(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> dict[str, float]:
    """Compute MAE, MSE, RMSE, and MAPE."""

    true_values = _as_1d_float(y_true)
    pred_values = _as_1d_float(y_pred)
    if true_values.size != pred_values.size:
        raise ValueError("y_true and y_pred must have the same length.")

    mask = np.isfinite(true_values) & np.isfinite(pred_values)
    if int(mask.sum()) == 0:
        return {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "MAPE": np.nan}

    true_values = true_values[mask]
    pred_values = pred_values[mask]
    error = pred_values - true_values
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    denom = np.maximum(np.abs(true_values), 1e-12)
    mape = float(np.mean(np.abs(error) / denom) * 100.0)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}


def save_prediction_csv(
    path: str | Path,
    curve_name: str,
    step: np.ndarray | None,
    lr: np.ndarray,
    loss_true: np.ndarray,
    loss_pred: np.ndarray,
) -> None:
    """Save one curve's Tissue prediction table."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lr_values = _as_1d_float(lr)
    true_values = _as_1d_float(loss_true)
    pred_values = _as_1d_float(loss_pred)
    if step is None or np.asarray(step).size != true_values.size:
        step_values = np.arange(true_values.size)
    else:
        step_values = _as_1d_float(step)

    df = pd.DataFrame(
        {
            "curve": curve_name,
            "step": step_values,
            "lr": lr_values,
            "loss_true": true_values,
            "loss_pred": pred_values,
            "error": pred_values - true_values,
            "abs_error": np.abs(pred_values - true_values),
        }
    )
    df.to_csv(path, index=False)


def save_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Save Tissue metrics."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_prediction(
    path: str | Path,
    title: str,
    step: np.ndarray | None,
    loss_true: np.ndarray,
    loss_pred: np.ndarray,
    metrics: dict[str, float],
) -> None:
    """Plot ground-truth and predicted loss."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    true_values = _as_1d_float(loss_true)
    pred_values = _as_1d_float(loss_pred)
    if step is None or np.asarray(step).size != true_values.size:
        x_values = np.arange(true_values.size)
    else:
        x_values = _as_1d_float(step)

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, true_values, label="Ground truth loss", linewidth=1.3)
    plt.plot(x_values, pred_values, label="Predicted loss", linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    text = f"MAE = {metrics['MAE']:.4f}\nMAPE = {metrics['MAPE']:.2f}%"
    plt.gca().text(
        0.02,
        0.04,
        text,
        transform=plt.gca().transAxes,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _predict_from_features(s1: np.ndarray, s2: np.ndarray, theta: np.ndarray) -> np.ndarray:
    l0, a_value, alpha, c_value = theta
    power_term = np.exp(-alpha * np.log(np.maximum(s1, 1e-12)))
    return l0 + a_value * power_term - c_value * s2


def _initial_params(s1: np.ndarray, loss: np.ndarray) -> np.ndarray:
    loss_start = float(loss[0])
    loss_end = float(loss[-1])
    loss_min = float(np.min(loss))
    alpha = 0.35
    power_start = float(s1[0] ** (-alpha))
    power_end = float(s1[-1] ** (-alpha))
    denom = max(power_start - power_end, 1e-12)
    l0 = max(loss_min * 0.9, 0.0)
    a_value = max((loss_start - loss_end) / denom, 1e-10)
    c_value = 0.0
    return np.asarray([l0, a_value, alpha, c_value], dtype=float)


def _param_bounds(loss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    loss_max = max(float(np.max(loss)), 1.0)
    lower = np.asarray([0.0, 0.0, 1e-4, 0.0], dtype=float)
    upper = np.asarray([loss_max * 2.5, 1e4, 3.0, 1e8], dtype=float)
    return lower, upper


def _params_to_array(
    params: dict[str, float] | list[float] | tuple[float, float, float, float],
) -> np.ndarray:
    if isinstance(params, dict):
        return np.asarray([params[name] for name in PARAM_NAMES], dtype=float)
    arr = np.asarray(params, dtype=float).reshape(-1)
    if arr.size != 4:
        raise ValueError("Tissue params must contain L0, A, alpha, C.")
    return arr


def _as_1d_float(values: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)
