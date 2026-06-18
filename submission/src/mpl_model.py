"""Simplified Luo et al. Multi-Power Law baseline.

This is a lightweight MPL-style baseline for the course project. It keeps the
core structure of a cumulative learning-rate power law plus an LR-decay
correction term, without reproducing every parameter and training detail from
the full paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


PARAM_NAMES = ("L0", "A", "alpha", "B")
EPS = 1e-12
COSINE_CURVE = "M:100M_gpt_D:20B_scheduler:cosine_rope"
WSD_CURVE = "M:100M_gpt_D:20B_scheduler:wsd_rope"
CURVE_811 = "M:100M_gpt_D:20B_scheduler:811_rope"


@dataclass
class MPLFitResult:
    """Fitted simplified MPL parameters and optimizer diagnostics."""

    params: dict[str, float]
    success: bool
    cost: float
    message: str
    nfev: int


def compute_s1(lr: np.ndarray | list[float]) -> np.ndarray:
    """Compute S1(t) = sum_{i=1}^t lr_i."""

    lr_values = _as_1d_float(lr)
    return np.maximum(np.cumsum(lr_values), EPS)


def compute_decay_feature(lr: np.ndarray | list[float]) -> np.ndarray:
    """Compute a simplified MPL learning-rate decay feature.

    The implemented feature is:

    D(t) = sum_{k=1}^t max(lr_{k-1} - lr_k, 0) * (S_tail(k,t) + eps)

    with beta fixed to 1 for stability. Since S_tail(k,t) is linear in S1, this
    can be computed exactly in O(n) rather than by a slow nested loop.
    """

    lr_values = _as_1d_float(lr)
    if lr_values.size == 0:
        return np.asarray([], dtype=float)

    s1 = compute_s1(lr_values)
    drops = np.zeros_like(lr_values, dtype=float)
    drops[1:] = np.maximum(lr_values[:-1] - lr_values[1:], 0.0)

    # For a drop at index k, S_tail(k,t) = S1[t] - S1[k-1].
    previous_s1 = np.concatenate(([0.0], s1[:-1]))
    cumulative_drops = np.cumsum(drops)
    cumulative_drop_offsets = np.cumsum(drops * previous_s1)
    decay_feature = s1 * cumulative_drops - cumulative_drop_offsets
    return np.maximum(decay_feature, 0.0)


def mpl_predict(
    lr: np.ndarray | list[float],
    params: dict[str, float] | list[float] | tuple[float, float, float, float],
) -> np.ndarray:
    """Predict loss using the simplified MPL baseline.

    L(t) = L0 + A * (S1(t) + eps)^(-alpha) - B * D(t)
    """

    lr_values = _as_1d_float(lr)
    theta = _params_to_array(params)
    s1 = compute_s1(lr_values)
    decay_feature = compute_decay_feature(lr_values)
    return _predict_from_features(s1, decay_feature, theta)


def fit_mpl_model(
    lr: np.ndarray | list[float],
    loss: np.ndarray | list[float],
) -> MPLFitResult:
    """Fit simplified MPL parameters L0, A, alpha, and B."""

    lr_values = _as_1d_float(lr)
    loss_values = _as_1d_float(loss)
    if lr_values.size != loss_values.size:
        raise ValueError("lr and loss must have the same length for MPL fitting.")

    mask = np.isfinite(lr_values) & np.isfinite(loss_values) & (lr_values >= 0)
    if int(mask.sum()) < 10:
        raise ValueError("Not enough finite points to fit MPL baseline.")

    fit_lr = lr_values[mask]
    fit_loss = loss_values[mask]
    s1 = compute_s1(fit_lr)
    decay_feature = compute_decay_feature(fit_lr)
    x0 = _initial_params(s1, fit_loss)
    lower, upper = _param_bounds(fit_loss)

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = _predict_from_features(s1, decay_feature, theta)
        if not np.all(np.isfinite(pred)):
            return np.full_like(fit_loss, 1e6, dtype=float)
        return pred - fit_loss

    result = least_squares(
        residual,
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        x_scale=np.asarray([1.0, max(x0[1], 1e-6), 0.3, 1e3], dtype=float),
        max_nfev=3000,
    )

    params = {name: float(value) for name, value in zip(PARAM_NAMES, result.x)}
    return MPLFitResult(
        params=params,
        success=bool(result.success),
        cost=float(result.cost),
        message=str(result.message),
        nfev=int(result.nfev),
    )


def run_mpl_experiment(curves: list[Any], output_dirs: dict[str, Path]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Fit simplified MPL on cosine and evaluate on cosine, WSD, and 811."""

    results_dir = Path(output_dirs["results"])
    figures_dir = Path(output_dirs["figures"])
    fit_curve = _find_curve(curves, COSINE_CURVE)
    wsd_curve = _find_curve(curves, WSD_CURVE)
    curve_811 = _find_curve(curves, CURVE_811)

    if fit_curve.lr is None or fit_curve.loss is None:
        raise ValueError("Cosine fit curve is missing lr or loss.")

    print("")
    print("Running simplified MPL baseline")
    print(f"Fit curve: {fit_curve.name}")
    print(f"Main test curve: {wsd_curve.name}")
    print(f"Additional test curve: {curve_811.name}")

    fit_result = fit_mpl_model(fit_curve.lr, fit_curve.loss)
    print(
        "Fitted MPL params: "
        + ", ".join(f"{key}={value:.8g}" for key, value in fit_result.params.items())
    )
    print(
        f"Optimizer success={fit_result.success}, "
        f"cost={fit_result.cost:.6g}, nfev={fit_result.nfev}"
    )

    eval_specs = [
        ("fit_cosine", "fit", fit_curve, "mpl_predictions_cosine.csv", "mpl_fit_cosine.png"),
        ("test_wsd", "main_test", wsd_curve, "mpl_predictions_wsd.csv", "mpl_predict_wsd.png"),
        ("test_811", "additional_test", curve_811, "mpl_predictions_811.csv", "mpl_predict_811.png"),
    ]

    metric_rows: list[dict[str, Any]] = []
    for label, split, curve, csv_name, fig_name in eval_specs:
        if curve.lr is None or curve.loss is None:
            raise ValueError(f"Curve is missing lr or loss: {curve.name}")

        pred = mpl_predict(curve.lr, fit_result.params)
        metrics = evaluate_prediction(curve.loss, pred)
        row = {
            "label": label,
            "split": split,
            "curve": curve.name,
            "optimizer_success": fit_result.success,
            "optimizer_cost": fit_result.cost,
            "optimizer_nfev": fit_result.nfev,
            **fit_result.params,
            **metrics,
        }
        metric_rows.append(row)

        save_prediction_csv(
            path=results_dir / csv_name,
            curve_name=curve.name,
            step=curve.step,
            lr=curve.lr,
            loss_true=curve.loss,
            loss_pred=pred,
        )
        title = "Simplified MPL fit on cosine LRS" if split == "fit" else f"Simplified MPL prediction on {curve.name}"
        plot_prediction(
            path=figures_dir / fig_name,
            title=title,
            step=curve.step,
            loss_true=curve.loss,
            loss_pred=pred,
            metrics=metrics,
        )
        print(
            f"{label}: MAE={metrics['MAE']:.6g}, "
            f"MSE={metrics['MSE']:.6g}, RMSE={metrics['RMSE']:.6g}, "
            f"MAPE={metrics['MAPE']:.4g}%"
        )

    save_metrics_csv(results_dir / "mpl_metrics.csv", metric_rows)
    print(f"Saved MPL metrics: {results_dir / 'mpl_metrics.csv'}")
    return fit_result.params, metric_rows


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
    denom = np.maximum(np.abs(true_values), EPS)
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
    """Save one curve's simplified MPL prediction table."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lr_values = _as_1d_float(lr)
    true_values = _as_1d_float(loss_true)
    pred_values = _as_1d_float(loss_pred)
    if step is None or np.asarray(step).size != true_values.size:
        step_values = np.arange(true_values.size)
    else:
        step_values = _as_1d_float(step)

    error = pred_values - true_values
    abs_error = np.abs(error)
    relative_error = abs_error / np.maximum(np.abs(true_values), EPS)
    df = pd.DataFrame(
        {
            "curve": curve_name,
            "step": step_values,
            "lr": lr_values,
            "loss_true": true_values,
            "loss_pred": pred_values,
            "error": error,
            "abs_error": abs_error,
            "relative_error": relative_error,
        }
    )
    df.to_csv(path, index=False)


def save_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Save simplified MPL metrics."""

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


def _predict_from_features(s1: np.ndarray, decay_feature: np.ndarray, theta: np.ndarray) -> np.ndarray:
    l0, a_value, alpha, b_value = theta
    power_term = np.exp(-alpha * np.log(np.maximum(s1, EPS)))
    return l0 + a_value * power_term - b_value * decay_feature


def _initial_params(s1: np.ndarray, loss: np.ndarray) -> np.ndarray:
    loss_start = float(loss[0])
    loss_end = float(loss[-1])
    loss_min = float(np.min(loss))
    alpha = 0.35
    power_start = float(s1[0] ** (-alpha))
    power_end = float(s1[-1] ** (-alpha))
    denom = max(power_start - power_end, EPS)
    l0 = max(loss_min * 0.9, 0.0)
    a_value = max((loss_start - loss_end) / denom, EPS)
    b_value = 0.0
    return np.asarray([l0, a_value, alpha, b_value], dtype=float)


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
        raise ValueError("MPL params must contain L0, A, alpha, and B.")
    return arr


def _find_curve(curves: list[Any], curve_name: str) -> Any:
    for curve in curves:
        if curve.name == curve_name:
            return curve
    available = ", ".join(curve.name for curve in curves)
    raise KeyError(f"Curve not found: {curve_name}. Available curves: {available}")


def _as_1d_float(values: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)
