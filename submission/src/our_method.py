"""Our simple method: Weighted Tissue fitting.

The prediction formula is exactly the Tissue formula. The only change is the
fitting objective, which gives later training steps larger residual weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.tissue_model import compute_s1, compute_s2, tissue_predict


PARAM_NAMES = ("L0", "A", "alpha", "C")
COSINE_CURVE = "M:100M_gpt_D:20B_scheduler:cosine_rope"
WSD_CURVE = "M:100M_gpt_D:20B_scheduler:wsd_rope"
CURVE_811 = "M:100M_gpt_D:20B_scheduler:811_rope"
EPS = 1e-12


@dataclass
class WeightedTissueFitResult:
    """Fitted Weighted Tissue parameters and optimizer diagnostics."""

    params: dict[str, float]
    gamma: float
    lambda_decay: float
    success: bool
    cost: float
    message: str
    nfev: int


def make_weights(n: int, gamma: float = 2.0) -> np.ndarray:
    """Create linearly increasing weights w_s = 1 + gamma * s / T."""

    if n <= 0:
        return np.asarray([], dtype=float)
    progress = np.linspace(0.0, 1.0, n)
    return 1.0 + float(gamma) * progress


def fit_weighted_tissue_model(
    lr: np.ndarray | list[float],
    loss: np.ndarray | list[float],
    gamma: float = 2.0,
    lambda_decay: float = 0.99,
) -> WeightedTissueFitResult:
    """Fit Tissue parameters with weighted least squares."""

    lr_values = _as_1d_float(lr)
    loss_values = _as_1d_float(loss)
    if lr_values.size != loss_values.size:
        raise ValueError("lr and loss must have the same length for Weighted Tissue fitting.")

    mask = np.isfinite(lr_values) & np.isfinite(loss_values) & (lr_values >= 0)
    if int(mask.sum()) < 10:
        raise ValueError("Not enough finite points to fit Weighted Tissue.")

    fit_lr = lr_values[mask]
    fit_loss = loss_values[mask]
    s1 = compute_s1(fit_lr)
    s2 = compute_s2(fit_lr, lambda_decay=lambda_decay)
    weights = make_weights(fit_loss.size, gamma=gamma)
    sqrt_weights = np.sqrt(weights)
    x0 = _initial_params(s1, fit_loss)
    lower, upper = _param_bounds(fit_loss)

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = _predict_from_features(s1, s2, theta)
        if not np.all(np.isfinite(pred)):
            return np.full_like(fit_loss, 1e6, dtype=float)
        return sqrt_weights * (pred - fit_loss)

    result = least_squares(
        residual,
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        x_scale=np.asarray([1.0, max(x0[1], 1e-6), 0.3, 1e5], dtype=float),
        max_nfev=3000,
    )

    params = {name: float(value) for name, value in zip(PARAM_NAMES, result.x)}
    return WeightedTissueFitResult(
        params=params,
        gamma=float(gamma),
        lambda_decay=float(lambda_decay),
        success=bool(result.success),
        cost=float(result.cost),
        message=str(result.message),
        nfev=int(result.nfev),
    )


def weighted_tissue_predict(
    lr: np.ndarray | list[float],
    params: dict[str, float] | list[float] | tuple[float, float, float, float],
    lambda_decay: float = 0.99,
) -> np.ndarray:
    """Predict loss with the weighted-fit Tissue parameters."""

    return tissue_predict(lr=lr, params=params, lambda_decay=lambda_decay)


def run_our_method_experiment(
    curves: list[Any],
    output_dirs: dict[str, Path],
    gamma: float = 2.0,
    lambda_decay: float = 0.99,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Fit Weighted Tissue on cosine and evaluate on cosine, WSD, and 811."""

    results_dir = Path(output_dirs["results"])
    figures_dir = Path(output_dirs["figures"])
    fit_curve = _find_curve(curves, COSINE_CURVE)
    wsd_curve = _find_curve(curves, WSD_CURVE)
    curve_811 = _find_curve(curves, CURVE_811)

    if fit_curve.lr is None or fit_curve.loss is None:
        raise ValueError("Cosine fit curve is missing lr or loss.")

    print("")
    print("Running our method: Weighted Tissue fitting")
    print(f"Fit curve: {fit_curve.name}")
    print(f"Main test curve: {wsd_curve.name}")
    print(f"Additional test curve: {curve_811.name}")
    print(f"Gamma: {gamma}")

    fit_result = fit_weighted_tissue_model(
        lr=fit_curve.lr,
        loss=fit_curve.loss,
        gamma=gamma,
        lambda_decay=lambda_decay,
    )
    print(
        "Fitted Weighted Tissue params: "
        + ", ".join(f"{key}={value:.8g}" for key, value in fit_result.params.items())
    )
    print(
        f"Optimizer success={fit_result.success}, "
        f"cost={fit_result.cost:.6g}, nfev={fit_result.nfev}"
    )

    eval_specs = [
        ("fit_cosine", "fit", fit_curve, "our_method_predictions_cosine.csv", "our_method_fit_cosine.png"),
        ("test_wsd", "main_test", wsd_curve, "our_method_predictions_wsd.csv", "our_method_predict_wsd.png"),
        ("test_811", "additional_test", curve_811, "our_method_predictions_811.csv", "our_method_predict_811.png"),
    ]

    metric_rows: list[dict[str, Any]] = []
    for label, split, curve, csv_name, fig_name in eval_specs:
        if curve.lr is None or curve.loss is None:
            raise ValueError(f"Curve is missing lr or loss: {curve.name}")

        pred = weighted_tissue_predict(
            lr=curve.lr,
            params=fit_result.params,
            lambda_decay=fit_result.lambda_decay,
        )
        metrics = evaluate_prediction(curve.loss, pred)
        row = {
            "label": label,
            "split": split,
            "curve": curve.name,
            "gamma": fit_result.gamma,
            "lambda_decay": fit_result.lambda_decay,
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
        title = "Weighted Tissue fit on cosine LRS" if split == "fit" else f"Weighted Tissue prediction on {curve.name}"
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

    save_metrics_csv(results_dir / "our_method_metrics.csv", metric_rows)
    print(f"Saved our method metrics: {results_dir / 'our_method_metrics.csv'}")
    return fit_result.params, metric_rows


def build_metrics_summary(results_dir: str | Path) -> pd.DataFrame:
    """Merge Tissue, MPL, and Weighted Tissue metrics into one summary CSV."""

    results_dir = Path(results_dir)
    specs = [
        ("Tissue", results_dir / "tissue_metrics.csv"),
        ("Simplified MPL", results_dir / "mpl_metrics.csv"),
        ("Weighted Tissue", results_dir / "our_method_metrics.csv"),
    ]
    rows: list[pd.DataFrame] = []
    for method, path in specs:
        df = pd.read_csv(path)
        subset = df[["curve", "split", "MAE", "MSE", "RMSE", "MAPE"]].copy()
        subset.insert(0, "method", method)
        rows.append(subset)

    summary = pd.concat(rows, ignore_index=True)
    output_path = results_dir / "metrics_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Saved metrics summary: {output_path}")
    return summary


def plot_method_comparison_wsd(results_dir: str | Path, figures_dir: str | Path) -> None:
    """Plot WSD ground truth against all three method predictions."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    tissue = pd.read_csv(results_dir / "tissue_predictions_wsd.csv")
    mpl = pd.read_csv(results_dir / "mpl_predictions_wsd.csv")
    weighted = pd.read_csv(results_dir / "our_method_predictions_wsd.csv")

    plt.figure(figsize=(10, 6))
    plt.plot(tissue["step"], tissue["loss_true"], label="Ground truth loss", linewidth=1.4)
    plt.plot(tissue["step"], tissue["loss_pred"], label="Tissue prediction", linewidth=1.4)
    plt.plot(mpl["step"], mpl["loss_pred"], label="Simplified MPL prediction", linewidth=1.4)
    plt.plot(weighted["step"], weighted["loss_pred"], label="Weighted Tissue prediction", linewidth=1.4)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Method comparison on WSD LRS")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path = figures_dir / "method_comparison_wsd.png"
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"Saved method comparison figure: {output_path}")


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
    """Save one curve's Weighted Tissue prediction table."""

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
    """Save Weighted Tissue metrics."""

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
    power_term = np.exp(-alpha * np.log(np.maximum(s1, EPS)))
    return l0 + a_value * power_term - c_value * s2


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
    c_value = 0.0
    return np.asarray([l0, a_value, alpha, c_value], dtype=float)


def _param_bounds(loss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    loss_max = max(float(np.max(loss)), 1.0)
    lower = np.asarray([0.0, 0.0, 1e-4, 0.0], dtype=float)
    upper = np.asarray([loss_max * 2.5, 1e4, 3.0, 1e8], dtype=float)
    return lower, upper


def _find_curve(curves: list[Any], curve_name: str) -> Any:
    for curve in curves:
        if curve.name == curve_name:
            return curve
    available = ", ".join(curve.name for curve in curves)
    raise KeyError(f"Curve not found: {curve_name}. Available curves: {available}")


def _as_1d_float(values: np.ndarray | list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)
