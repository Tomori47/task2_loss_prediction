"""Run data inspection and preview generation for Task 2."""

from pathlib import Path

from src.data_utils import (
    extract_curves,
    load_pkl,
    print_structure,
    save_data_summary,
    save_preview_plots,
)
from src.mpl_model import run_mpl_experiment
from src.our_method import (
    build_metrics_summary,
    plot_method_comparison_wsd,
    run_our_method_experiment,
)
from src.tissue_model import (
    evaluate_prediction,
    fit_tissue_model,
    plot_prediction,
    save_metrics_csv,
    save_prediction_csv,
    tissue_predict,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
DATA_FILE = DATA_DIR / "gpt_loss+lrs.pkl"
COSINE_CURVE = "M:100M_gpt_D:20B_scheduler:cosine_rope"
WSD_CURVE = "M:100M_gpt_D:20B_scheduler:wsd_rope"
CURVE_811 = "M:100M_gpt_D:20B_scheduler:811_rope"
LAMBDA_DECAY = 0.99


def scan_data_dir() -> list[Path]:
    """Print and return files found in data/."""

    print(f"Scanning data directory: {DATA_DIR}")
    if not DATA_DIR.exists():
        print("data/ directory is missing.")
        return []

    files = sorted(path for path in DATA_DIR.iterdir() if path.is_file())
    if not files:
        print("data/ directory is empty.")
        return []

    for path in files:
        print(f"- {path.name} ({path.stat().st_size} bytes)")
    return files


def find_curve(curves, curve_name: str):
    """Find a curve by its exact extracted name."""

    for curve in curves:
        if curve.name == curve_name:
            return curve
    available = ", ".join(curve.name for curve in curves)
    raise KeyError(f"Curve not found: {curve_name}. Available curves: {available}")


def run_tissue_baseline(curves) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Fit Tissue on cosine and evaluate on cosine, WSD, and 811."""

    fit_curve = find_curve(curves, COSINE_CURVE)
    wsd_curve = find_curve(curves, WSD_CURVE)
    curve_811 = find_curve(curves, CURVE_811)

    if fit_curve.lr is None or fit_curve.loss is None:
        raise ValueError("Cosine fit curve is missing lr or loss.")

    print("")
    print("Running Tissue baseline")
    print(f"Fit curve: {fit_curve.name}")
    print(f"Main test curve: {wsd_curve.name}")
    print(f"Additional test curve: {curve_811.name}")

    fit_result = fit_tissue_model(
        lr=fit_curve.lr,
        loss=fit_curve.loss,
        lambda_decay=LAMBDA_DECAY,
    )
    print(
        "Fitted Tissue params: "
        + ", ".join(f"{key}={value:.8g}" for key, value in fit_result.params.items())
    )
    print(
        f"Optimizer success={fit_result.success}, "
        f"cost={fit_result.cost:.6g}, nfev={fit_result.nfev}"
    )

    eval_specs = [
        ("fit_cosine", "fit", fit_curve, "tissue_predictions_cosine.csv", "tissue_fit_cosine.png"),
        ("test_wsd", "main_test", wsd_curve, "tissue_predictions_wsd.csv", "tissue_predict_wsd.png"),
        ("test_811", "additional_test", curve_811, "tissue_predictions_811.csv", "tissue_predict_811.png"),
    ]

    metric_rows: list[dict[str, float]] = []
    for label, split, curve, csv_name, fig_name in eval_specs:
        if curve.lr is None or curve.loss is None:
            raise ValueError(f"Curve is missing lr or loss: {curve.name}")

        pred = tissue_predict(
            lr=curve.lr,
            params=fit_result.params,
            lambda_decay=fit_result.lambda_decay,
        )
        metrics = evaluate_prediction(curve.loss, pred)
        row = {
            "label": label,
            "split": split,
            "curve": curve.name,
            "lambda_decay": fit_result.lambda_decay,
            "optimizer_success": fit_result.success,
            "optimizer_cost": fit_result.cost,
            "optimizer_nfev": fit_result.nfev,
            **fit_result.params,
            **metrics,
        }
        metric_rows.append(row)

        save_prediction_csv(
            path=RESULTS_DIR / csv_name,
            curve_name=curve.name,
            step=curve.step,
            lr=curve.lr,
            loss_true=curve.loss,
            loss_pred=pred,
        )
        title_kind = "Tissue fit on cosine LRS" if split == "fit" else f"Tissue prediction on {curve.name}"
        plot_prediction(
            path=FIGURES_DIR / fig_name,
            title=title_kind,
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

    save_metrics_csv(RESULTS_DIR / "tissue_metrics.csv", metric_rows)
    print(f"Saved Tissue metrics: {RESULTS_DIR / 'tissue_metrics.csv'}")
    return fit_result.params, metric_rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    scan_data_dir()
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Expected data file not found: {DATA_FILE}")

    print("")
    print(f"Loading data file: {DATA_FILE}")
    obj, load_method = load_pkl(DATA_FILE)
    print(f"Load method: {load_method}")
    print("")
    print("Raw data structure:")
    print_structure(obj)

    curves = extract_curves(obj)
    print("")
    print(f"Extracted {len(curves)} curve(s).")
    for curve in curves:
        loss_len = len(curve.loss) if curve.loss is not None else 0
        lr_len = len(curve.lr) if curve.lr is not None else 0
        print(
            f"- {curve.name}: loss_len={loss_len}, "
            f"lr_len={lr_len}, schedule_guess={curve.schedule_guess}"
        )

    plotted_count = save_preview_plots(
        curves=curves,
        loss_path=FIGURES_DIR / "data_preview_loss.png",
        lr_path=FIGURES_DIR / "data_preview_lr.png",
        max_curves=10,
    )
    summary = save_data_summary(
        obj=obj,
        curves=curves,
        load_method=load_method,
        data_path=DATA_FILE,
        output_path=RESULTS_DIR / "data_summary.txt",
        plotted_count=plotted_count,
    )

    print("")
    print(f"Saved summary: {RESULTS_DIR / 'data_summary.txt'}")
    if plotted_count:
        print(f"Saved loss preview: {FIGURES_DIR / 'data_preview_loss.png'}")
        print(f"Saved lr preview: {FIGURES_DIR / 'data_preview_lr.png'}")
    else:
        print("No preview plots were generated because no complete loss/lr curves were found.")
    print("")
    print(summary.splitlines()[0])

    run_tissue_baseline(curves)
    run_mpl_experiment(curves, {"results": RESULTS_DIR, "figures": FIGURES_DIR})
    run_our_method_experiment(curves, {"results": RESULTS_DIR, "figures": FIGURES_DIR})
    build_metrics_summary(RESULTS_DIR)
    plot_method_comparison_wsd(RESULTS_DIR, FIGURES_DIR)


if __name__ == "__main__":
    main()
