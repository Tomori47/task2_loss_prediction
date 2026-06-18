"""Data loading and summary utilities for Task 2.

This module only inspects and summarizes the provided loss curve data. It does
not implement any fitting, baseline, or prediction algorithm.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LOSS_KEYWORDS = ("metrics/loss", "loss", "train/loss", "training_loss")
LR_KEYWORDS = ("lr", "learning_rate", "learning rate")
NAME_KEYWORDS = (
    "name",
    "curve",
    "run",
    "schedule",
    "scheduler",
    "sched",
    "lrs",
    "config",
    "setting",
    "tag",
)
STEP_KEYWORDS = ("step", "iter", "iteration", "token", "tokens")


@dataclass
class CurveData:
    """A single loss/lr curve extracted from the raw pkl object."""

    name: str
    loss: np.ndarray | None
    lr: np.ndarray | None
    source: str
    step: np.ndarray | None = None
    schedule_guess: str = "unknown"


def load_pkl(path: str | Path) -> tuple[Any, str]:
    """Load a pkl file using pickle, pandas.read_pickle, then torch.load.

    Torch is optional and imported only as a last-resort fallback.
    """

    path = Path(path)
    errors: list[str] = []

    try:
        with path.open("rb") as f:
            return pickle.load(f), "pickle"
    except Exception as exc:  # pragma: no cover - diagnostic path
        errors.append(f"pickle failed: {type(exc).__name__}: {exc}")

    try:
        import pandas as pd

        return pd.read_pickle(path), "pandas.read_pickle"
    except Exception as exc:  # pragma: no cover - diagnostic path
        errors.append(f"pandas.read_pickle failed: {type(exc).__name__}: {exc}")

    try:
        import torch

        return torch.load(path, map_location="cpu"), "torch.load"
    except ImportError:
        errors.append("torch.load skipped: torch is not installed")
    except Exception as exc:  # pragma: no cover - diagnostic path
        errors.append(f"torch.load failed: {type(exc).__name__}: {exc}")

    raise RuntimeError("Could not load pkl file.\n" + "\n".join(errors))


def inspect_structure(obj: Any, max_depth: int = 3, max_items: int = 8) -> str:
    """Return a readable recursive summary of the raw data object."""

    lines: list[str] = []
    _inspect(obj, lines, name="root", depth=0, max_depth=max_depth, max_items=max_items)
    return "\n".join(lines)


def print_structure(obj: Any, max_depth: int = 3, max_items: int = 8) -> None:
    """Print raw data type, keys/fields, shapes, lengths, and previews."""

    print(inspect_structure(obj, max_depth=max_depth, max_items=max_items))


def extract_curves(obj: Any) -> list[CurveData]:
    """Extract loss/lr curves from common pkl structures."""

    curves: list[CurveData] = []
    _extract(obj, curves, path="root")

    seen: set[str] = set()
    for curve in curves:
        base = _clean_name(curve.name)
        name = base
        suffix = 2
        while name in seen:
            name = f"{base}_{suffix}"
            suffix += 1
        seen.add(name)
        curve.name = name
        curve.schedule_guess = guess_schedule_type(curve.name, curve.lr)

    return curves


def build_data_summary(
    obj: Any,
    curves: list[CurveData],
    load_method: str,
    data_path: str | Path,
    plotted_count: int | None = None,
) -> str:
    """Build a text summary for results/data_summary.txt."""

    data_path = Path(data_path)
    data_path_display = (
        f"{data_path.parent.name}/{data_path.name}"
        if data_path.parent.name
        else data_path.name
    )
    lines: list[str] = []
    lines.append("Task 2 Loss Curve Data Summary")
    lines.append("=" * 34)
    lines.append(f"Data file: {data_path_display}")
    lines.append(f"Load method: {load_method}")
    lines.append("")
    lines.append("Top-level structure")
    lines.append("-" * 19)
    lines.append(inspect_structure(obj, max_depth=3, max_items=8))
    lines.append("")
    lines.append("Detected fields")
    lines.append("-" * 15)
    lines.extend(_field_detection_lines(obj))
    lines.append("")
    lines.append("Extracted curves")
    lines.append("-" * 16)

    if not curves:
        lines.append("No complete loss/lr curves were extracted.")
    else:
        for i, curve in enumerate(curves, start=1):
            step_stats = summarize_array(curve.step)
            loss_stats = summarize_array(curve.loss)
            lr_stats = summarize_array(curve.lr)
            anomalies = detect_anomalies(curve.loss, curve.lr)
            lines.append(f"[{i}] {curve.name}")
            lines.append(f"  source: {curve.source}")
            lines.append(f"  schedule_guess: {curve.schedule_guess}")
            lines.append(f"  step_length: {step_stats['length']}")
            lines.append(f"  step_start: {step_stats['start']}")
            lines.append(f"  step_end: {step_stats['end']}")
            lines.append(f"  loss_length: {loss_stats['length']}")
            lines.append(f"  lr_length: {lr_stats['length']}")
            lines.append(f"  lengths_match: {loss_stats['length'] == lr_stats['length']}")
            lines.append(
                "  loss: "
                f"min={loss_stats['min']}, max={loss_stats['max']}, "
                f"start={loss_stats['start']}, end={loss_stats['end']}"
            )
            lines.append(
                "  lr: "
                f"min={lr_stats['min']}, max={lr_stats['max']}, "
                f"start={lr_stats['start']}, end={lr_stats['end']}"
            )
            lines.append(f"  anomalies: {', '.join(anomalies) if anomalies else 'none'}")

    lines.append("")
    lines.append("Schedule grouping")
    lines.append("-" * 17)
    schedule_groups: dict[str, list[str]] = {}
    for curve in curves:
        schedule_groups.setdefault(curve.schedule_guess, []).append(curve.name)
    if schedule_groups:
        for schedule, names in sorted(schedule_groups.items()):
            lines.append(f"{schedule}: {', '.join(names)}")
    else:
        lines.append("No schedules could be grouped.")

    if plotted_count is not None and curves and plotted_count < len(curves):
        lines.append("")
        lines.append(
            f"Preview plots include only the first {plotted_count} curves out of {len(curves)}."
        )

    return "\n".join(lines) + "\n"


def save_data_summary(
    obj: Any,
    curves: list[CurveData],
    load_method: str,
    data_path: str | Path,
    output_path: str | Path,
    plotted_count: int | None = None,
) -> str:
    """Write the data summary file and return its text."""

    summary = build_data_summary(
        obj=obj,
        curves=curves,
        load_method=load_method,
        data_path=data_path,
        plotted_count=plotted_count,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
    return summary


def save_preview_plots(
    curves: list[CurveData],
    loss_path: str | Path,
    lr_path: str | Path,
    max_curves: int = 10,
) -> int:
    """Save preview plots for loss and learning-rate curves.

    Returns the number of curves plotted.
    """

    curves_to_plot = [
        curve for curve in curves if curve.loss is not None and curve.lr is not None
    ][:max_curves]
    if not curves_to_plot:
        return 0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss_path = Path(loss_path)
    lr_path = Path(lr_path)
    loss_path.parent.mkdir(parents=True, exist_ok=True)
    lr_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for curve in curves_to_plot:
        x = _x_values(curve.step, curve.loss)
        plt.plot(x, curve.loss, label=curve.name, linewidth=1.4)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve Preview")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for curve in curves_to_plot:
        x = _x_values(curve.step, curve.lr)
        plt.plot(x, curve.lr, label=curve.name, linewidth=1.4)
    plt.xlabel("Step")
    plt.ylabel("Learning rate")
    plt.title("Learning Rate Schedule Preview")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(lr_path, dpi=180)
    plt.close()

    return len(curves_to_plot)


def summarize_array(arr: np.ndarray | None) -> dict[str, Any]:
    """Summarize numeric sequence length and basic values."""

    if arr is None:
        return {"length": 0, "min": "NA", "max": "NA", "start": "NA", "end": "NA"}

    values = np.asarray(arr, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        min_value = max_value = "NA"
    else:
        min_value = _format_float(float(np.min(finite)))
        max_value = _format_float(float(np.max(finite)))

    return {
        "length": int(values.size),
        "min": min_value,
        "max": max_value,
        "start": _format_float_or_na(values[0]) if values.size else "NA",
        "end": _format_float_or_na(values[-1]) if values.size else "NA",
    }


def detect_anomalies(loss: np.ndarray | None, lr: np.ndarray | None) -> list[str]:
    """Flag NaN, inf, length mismatch, and simple invalid numeric values."""

    anomalies: list[str] = []

    if loss is not None:
        loss_values = np.asarray(loss, dtype=float).reshape(-1)
        if np.isnan(loss_values).any():
            anomalies.append(f"loss has {int(np.isnan(loss_values).sum())} NaN")
        if np.isinf(loss_values).any():
            anomalies.append(f"loss has {int(np.isinf(loss_values).sum())} inf")
        if np.isfinite(loss_values).any() and np.nanmin(loss_values) <= 0:
            anomalies.append("loss has non-positive values")

    if lr is not None:
        lr_values = np.asarray(lr, dtype=float).reshape(-1)
        if np.isnan(lr_values).any():
            anomalies.append(f"lr has {int(np.isnan(lr_values).sum())} NaN")
        if np.isinf(lr_values).any():
            anomalies.append(f"lr has {int(np.isinf(lr_values).sum())} inf")
        if np.isfinite(lr_values).any() and np.nanmin(lr_values) < 0:
            anomalies.append("lr has negative values")

    if loss is not None and lr is not None:
        if np.asarray(loss).size != np.asarray(lr).size:
            anomalies.append("loss/lr length mismatch")

    return anomalies


def guess_schedule_type(name: str, lr: np.ndarray | None) -> str:
    """Guess whether a curve is cosine, WSD/multi-step, constant, or unknown."""

    lower_name = name.lower()
    if "cosine" in lower_name:
        return "cosine"
    if any(token in lower_name for token in ("wsd", "8-1-1", "811", "multi-step", "multistep")):
        return "wsd_or_multi_step"
    if "constant" in lower_name:
        return "constant"

    if lr is None:
        return "unknown"

    values = np.asarray(lr, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return "unknown"

    value_range = float(np.max(values) - np.min(values))
    max_abs = max(float(np.max(np.abs(values))), 1e-12)
    if value_range / max_abs < 1e-8:
        return "constant"

    diffs = np.diff(values)
    tol = max(value_range * 1e-6, 1e-12)
    change_fraction = float(np.mean(np.abs(diffs) > tol)) if diffs.size else 0.0
    unique_count = int(np.unique(np.round(values, 12)).size)
    unique_ratio = unique_count / max(values.size, 1)

    if unique_count <= max(12, int(0.08 * values.size)) and change_fraction < 0.2:
        return "wsd_or_multi_step"
    if unique_count >= 30 and unique_ratio > 0.1 and change_fraction > 0.2:
        return "cosine_like"
    return "unknown"


def _extract(obj: Any, curves: list[CurveData], path: str) -> None:
    pd = _try_import_pandas()

    if pd is not None and isinstance(obj, pd.DataFrame):
        _extract_from_dataframe(obj, curves, path)
        return

    if pd is not None and isinstance(obj, pd.Series):
        _extract(obj.to_dict(), curves, path)
        return

    if isinstance(obj, dict):
        loss_key = _find_key(obj.keys(), LOSS_KEYWORDS)
        lr_key = _find_key(obj.keys(), LR_KEYWORDS)
        step_key = _find_key(obj.keys(), STEP_KEYWORDS)
        if loss_key is not None or lr_key is not None:
            name = _name_from_dict(obj, fallback=path)
            curves.append(
                CurveData(
                    name=name,
                    loss=_to_numeric_array(obj.get(loss_key)) if loss_key is not None else None,
                    lr=_to_numeric_array(obj.get(lr_key)) if lr_key is not None else None,
                    source=path,
                    step=_to_numeric_array(obj.get(step_key)) if step_key is not None else None,
                )
            )
            return

        for key, value in obj.items():
            _extract(value, curves, f"{path}.{key}")
        return

    if isinstance(obj, (list, tuple)):
        if _looks_like_loss_lr_pair(obj):
            curves.append(
                CurveData(
                    name=path,
                    loss=_to_numeric_array(obj[0]),
                    lr=_to_numeric_array(obj[1]),
                    source=path,
                )
            )
            return

        for i, item in enumerate(obj):
            _extract(item, curves, f"{path}[{i}]")


def _x_values(step: np.ndarray | None, values: np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.asarray([])
    if step is not None and np.asarray(step).size == np.asarray(values).size:
        return np.asarray(step, dtype=float).reshape(-1)
    return np.arange(np.asarray(values).size)


def _extract_from_dataframe(df: Any, curves: list[CurveData], path: str) -> None:
    loss_col = _find_key(df.columns, LOSS_KEYWORDS)
    lr_col = _find_key(df.columns, LR_KEYWORDS)
    step_col = _find_key(df.columns, STEP_KEYWORDS)
    if loss_col is None and lr_col is None:
        return

    loss_is_sequence = loss_col is not None and _column_contains_sequences(df[loss_col])
    lr_is_sequence = lr_col is not None and _column_contains_sequences(df[lr_col])
    name_cols = _candidate_name_columns(df)

    if loss_is_sequence or lr_is_sequence:
        for idx, row in df.iterrows():
            name = _row_name(row, idx, path, name_cols)
            curves.append(
                CurveData(
                    name=name,
                    loss=_to_numeric_array(row[loss_col]) if loss_col is not None else None,
                    lr=_to_numeric_array(row[lr_col]) if lr_col is not None else None,
                    source=f"{path}.row[{idx}]",
                    step=_to_numeric_array(row[step_col]) if step_col is not None else None,
                )
            )
        return

    group_cols = _group_columns(df, exclude={loss_col, lr_col})
    if group_cols:
        for group_key, group_df in df.groupby(group_cols, sort=False, dropna=False):
            name = _group_name(group_cols, group_key)
            curves.append(
                CurveData(
                    name=name,
                    loss=_to_numeric_array(group_df[loss_col].to_numpy())
                    if loss_col is not None
                    else None,
                    lr=_to_numeric_array(group_df[lr_col].to_numpy()) if lr_col is not None else None,
                    source=f"{path}.group[{name}]",
                    step=_to_numeric_array(group_df[step_col].to_numpy())
                    if step_col is not None
                    else None,
                )
            )
        return

    curves.append(
        CurveData(
            name=path,
            loss=_to_numeric_array(df[loss_col].to_numpy()) if loss_col is not None else None,
        lr=_to_numeric_array(df[lr_col].to_numpy()) if lr_col is not None else None,
        source=path,
        step=_to_numeric_array(df[step_col].to_numpy()) if step_col is not None else None,
    )
    )


def _candidate_name_columns(df: Any) -> list[Any]:
    columns: list[Any] = []
    for col in df.columns:
        lower = str(col).lower()
        if any(keyword in lower for keyword in NAME_KEYWORDS):
            columns.append(col)
    return columns


def _group_columns(df: Any, exclude: set[Any]) -> list[Any]:
    candidates: list[Any] = []
    n_rows = len(df)
    for col in df.columns:
        if col in exclude:
            continue
        lower = str(col).lower()
        if any(keyword in lower for keyword in STEP_KEYWORDS):
            continue
        if any(keyword in lower for keyword in NAME_KEYWORDS):
            unique_count = int(df[col].nunique(dropna=False))
            if 1 < unique_count < n_rows:
                candidates.append(col)
    return candidates


def _group_name(group_cols: list[Any], group_key: Any) -> str:
    if not isinstance(group_key, tuple):
        group_key = (group_key,)
    parts = [f"{col}={value}" for col, value in zip(group_cols, group_key)]
    return "__".join(parts)


def _row_name(row: Any, idx: Any, path: str, name_cols: list[Any]) -> str:
    values: list[str] = []
    for col in name_cols:
        value = row[col]
        if _is_scalar_name(value):
            values.append(f"{col}={value}")
    if values:
        return "__".join(values)
    return f"{path}.row_{idx}"


def _name_from_dict(obj: dict[Any, Any], fallback: str) -> str:
    for key, value in obj.items():
        lower = str(key).lower()
        if any(keyword in lower for keyword in NAME_KEYWORDS) and _is_scalar_name(value):
            return f"{key}={value}"
    return fallback


def _find_key(keys: Iterable[Any], candidates: Iterable[str]) -> Any | None:
    key_list = list(keys)
    exact_map = {str(key).lower(): key for key in key_list}
    for candidate in candidates:
        if candidate in exact_map:
            return exact_map[candidate]

    for key in key_list:
        lower = str(key).lower()
        if any(candidate == lower for candidate in candidates):
            return key
    for key in key_list:
        lower = str(key).lower()
        if any(candidate in lower for candidate in candidates):
            return key
    return None


def _column_contains_sequences(series: Any) -> bool:
    sample = series.dropna().head(20)
    return any(_is_sequence_like(value) for value in sample)


def _looks_like_loss_lr_pair(obj: list[Any] | tuple[Any, ...]) -> bool:
    if len(obj) != 2:
        return False
    return _is_sequence_like(obj[0]) and _is_sequence_like(obj[1])


def _to_numeric_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()

    if isinstance(value, dict):
        key = _find_key(value.keys(), LOSS_KEYWORDS + LR_KEYWORDS)
        if key is not None:
            value = value[key]

    if _is_scalar_number(value):
        return np.asarray([float(value)], dtype=float)

    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None

    return arr


def _is_scalar_number(value: Any) -> bool:
    try:
        return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
            value, bool
        )
    except TypeError:
        return False


def _is_scalar_name(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, np.integer, np.floating))


def _is_sequence_like(value: Any) -> bool:
    if isinstance(value, (str, bytes, dict)):
        return False
    if _is_scalar_number(value):
        return False
    return hasattr(value, "__len__") or hasattr(value, "shape")


def _inspect(
    obj: Any,
    lines: list[str],
    name: str,
    depth: int,
    max_depth: int,
    max_items: int,
) -> None:
    indent = "  " * depth
    lines.append(f"{indent}{name}: {_object_info(obj)}")

    if depth >= max_depth:
        return

    pd = _try_import_pandas()
    if pd is not None and isinstance(obj, pd.DataFrame):
        lines.append(f"{indent}  columns: {list(obj.columns)}")
        for col in list(obj.columns)[:max_items]:
            series = obj[col]
            lines.append(
                f"{indent}  field {col}: dtype={series.dtype}, "
                f"len={len(series)}, preview={_preview(series.head(3).tolist())}"
            )
        if len(obj.columns) > max_items:
            lines.append(f"{indent}  ... {len(obj.columns) - max_items} more columns")
        return

    if pd is not None and isinstance(obj, pd.Series):
        lines.append(f"{indent}  index_preview: {list(obj.index[:max_items])}")
        for idx, value in obj.head(max_items).items():
            _inspect(value, lines, f"[{idx}]", depth + 1, max_depth, max_items)
        return

    if isinstance(obj, dict):
        keys = list(obj.keys())
        lines.append(f"{indent}  keys: {keys[:max_items]}")
        for key in keys[:max_items]:
            _inspect(obj[key], lines, str(key), depth + 1, max_depth, max_items)
        if len(keys) > max_items:
            lines.append(f"{indent}  ... {len(keys) - max_items} more keys")
        return

    if isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj[:max_items]):
            _inspect(value, lines, f"[{i}]", depth + 1, max_depth, max_items)
        if len(obj) > max_items:
            lines.append(f"{indent}  ... {len(obj) - max_items} more items")


def _object_info(obj: Any) -> str:
    parts = [f"type={type(obj).__name__}"]
    if hasattr(obj, "shape"):
        parts.append(f"shape={getattr(obj, 'shape')}")
    try:
        parts.append(f"len={len(obj)}")
    except TypeError:
        pass
    preview = _preview(obj)
    if preview:
        parts.append(f"preview={preview}")
    return ", ".join(parts)


def _preview(obj: Any, max_items: int = 5) -> str:
    try:
        if isinstance(obj, np.ndarray):
            return repr(obj.reshape(-1)[:max_items].tolist())
        if isinstance(obj, (list, tuple)):
            return repr([_short_value(value) for value in obj[:max_items]])
        if isinstance(obj, dict):
            return repr(list(obj.keys())[:max_items])
        if _is_scalar_name(obj):
            return repr(obj)
    except Exception:
        return ""
    return ""


def _short_value(value: Any) -> Any:
    if _is_scalar_name(value):
        return value
    if hasattr(value, "shape"):
        return f"{type(value).__name__}(shape={value.shape})"
    try:
        return f"{type(value).__name__}(len={len(value)})"
    except TypeError:
        return type(value).__name__


def _field_detection_lines(obj: Any) -> list[str]:
    lines: list[str] = []
    fields = _collect_field_names(obj)
    lower_fields = [field.lower() for field in fields]

    has_loss = any("metrics/loss" == field or "loss" in field for field in lower_fields)
    has_lr = any(field == "lr" or "learning_rate" in field or "learning rate" in field for field in lower_fields)
    schedule_like = [
        field
        for field in fields
        if any(keyword in field.lower() for keyword in ("cosine", "wsd", "constant", "multi", "step", "schedule", "lrs"))
    ]

    lines.append(f"contains loss or metrics/loss: {has_loss}")
    lines.append(f"contains lr or learning_rate: {has_lr}")
    lines.append(
        "schedule/name-like fields or keys: "
        + (", ".join(schedule_like[:30]) if schedule_like else "none detected")
    )
    return lines


def _collect_field_names(obj: Any, max_depth: int = 4, depth: int = 0) -> list[str]:
    if depth > max_depth:
        return []

    pd = _try_import_pandas()
    if pd is not None and isinstance(obj, pd.DataFrame):
        return [str(col) for col in obj.columns]
    if pd is not None and isinstance(obj, pd.Series):
        return [str(idx) for idx in obj.index]
    if isinstance(obj, dict):
        fields = [str(key) for key in obj.keys()]
        for value in list(obj.values())[:20]:
            fields.extend(_collect_field_names(value, max_depth=max_depth, depth=depth + 1))
        return fields
    if isinstance(obj, (list, tuple)):
        fields: list[str] = []
        for value in obj[:20]:
            fields.extend(_collect_field_names(value, max_depth=max_depth, depth=depth + 1))
        return fields
    return []


def _clean_name(name: str) -> str:
    name = str(name).replace("root.", "")
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace(" ", "_")
    return name.strip("_") or "curve"


def _format_float_or_na(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(value):
        return str(value)
    return _format_float(value)


def _format_float(value: float) -> str:
    return f"{value:.8g}"


def _try_import_pandas() -> Any | None:
    try:
        import pandas as pd

        return pd
    except ImportError:
        return None
