# Predicting Loss Curve of LLM Pretraining

GitHub repository: https://github.com/Tomori47/task2_loss_prediction

Team members and contribution: 王渭臻，1900010767，本科生，负责全部代码实现、实验复现、方法改进、结果分析与 slides 整理。

## Project Introduction

This repository is for the course final project Task 2: Predicting Loss Curve of LLM Pretraining.

The goal is to study loss curve prediction for LLM pretraining under different learning rate schedules. Following the task requirement, we fit models on the cosine learning rate schedule and evaluate prediction quality on the WSD learning rate schedule. We also report an additional test on the 811 schedule.

## Data

We use the teacher-provided `data/gpt_loss+lrs.pkl`. The file contains three loss curves, each stored as a pandas DataFrame with columns `step`, `Metrics/loss`, and `lr`.

Curves:

- `M:100M_gpt_D:20B_scheduler:cosine_rope`
- `M:100M_gpt_D:20B_scheduler:wsd_rope`
- `M:100M_gpt_D:20B_scheduler:811_rope`

The data summary is saved to `results/data_summary.txt`. Preview plots are saved to:

- `figures/data_preview_loss.png`
- `figures/data_preview_lr.png`

## Experimental Setting

Main setting:

- Fit curve: `M:100M_gpt_D:20B_scheduler:cosine_rope`
- Main test curve: `M:100M_gpt_D:20B_scheduler:wsd_rope`
- Additional test curve: `M:100M_gpt_D:20B_scheduler:811_rope`

Metrics:

- MAE
- MSE
- RMSE
- MAPE

## Methods

### Tissue Baseline

The Tissue baseline uses:

```text
L(s) = L0 + A * S1(s)^(-alpha) - C * S2(s)
```

where `S1(s)` is the cumulative learning rate and `S2(s)` is the learning-rate annealing feature. In this implementation, `lambda_decay = 0.99`.

Results:

- `results/tissue_metrics.csv`
- `results/tissue_predictions_cosine.csv`
- `results/tissue_predictions_wsd.csv`
- `results/tissue_predictions_811.csv`

Figures:

- `figures/tissue_fit_cosine.png`
- `figures/tissue_predict_wsd.png`
- `figures/tissue_predict_811.png`

### Simplified Multi-Power Law Baseline

This project implements a simplified Multi-Power Law baseline. The full Luo et al. method has more parameters and training details, and official code may require extra adaptation. For a low-cost course reproduction, the simplified version keeps the two main ingredients: a cumulative learning-rate power-law term and a learning-rate decay correction term.

Formula:

```text
L(t) = L0 + A * (S1(t) + eps)^(-alpha) - B * D(t)
S1(t) = sum_{i=1}^t lr_i
D(t) = sum_{k=1}^t max(lr_{k-1} - lr_k, 0) * (S_tail(k,t) + eps)
```

Results:

- `results/mpl_metrics.csv`
- `results/mpl_predictions_cosine.csv`
- `results/mpl_predictions_wsd.csv`
- `results/mpl_predictions_811.csv`

Figures:

- `figures/mpl_fit_cosine.png`
- `figures/mpl_predict_wsd.png`
- `figures/mpl_predict_811.png`

### Our Method: Weighted Tissue Fitting

Weighted Tissue uses the same prediction formula as the Tissue baseline:

```text
L(s) = L0 + A * S1(s)^(-alpha) - C * S2(s)
```

The difference is the fitting objective. Ordinary Tissue fitting uses unweighted least squares over the whole cosine loss curve. Weighted Tissue gives later steps higher weights so that the fitted model pays more attention to late-stage and final loss behavior.

Weight formula:

```text
w_s = 1 + gamma * s / T
```

In the main experiment, `gamma = 2.0`. This makes the final steps receive about three times the weight of the first step. Compared with Tissue, the model class is unchanged; only the fitting objective is weighted.

Results:

- `results/our_method_metrics.csv`
- `results/our_method_predictions_cosine.csv`
- `results/our_method_predictions_wsd.csv`
- `results/our_method_predictions_811.csv`

Figures:

- `figures/our_method_fit_cosine.png`
- `figures/our_method_predict_wsd.png`
- `figures/our_method_predict_811.png`

## Result Summary

The merged result table is saved to:

- `results/metrics_summary.csv`

Key WSD main-test results:

| Method | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| Tissue | 0.15364 | 0.20276 | 5.14% |
| Simplified MPL | 0.15364 | 0.20275 | 5.14% |
| Weighted Tissue | 0.13999 | 0.19436 | 4.64% |

Weighted Tissue gives a small improvement over both baselines on the WSD main test. In this dataset, the decay correction terms in Tissue and simplified MPL are fitted very close to zero, so all methods mainly depend on the cumulative learning-rate power-law term. Weighted Tissue improves the WSD prediction slightly by emphasizing late-stage loss during fitting.

Main comparison figure:

- `figures/method_comparison_wsd.png`

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all experiments:

```bash
python run_all.py
```

The script will:

1. Load and summarize the data.
2. Run Tissue baseline.
3. Run simplified MPL baseline.
4. Run Weighted Tissue fitting.
5. Save metrics and prediction CSV files.
6. Generate figures.

## Key Output Files

Results:

- `results/data_summary.txt`
- `results/tissue_metrics.csv`
- `results/mpl_metrics.csv`
- `results/our_method_metrics.csv`
- `results/metrics_summary.csv`
- `results/*predictions*.csv`

Figures:

- `figures/data_preview_loss.png`
- `figures/data_preview_lr.png`
- `figures/tissue_predict_wsd.png`
- `figures/mpl_predict_wsd.png`
- `figures/our_method_predict_wsd.png`
- `figures/method_comparison_wsd.png`

Slides:

- `slides/slides.md`
- `slides/task2_loss_prediction_slides.tex`
- `slides/task2_loss_prediction_slides.pdf`
- `slides/compile_instructions.md`

The final slides file for submission is:

- `slides/task2_loss_prediction_slides.pdf`

The LaTeX source file is also included:

- `slides/task2_loss_prediction_slides.tex`

## Notes

This project is a low-cost course reproduction. The simplified MPL implementation does not fully reproduce all details from Luo et al. The main goal is to provide a complete, runnable project with baselines, one simple method, quantitative results, figures, README, and slides draft.
