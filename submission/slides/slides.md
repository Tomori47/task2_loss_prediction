# 1. Title

## Predicting Loss Curve of LLM Pretraining

课程项目 Task 2

小组成员：王渭臻，1900010767，本科生

GitHub repository: https://github.com/Tomori47/task2_loss_prediction

本项目关注 LLM pretraining 中不同 learning rate schedule 下的 loss curve prediction。我们完成了两个 baseline，并提出一个简单的 Weighted Tissue fitting 方法进行比较。

---

# 2. Problem Background

LLM 预训练通常成本很高，因此我们希望在训练早期或给定训练设置时，尽可能预测后续 loss curve 的走势。

Scaling laws 研究了模型规模、数据规模、训练计算量和 loss 之间的关系。进一步地，loss curve prediction 关注的不只是最终 loss，而是整个训练过程中的 loss 变化。

在实际训练中，learning rate schedule 会显著影响 loss curve。比如 cosine decay 和 WSD schedule 的后期学习率变化不同，可能导致 loss 的下降速度和最终效果不同。

---

# 3. Task Objective

课程 Task 2 的核心目标是：

使用 cosine LRS 的 loss curve 拟合模型，并预测 WSD LRS 下的 loss curve。

我们的主实验设置为：

- fit: `M:100M_gpt_D:20B_scheduler:cosine_rope`
- test: `M:100M_gpt_D:20B_scheduler:wsd_rope`
- additional test: `M:100M_gpt_D:20B_scheduler:811_rope`

评价指标包括 MAE、MSE、RMSE 和 MAPE。

---

# 4. Data Description

数据文件为老师提供的：

`data/gpt_loss+lrs.pkl`

文件中包含三条 loss curve：

- `M:100M_gpt_D:20B_scheduler:cosine_rope`
- `M:100M_gpt_D:20B_scheduler:wsd_rope`
- `M:100M_gpt_D:20B_scheduler:811_rope`

每条曲线包含三个字段：

- `step`
- `Metrics/loss`
- `lr`

数据预览图：

- `figures/data_preview_loss.png`
- `figures/data_preview_lr.png`

---

# 5. Experimental Setting

为了符合课程要求，我们固定使用 cosine LRS 曲线拟合参数，并在 WSD LRS 上做主测试。

具体划分：

| Split | Curve |
| --- | --- |
| Fit | `cosine_rope` |
| Main test | `wsd_rope` |
| Additional test | `811_rope` |

这样可以检查模型是否能从一种 learning rate schedule 泛化到另一种 schedule。

---

# 6. Baseline 1: Tissue et al.

Tissue baseline 的形式为：

```text
L(s) = L0 + A * S1(s)^(-alpha) - C * S2(s)
```

其中：

- `S1(s)` 是累计学习率；
- `S2(s)` 是学习率 annealing 修正项；
- 本项目中 `lambda_decay = 0.99`。

Tissue baseline 的直觉是：loss 主要随着累计训练强度下降，同时 learning rate decay 会带来额外修正。

---

# 7. Baseline 2: Simplified MPL

我们实现了 simplified Multi-Power Law baseline：

```text
L(t) = L0 + A * (S1(t) + eps)^(-alpha) - B * D(t)
```

其中：

```text
S1(t) = sum lr_i
D(t) = sum max(lr_{k-1} - lr_k, 0) * (S_tail(k,t) + eps)
```

这是一个简化实现。它没有完全复现 Luo et al. 的所有参数和训练细节，但保留了两个核心思想：

- 累计学习率项；
- 学习率衰减修正项。

---

# 8. Our Method: Weighted Tissue

我们的改进方法仍然使用 Tissue 的预测公式，但改变拟合目标。

普通 Tissue 对所有 step 使用相同权重。Weighted Tissue 对后期 step 给更高权重：

```text
w_s = 1 + gamma * s / T
```

本实验中：

```text
gamma = 2.0
```

直觉是：课程评估通常更关心后期 loss 和最终 loss，因此拟合时适当强调后期训练阶段，可能提升 WSD schedule 上的预测效果。

---

# 9. Results on Cosine Fit

三种方法都在 cosine LRS 上拟合。

对应图片：

- `figures/tissue_fit_cosine.png`
- `figures/mpl_fit_cosine.png`
- `figures/our_method_fit_cosine.png`

cosine fit 上的结果：

| Method | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| Tissue | 0.07730 | 0.15378 | 2.40% |
| Simplified MPL | 0.07730 | 0.15378 | 2.40% |
| Weighted Tissue | 0.07200 | 0.15463 | 2.20% |

Weighted Tissue 的 MAE 和 MAPE 更低，但 RMSE 略高，说明加权拟合更偏向后期 loss。

---

# 10. Results on WSD Prediction

主测试是在 WSD LRS 上预测 loss curve。

重点对比图：

`figures/method_comparison_wsd.png`

图中包含：

- ground truth loss；
- Tissue prediction；
- simplified MPL prediction；
- Weighted Tissue prediction。

从整体趋势看，三种方法都能捕捉主要下降趋势。Weighted Tissue 在后期 loss 区域更贴近真实曲线，因此整体误差略低。

---

# 11. Quantitative Comparison

WSD main test 定量结果：

| Method | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| Tissue | 0.15364 | 0.20276 | 5.14% |
| Simplified MPL | 0.15364 | 0.20275 | 5.14% |
| Weighted Tissue | 0.13999 | 0.19436 | 4.64% |

Weighted Tissue 相比两个 baseline 有小幅改善：

- MAE 从约 0.15364 降到 0.13999；
- MAPE 从约 5.14% 降到 4.64%。

---

# 12. Analysis

从拟合参数看，Tissue 和 simplified MPL 的学习率衰减修正项在当前数据中几乎收敛到 0。

这说明：在只用一条 cosine 曲线拟合时，模型主要依赖累计学习率 power-law 项来解释 loss 变化。

Weighted Tissue 没有改变模型形式，而是改变拟合目标。通过强调后期 step，它牺牲了一点点全局 RMSE，但降低了 MAE 和 MAPE，并在 WSD 预测上取得小幅改善。

---

# 13. Limitations

本项目是低成本课程复现实验，仍有明显限制：

- 只使用了三条 loss curves；
- simplified MPL 没有完全复现 Luo et al. 的完整模型；
- 没有进行大规模超参数搜索；
- Weighted Tissue 只使用了固定的 `gamma = 2.0`；
- 结果主要用于课程项目展示，不能代表大规模 LLM 训练中的普遍结论。

---

# 14. Conclusion

本项目完成了 Task 2 的基本要求：

- 实现并运行 Tissue baseline；
- 实现并运行 simplified MPL baseline；
- 提出并实现 Weighted Tissue fitting；
- 使用 cosine LRS 拟合，在 WSD LRS 上测试；
- 生成了指标、预测 CSV、图像、README 和 slides 草稿。

实验结果显示，Weighted Tissue 在 WSD 测试上相比 baseline 有小幅改善。

---

# 15. Code and Contributions

GitHub repository: https://github.com/Tomori47/task2_loss_prediction

Team members and contribution: 王渭臻，1900010767，本科生，负责全部代码实现、实验复现、方法改进、结果分析与 slides 整理。

主要输出文件：

- `results/metrics_summary.csv`
- `figures/method_comparison_wsd.png`
- `slides/slides.md`

本项目当前已经具备最低可提交版本的主要材料。
