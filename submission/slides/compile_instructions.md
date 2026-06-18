# Slides Compile Instructions

The final slides PDF has already been generated:

```text
slides/task2_loss_prediction_slides.pdf
```

The LaTeX Beamer source is:

```text
slides/task2_loss_prediction_slides.tex
```

To regenerate the PDF, install a LaTeX distribution with XeLaTeX support, then run the following command inside the `slides/` directory:

```bash
xelatex task2_loss_prediction_slides.tex
```

The figure paths in the TeX file are relative to the `slides/` directory and point to `../figures/`.
