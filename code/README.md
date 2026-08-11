# Code for Multidimensional Student Stress Prediction

This repository provides the code and generated results for the study:

> **Multidimensional Student Stress Prediction with Interpretable Machine Learning: A Reproducible Comparative Study**

## Abstract

Student stress reflects interacting psychological, physiological, environmental, academic, and social conditions, making prediction from structured survey data a multidimensional classification problem. This reproducible comparative study uses a public dataset containing 1,100 records, 20 predictors, and three approximately balanced stress-level labels. Eight supervised classifiers were evaluated under identical fold-wise preprocessing and a five-times repeated five-fold stratified cross-validation protocol. Multinomial logistic regression attained the highest mean macro-F1 (0.8866 ± 0.0169) and tied gradient boosting for the highest accuracy (0.8864), indicating that greater model complexity did not improve performance on this dataset. Complementary domain analyses produced standalone macro-F1 values between 0.8634 and 0.8884 and showed that omitting physiological variables caused the largest decline, revealing substantial redundancy but unequal predictive reliance across domains. Cross-validated permutation analysis identified blood pressure and social support as the most influential predictive variables. Together, matched model comparison, domain ablation, uncertainty analysis, calibration assessment, and held-out permutation importance provide an interpretable account of model behaviour. These associations are predictive rather than causal and may partly reflect how the public dataset and its labels were constructed; independent-cohort validation remains necessary.

## Code overview

The complete analysis is implemented in:

```text
run_analysis.py
```

Running this script reproduces:

- comparison of eight classification models;
- repeated stratified cross-validation results;
- fold-level performance scores;
- out-of-fold confusion matrix and per-class metrics;
- bootstrap confidence intervals;
- probability calibration assessment;
- domain-only and leave-one-domain-out analyses;
- cross-validated permutation importance;
- learning curves; and
- random-forest parameter sensitivity analysis.

All randomized procedures use fixed seeds. The eight models are evaluated using the same 25 folds from five-times repeated five-fold stratified cross-validation. Standardization is performed inside the relevant scikit-learn pipelines and is fitted separately within each training fold.

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn

Create an isolated environment and install the dependencies:

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn
```

## Dataset

Download `StressLevelDataset.csv` from:

<https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis>

Place the dataset in the following structure:

```text
parent_directory/
├── datasets/
│   └── StressLevelDataset.csv
└── Submission_Ready_Manuscript_ESWA_v3/
    ├── run_analysis.py
    ├── analysis_outputs/
    └── figures/
```

The dataset used for the reported results has the following properties:

| Property | Expected value |
|---|---:|
| Records | 1,100 |
| Predictors | 20 |
| Outcome | `stress_level` |
| Label 0 records | 373 |
| Label 1 records | 358 |
| Label 2 records | 369 |
| SHA-256 | `14A45E92708B0C063AD4AB04563AA8FD4E3FC27157FD282E1C0658DC5161FAED` |

Verify the dataset checksum before running the analysis:

```bash
# Linux or macOS
sha256sum ../datasets/StressLevelDataset.csv
```

```powershell
# Windows PowerShell
Get-FileHash ..\datasets\StressLevelDataset.csv -Algorithm SHA256
```

The raw CSV is read directly and is not modified by the script.

## Run the analysis

From the directory containing `run_analysis.py`, run:

```bash
python run_analysis.py
```

On Windows, the following command can also be used:

```powershell
py run_analysis.py
```

No GPU is required. Operations supporting `n_jobs=-1` use all available CPU cores.

## Expected results

The leading results in `analysis_outputs/model_comparison.csv` should be approximately:

| Model | Accuracy, mean ± SD | Macro-F1, mean ± SD |
|---|---:|---:|
| Multinomial logistic regression | 0.8864 ± 0.0169 | 0.8866 ± 0.0169 |
| Gradient boosting | 0.8864 ± 0.0193 | 0.8863 ± 0.0193 |
| Decision tree | 0.8844 ± 0.0208 | 0.8844 ± 0.0208 |

The expected best model is:

```text
Multinomial logistic regression
```

Exact results require the same dataset and software environment. Small floating-point differences may occur across operating systems or library versions. A different record count, checksum, or model ranking should be investigated before interpreting the results.

## Generated outputs

The script writes numerical results to `analysis_outputs/`:

```text
bootstrap_intervals.csv
calibration_metrics.csv
confusion_matrix.csv
domain_ablation.csv
fold_level_scores.csv
learning_curve.csv
leave_one_domain_out.csv
model_comparison.csv
per_class_metrics.csv
permutation_importance.csv
rf_parameter_sensitivity.csv
```

It writes vector figures to `figures/`:

```text
calibration_curves.pdf
class_distribution.pdf
confusion_matrix.pdf
correlation_heatmap.pdf
domain_robustness.pdf
learning_curve.pdf
model_comparison.pdf
permutation_importance.pdf
rf_sensitivity_heatmap.pdf
```

## Reproducibility notes

- Primary evaluation: repeated stratified 5-fold cross-validation with 5 repeats.
- Fixed primary random seed: `42`.
- Bootstrap resamples: `2,000`.
- Permutation-importance repeats per fold: `20`.
- Scaling is learned only from each training fold.
- No feature selection is performed before cross-validation.
- All models receive identical folds and evaluation metrics.
- Parameters not explicitly specified in the script use the installed scikit-learn defaults.

The supplied analysis is an internally validated benchmark on one public cross-sectional dataset. It should not be interpreted as a clinical diagnostic system or evidence of causal relationships.
