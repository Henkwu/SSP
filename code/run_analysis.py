"""Reproducible analyses for the student-stress manuscript.

All reported values are derived from datasets/StressLevelDataset.csv.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, classification_report, log_loss,
                             brier_score_loss)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate, cross_val_predict,
                                     learning_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "datasets" / "StressLevelDataset.csv"
OUT = ROOT / "analysis_outputs"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
X = df.drop(columns="stress_level")
y = df["stress_level"]

groups = {
    "Psychological": ["anxiety_level", "self_esteem", "mental_health_history", "depression"],
    "Physiological": ["headache", "blood_pressure", "sleep_quality", "breathing_problem"],
    "Environmental": ["noise_level", "living_conditions", "safety", "basic_needs"],
    "Academic": ["academic_performance", "study_load", "teacher_student_relationship", "future_career_concerns"],
    "Social": ["social_support", "peer_pressure", "extracurricular_activities", "bullying"],
}

models = {
    "Multinomial logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "k-nearest neighbours": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "RBF support vector machine": make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale")),
    "Decision tree": DecisionTreeClassifier(random_state=42),
    "Random forest": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1),
    "Gradient boosting": GradientBoostingClassifier(random_state=42),
    "Multilayer perceptron": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, early_stopping=True, random_state=42)),
}

scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "precision_macro": "precision_macro", "recall_macro": "recall_macro", "f1_macro": "f1_macro"}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
rows = []
fold_rows = []
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    row = {"model": name}
    for metric in scoring:
        vals = scores[f"test_{metric}"]
        row[f"{metric}_mean"] = vals.mean()
        row[f"{metric}_std"] = vals.std(ddof=1)
    rows.append(row)
    for i in range(len(scores["test_f1_macro"])):
        fold_rows.append({"model": name, "fold_index": i, **{m: scores[f"test_{m}"][i] for m in scoring}})
results = pd.DataFrame(rows).sort_values("f1_macro_mean", ascending=False)
results.to_csv(OUT / "model_comparison.csv", index=False)
pd.DataFrame(fold_rows).to_csv(OUT / "fold_level_scores.csv", index=False)

best_name = results.iloc[0]["model"]
best_model = models[best_name]
single_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pred = cross_val_predict(best_model, X, y, cv=single_cv, n_jobs=-1)
proba = cross_val_predict(best_model, X, y, cv=single_cv, method="predict_proba", n_jobs=-1)
cm = confusion_matrix(y, pred)
pd.DataFrame(cm, index=["true_0", "true_1", "true_2"], columns=["pred_0", "pred_1", "pred_2"]).to_csv(OUT / "confusion_matrix.csv")
report = pd.DataFrame(classification_report(y, pred, output_dict=True)).T
report.to_csv(OUT / "per_class_metrics.csv")
calibration = []
for k in range(3):
    yt = (y.to_numpy() == k).astype(int)
    calibration.append({"class": k, "brier_score": brier_score_loss(yt, proba[:, k])})
pd.DataFrame(calibration).to_csv(OUT / "calibration_metrics.csv", index=False)

rng = np.random.default_rng(42)
boot = []
for _ in range(2000):
    idx = rng.integers(0, len(y), len(y))
    boot.append({"accuracy": accuracy_score(y.iloc[idx], pred[idx]), "f1_macro": f1_score(y.iloc[idx], pred[idx], average="macro")})
boot = pd.DataFrame(boot)
pd.DataFrame({"metric": ["accuracy", "f1_macro"], "estimate": [accuracy_score(y, pred), f1_score(y, pred, average="macro")], "ci_low": [boot.accuracy.quantile(.025), boot.f1_macro.quantile(.025)], "ci_high": [boot.accuracy.quantile(.975), boot.f1_macro.quantile(.975)]}).to_csv(OUT / "bootstrap_intervals.csv", index=False)

ablation = []
for group, cols in {"All features": list(X.columns), **groups}.items():
    score = cross_validate(clone(best_model), X[cols], y, cv=cv, scoring=scoring, n_jobs=-1)
    ablation.append({"feature_set": group, "n_features": len(cols), **{f"{m}_mean": score[f"test_{m}"].mean() for m in scoring}, **{f"{m}_std": score[f"test_{m}"].std(ddof=1) for m in scoring}})
pd.DataFrame(ablation).sort_values("f1_macro_mean", ascending=False).to_csv(OUT / "domain_ablation.csv", index=False)

leave_out = []
for omitted, omitted_cols in groups.items():
    cols = [c for c in X.columns if c not in omitted_cols]
    score = cross_validate(clone(best_model), X[cols], y, cv=cv, scoring=scoring, n_jobs=-1)
    leave_out.append({"omitted_domain": omitted, "n_features": len(cols), **{f"{m}_mean": score[f"test_{m}"].mean() for m in scoring}, **{f"{m}_std": score[f"test_{m}"].std(ddof=1) for m in scoring}})
pd.DataFrame(leave_out).sort_values("f1_macro_mean", ascending=False).to_csv(OUT / "leave_one_domain_out.csv", index=False)

param_rows = []
for trees in [50, 100, 200, 500, 800]:
    for depth in [None, 5, 10, 20]:
        model = RandomForestClassifier(n_estimators=trees, max_depth=depth, random_state=42, n_jobs=-1)
        s = cross_validate(model, X, y, cv=single_cv, scoring="f1_macro", n_jobs=-1)["test_score"]
        param_rows.append({"n_estimators": trees, "max_depth": "None" if depth is None else depth, "f1_macro_mean": s.mean(), "f1_macro_std": s.std(ddof=1)})
pd.DataFrame(param_rows).to_csv(OUT / "rf_parameter_sensitivity.csv", index=False)

sns.set_theme(style="whitegrid", context="paper")
plt.figure(figsize=(5.2, 3.5))
ax = sns.countplot(data=df, x="stress_level", color="#4C78A8")
for c in ax.containers:
    ax.bar_label(c)
ax.set(xlabel="Stress-level label", ylabel="Number of records")
plt.tight_layout(); plt.savefig(FIG / "class_distribution.pdf", bbox_inches="tight"); plt.close()

plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, square=True, cbar_kws={"label": "Pearson correlation"})
plt.tight_layout(); plt.savefig(FIG / "correlation_heatmap.pdf", bbox_inches="tight"); plt.close()

plt.figure(figsize=(4.5, 3.8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Predicted label"); plt.ylabel("True label"); plt.title(f"{best_name}: out-of-fold predictions")
plt.tight_layout(); plt.savefig(FIG / "confusion_matrix.pdf", bbox_inches="tight"); plt.close()

fold_importances = []
for fold, (train_idx, test_idx) in enumerate(single_cv.split(X, y)):
    fitted = clone(best_model).fit(X.iloc[train_idx], y.iloc[train_idx])
    perm = permutation_importance(fitted, X.iloc[test_idx], y.iloc[test_idx], scoring="f1_macro", n_repeats=20, random_state=42 + fold, n_jobs=-1)
    fold_importances.append(perm.importances)
stacked = np.concatenate(fold_importances, axis=1)
imp = pd.DataFrame({"feature": X.columns, "importance_mean": stacked.mean(axis=1), "importance_std": stacked.std(axis=1, ddof=1)}).sort_values("importance_mean", ascending=False)
imp.to_csv(OUT / "permutation_importance.csv", index=False)
top = imp.head(15).sort_values("importance_mean")
plt.figure(figsize=(6.5, 5.2))
plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#59A14F", alpha=.9)
plt.xlabel("Decrease in macro-F1 after permutation")
plt.tight_layout(); plt.savefig(FIG / "permutation_importance.pdf", bbox_inches="tight"); plt.close()

train_sizes, train_scores, valid_scores = learning_curve(clone(best_model), X, y, cv=single_cv, scoring="f1_macro", train_sizes=np.linspace(.1, 1.0, 10), n_jobs=-1, shuffle=True, random_state=42)
lc = pd.DataFrame({"train_size": train_sizes, "train_mean": train_scores.mean(axis=1), "train_std": train_scores.std(axis=1, ddof=1), "validation_mean": valid_scores.mean(axis=1), "validation_std": valid_scores.std(axis=1, ddof=1)})
lc.to_csv(OUT / "learning_curve.csv", index=False)
plt.figure(figsize=(5.8, 4.0))
plt.plot(train_sizes, lc.train_mean, marker="o", label="Training macro-F1")
plt.fill_between(train_sizes, lc.train_mean-lc.train_std, lc.train_mean+lc.train_std, alpha=.18)
plt.plot(train_sizes, lc.validation_mean, marker="s", label="Validation macro-F1")
plt.fill_between(train_sizes, lc.validation_mean-lc.validation_std, lc.validation_mean+lc.validation_std, alpha=.18)
plt.xlabel("Training records per fold"); plt.ylabel("Macro-F1"); plt.ylim(.65, 1.01); plt.legend()
plt.tight_layout(); plt.savefig(FIG / "learning_curve.pdf", bbox_inches="tight"); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True)
for k, ax in enumerate(axes):
    bins = pd.qcut(proba[:, k], q=8, duplicates="drop")
    tmp = pd.DataFrame({"p": proba[:, k], "y": (y.to_numpy()==k).astype(int), "bin": bins}).groupby("bin", observed=True).agg(predicted=("p", "mean"), observed=("y", "mean"))
    ax.plot([0,1],[0,1],"--",color="gray",linewidth=1); ax.plot(tmp.predicted,tmp.observed,"o-")
    ax.set_title(f"Label {k}"); ax.set_xlabel("Mean predicted probability"); ax.set_xlim(0,1); ax.set_ylim(0,1)
axes[0].set_ylabel("Observed frequency")
plt.tight_layout(); plt.savefig(FIG / "calibration_curves.pdf", bbox_inches="tight"); plt.close()

plot_results = results.sort_values("f1_macro_mean", ascending=True)
plt.figure(figsize=(7.0, 4.6))
plt.barh(plot_results["model"], plot_results["f1_macro_mean"], xerr=plot_results["f1_macro_std"], color="#4C78A8", alpha=.9)
plt.xlabel("Repeated-CV macro-F1 (mean $\\pm$ SD)"); plt.xlim(.82, .92)
plt.tight_layout(); plt.savefig(FIG / "model_comparison.pdf", bbox_inches="tight"); plt.close()

only_df = pd.DataFrame(ablation).set_index("feature_set")
omit_df = pd.DataFrame(leave_out).set_index("omitted_domain")
domains = list(groups.keys())
xx = np.arange(len(domains)); width = .38
plt.figure(figsize=(7.2, 4.2))
plt.bar(xx-width/2, [only_df.loc[d,"f1_macro_mean"] for d in domains], width, yerr=[only_df.loc[d,"f1_macro_std"] for d in domains], label="Domain only", color="#59A14F")
plt.bar(xx+width/2, [omit_df.loc[d,"f1_macro_mean"] for d in domains], width, yerr=[omit_df.loc[d,"f1_macro_std"] for d in domains], label="Domain omitted", color="#F28E2B")
plt.axhline(only_df.loc["All features","f1_macro_mean"], linestyle="--", color="black", linewidth=1, label="All features")
plt.xticks(xx, domains, rotation=20, ha="right"); plt.ylabel("Repeated-CV macro-F1"); plt.ylim(.82,.92); plt.legend(ncol=3, fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "domain_robustness.pdf", bbox_inches="tight"); plt.close()

rf_grid = pd.DataFrame(param_rows).copy()
rf_grid["max_depth"] = pd.Categorical(rf_grid["max_depth"].astype(str), categories=["5","10","20","None"], ordered=True)
pivot = rf_grid.pivot(index="max_depth", columns="n_estimators", values="f1_macro_mean")
plt.figure(figsize=(6.5, 3.8))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label":"Mean macro-F1"})
plt.xlabel("Number of trees"); plt.ylabel("Maximum depth")
plt.tight_layout(); plt.savefig(FIG / "rf_sensitivity_heatmap.pdf", bbox_inches="tight"); plt.close()

print(results.to_string(index=False))
print("\nBest model:", best_name)
