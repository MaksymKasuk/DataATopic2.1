"""Практична робота 4. Логістична регресія.

Набір даних: Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning
Repository, DOI 10.24432/C5DW2B. Позитивний клас дорівнює 1 та означає
злоякісне новоутворення.

"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
OUTPUT_DIR = Path("results_logistic_regression")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 180,
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    }
)


def load_uci_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Завантажує локальну копію WDBC, що постачається зі scikit-learn."""

    dataset = load_breast_cancer(as_frame=True)
    x = dataset.data.copy()
    # У scikit-learn: 0 = malignant, 1 = benign. Для аналізу події
    # перетворюємо кодування так, щоб 1 означала malignant.
    y = (1 - dataset.target).astype(int).rename("malignant")
    return x, y


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp))


def metric_row(
    model_name: str,
    sample_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | str | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "Модель": model_name,
        "Вибірка": sample_name,
        "Поріг": threshold,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Sensitivity": recall_score(y_true, predictions, zero_division=0),
        "Specificity": specificity(y_true, predictions),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "PR_AUC": average_precision_score(y_true, probabilities),
        "Log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "Brier": brier_score_loss(y_true, probabilities),
    }


def bootstrap_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Стратифікований bootstrap для 95-відсоткових інтервалів."""

    rng = np.random.default_rng(SEED + 1)
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    negative = np.flatnonzero(y_true == 0)
    positive = np.flatnonzero(y_true == 1)
    storage = {"Accuracy": [], "Sensitivity": [], "Specificity": [], "ROC_AUC": []}
    for _ in range(n_boot):
        indices = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        rng.shuffle(indices)
        y_boot = y_true[indices]
        p_boot = probabilities[indices]
        pred_boot = (p_boot >= threshold).astype(int)
        storage["Accuracy"].append(accuracy_score(y_boot, pred_boot))
        storage["Sensitivity"].append(recall_score(y_boot, pred_boot))
        storage["Specificity"].append(specificity(y_boot, pred_boot))
        storage["ROC_AUC"].append(roc_auc_score(y_boot, p_boot))
    rows = []
    for name, values in storage.items():
        point = (
            roc_auc_score(y_true, probabilities)
            if name == "ROC_AUC"
            else {
                "Accuracy": accuracy_score(y_true, probabilities >= threshold),
                "Sensitivity": recall_score(y_true, probabilities >= threshold),
                "Specificity": specificity(y_true, probabilities >= threshold),
            }[name]
        )
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append({"Показник": name, "Оцінка": point, "Нижня межа": low, "Верхня межа": high})
    return pd.DataFrame(rows)


def save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close()


def main() -> None:
    x, y = load_uci_dataset()
    class_table = y.value_counts().sort_index().rename(index={0: "Доброякісні", 1: "Злоякісні"})
    class_table.rename("Кількість").to_csv(OUTPUT_DIR / "class_distribution.csv", encoding="utf-8-sig")
    x.describe().T.to_csv(OUTPUT_DIR / "descriptive_statistics.csv", encoding="utf-8-sig")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=SEED,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    search = GridSearchCV(
        pipeline,
        param_grid={"model__C": np.logspace(-3, 3, 13)},
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_

    cv_results = pd.DataFrame(search.cv_results_)[
        ["param_model__C", "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]
    ].rename(
        columns={
            "param_model__C": "C",
            "mean_train_score": "Середній train ROC_AUC",
            "mean_test_score": "Середній CV ROC_AUC",
            "std_test_score": "Стандартне відхилення CV ROC_AUC",
            "rank_test_score": "Ранг",
        }
    )
    cv_results.to_csv(OUTPUT_DIR / "cross_validation.csv", index=False, encoding="utf-8-sig")

    # Поріг визначається лише за out-of-fold прогнозами навчальної вибірки.
    oof_prob = cross_val_predict(model, x_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    fpr_oof, tpr_oof, thresholds_oof = roc_curve(y_train, oof_prob)
    youden = tpr_oof - fpr_oof
    finite = np.isfinite(thresholds_oof)
    best_index = np.arange(len(youden))[finite][np.argmax(youden[finite])]
    chosen_threshold = float(thresholds_oof[best_index])

    train_prob = model.predict_proba(x_train)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    dummy = DummyClassifier(strategy="prior").fit(x_train, y_train)
    dummy_test_prob = dummy.predict_proba(x_test)[:, 1]

    rows = [
        metric_row("Фіктивна модель", "Тестова", y_test.to_numpy(), dummy_test_prob, 0.5),
        metric_row("Логістична регресія", "Навчальна", y_train.to_numpy(), train_prob, chosen_threshold),
        metric_row("Логістична регресія", "Тестова, поріг 0,5", y_test.to_numpy(), test_prob, 0.5),
        metric_row("Логістична регресія", "Тестова, поріг Юдена", y_test.to_numpy(), test_prob, chosen_threshold),
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")

    intervals = bootstrap_intervals(y_test.to_numpy(), test_prob, chosen_threshold)
    intervals.to_csv(OUTPUT_DIR / "bootstrap_intervals.csv", index=False, encoding="utf-8-sig")

    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["model"]
    coefficients = pd.DataFrame(
        {
            "Ознака": x.columns,
            "Коефіцієнт": classifier.coef_[0],
            "Відношення шансів": np.exp(classifier.coef_[0]),
            "Модуль коефіцієнта": np.abs(classifier.coef_[0]),
        }
    ).sort_values("Модуль коефіцієнта", ascending=False)
    coefficients.to_csv(OUTPUT_DIR / "coefficients.csv", index=False, encoding="utf-8-sig")

    perm = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=100,
        random_state=SEED,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "Ознака": x.columns,
            "Середнє зниження ROC_AUC": perm.importances_mean,
            "Стандартне відхилення": perm.importances_std,
        }
    ).sort_values("Середнє зниження ROC_AUC", ascending=False)
    importance.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False, encoding="utf-8-sig")

    # Розподіл класів.
    plt.figure(figsize=(6.4, 4.0))
    ax = sns.barplot(x=class_table.index, y=class_table.values, hue=class_table.index, palette=["#6BAED6", "#CB454A"], legend=False)
    ax.set(title="Розподіл діагнозів", xlabel="Клас", ylabel="Кількість спостережень")
    for patch, value in zip(ax.patches, class_table.values):
        ax.annotate(str(value), (patch.get_x() + patch.get_width() / 2, patch.get_height()), ha="center", va="bottom")
    save_figure("class_distribution.png")

    # Перехресна перевірка параметра C.
    plt.figure(figsize=(7.0, 4.2))
    plt.errorbar(
        cv_results["C"].astype(float),
        cv_results["Середній CV ROC_AUC"],
        yerr=cv_results["Стандартне відхилення CV ROC_AUC"],
        marker="o",
        capsize=3,
        color="#1F4E79",
    )
    plt.xscale("log")
    plt.axvline(float(search.best_params_["model__C"]), color="#CB454A", linestyle="--", label=f"Найкраще C = {search.best_params_['model__C']:.4g}")
    plt.xlabel("Параметр регуляризації C")
    plt.ylabel("ROC AUC")
    plt.title("Підбір гіперпараметра за п'ятикратною перевіркою")
    plt.legend()
    save_figure("cross_validation.png")

    # Матриця помилок на незалежній вибірці.
    test_pred = (test_prob >= chosen_threshold).astype(int)
    matrix = confusion_matrix(y_test, test_pred, labels=[0, 1])
    plt.figure(figsize=(5.4, 4.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Доброякісне", "Злоякісне"], yticklabels=["Доброякісне", "Злоякісне"])
    plt.xlabel("Прогноз")
    plt.ylabel("Фактичний клас")
    plt.title(f"Матриця помилок, поріг = {chosen_threshold:.3f}")
    save_figure("confusion_matrix.png")

    # ROC та Precision Recall.
    fpr, tpr, _ = roc_curve(y_test, test_prob)
    precision_values, recall_values, _ = precision_recall_curve(y_test, test_prob)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].plot(fpr, tpr, color="#1F4E79", linewidth=2, label=f"ROC AUC = {roc_auc_score(y_test, test_prob):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="1 мінус специфічність", ylabel="Чутливість", title="ROC крива")
    axes[0].legend(loc="lower right")
    axes[1].plot(recall_values, precision_values, color="#CB454A", linewidth=2, label=f"PR AUC = {average_precision_score(y_test, test_prob):.3f}")
    axes[1].axhline(y_test.mean(), linestyle="--", color="gray", label="Частка позитивного класу")
    axes[1].set(xlabel="Чутливість", ylabel="Точність позитивного прогнозу", title="Precision Recall крива")
    axes[1].legend(loc="lower left")
    save_figure("roc_pr_curves.png")

    # Калібрування та розподіл імовірностей.
    observed, predicted = calibration_curve(y_test, test_prob, n_bins=8, strategy="quantile")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].plot(predicted, observed, marker="o", linewidth=2, color="#1F4E79")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="Середня прогнозована імовірність", ylabel="Фактична частота", title="Калібрувальна крива")
    axes[1].hist(test_prob[y_test.to_numpy() == 0], bins=12, alpha=0.70, label="Доброякісні", color="#6BAED6")
    axes[1].hist(test_prob[y_test.to_numpy() == 1], bins=12, alpha=0.70, label="Злоякісні", color="#CB454A")
    axes[1].axvline(chosen_threshold, color="black", linestyle="--", label=f"Поріг {chosen_threshold:.3f}")
    axes[1].set(xlabel="Прогнозована імовірність злоякісності", ylabel="Кількість", title="Розподіл прогнозів")
    axes[1].legend()
    save_figure("calibration_probabilities.png")

    # Найбільші стандартизовані коефіцієнти.
    top_coefficients = coefficients.head(12).sort_values("Коефіцієнт")
    plt.figure(figsize=(7.3, 5.2))
    colors = ["#1F4E79" if value > 0 else "#6BAED6" for value in top_coefficients["Коефіцієнт"]]
    plt.barh(top_coefficients["Ознака"], top_coefficients["Коефіцієнт"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Коефіцієнт у стандартизованій моделі")
    plt.title("Ознаки з найбільшим впливом на логіт")
    save_figure("top_coefficients.png")

    # Пермутаційна важливість.
    top_importance = importance.head(12).sort_values("Середнє зниження ROC_AUC")
    plt.figure(figsize=(7.3, 5.2))
    plt.barh(top_importance["Ознака"], top_importance["Середнє зниження ROC_AUC"], xerr=top_importance["Стандартне відхилення"], color="#1F4E79", alpha=0.9, capsize=2)
    plt.xlabel("Зменшення ROC AUC після перестановки")
    plt.title("Пермутаційна важливість ознак")
    save_figure("permutation_importance.png")

    summary = {
        "dataset": {
            "name": "Breast Cancer Wisconsin (Diagnostic)",
            "uci_id": 17,
            "doi": "10.24432/C5DW2B",
            "n_samples": int(len(x)),
            "n_features": int(x.shape[1]),
            "benign": int((y == 0).sum()),
            "malignant": int((y == 1).sum()),
            "missing_values": int(x.isna().sum().sum()),
        },
        "split": {"train": int(len(x_train)), "test": int(len(x_test)), "test_fraction": 0.25, "random_state": SEED},
        "model": {
            "best_C": float(search.best_params_["model__C"]),
            "best_cv_roc_auc": float(search.best_score_),
            "intercept_standardized": float(classifier.intercept_[0]),
            "threshold_youden_training_oof": chosen_threshold,
        },
        "test_metrics_youden": rows[-1],
        "test_metrics_threshold_05": rows[-2],
        "bootstrap_intervals": intervals.to_dict(orient="records"),
        "top_coefficients": coefficients.head(10).to_dict(orient="records"),
        "feature_means": dict(zip(x.columns, scaler.mean_.tolist())),
        "feature_scales": dict(zip(x.columns, scaler.scale_.tolist())),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Аналіз завершено.")
    print(f"Найкраще C: {summary['model']['best_C']}")
    print(f"Поріг Юдена: {chosen_threshold:.4f}")
    print(metrics.to_string(index=False))
    print(f"Результати збережено у: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
