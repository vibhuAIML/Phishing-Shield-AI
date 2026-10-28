"""
train_dnn.py
------------
Trains a DNN on the SMOTE-balanced training set produced by preprocessing.py,
then evaluates it BOTH overall and broken down per-source on the untouched
test set. Per-source evaluation matters here because the 5 source datasets
have very different natural class balances (e.g. mendeley_url_phish is ~91%
legit, phishbd_2026 is ~70% phishing) -- overall accuracy can look great while
one source's precision or recall quietly falls apart. See the printed table
and per_source_eval.csv after running.

Requirements: pip install tensorflow scikit-learn pandas numpy
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

PREPROCESSED_DIR = "./preprocessed"
MODEL_OUT = f"{PREPROCESSED_DIR}/dnn_model.keras"
RESULTS_OUT = f"{PREPROCESSED_DIR}/per_source_eval.csv"
HISTORY_PLOT = f"{PREPROCESSED_DIR}/training_history.png"
CM_PLOT = f"{PREPROCESSED_DIR}/confusion_matrices.png"
ROC_PR_PLOT = f"{PREPROCESSED_DIR}/roc_pr_curves.png"

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading preprocessed splits...")
train_smote = pd.read_csv(f"{PREPROCESSED_DIR}/train_smote.csv")
test_scaled = pd.read_csv(f"{PREPROCESSED_DIR}/test_scaled.csv")

if "source" not in test_scaled.columns:
    raise SystemExit(
        "\n'source' column not found in test_scaled.csv.\n"
        "Per-source evaluation needs it. In preprocessing.py, when you save the "
        "test split, keep the original 'source' column alongside the scaled "
        "features and label (only for the TEST set -- SMOTE'd training rows are "
        "synthetic and don't have a real source, so leave train files as-is).\n"
        "e.g.:  test_scaled_df = X_test_scaled_df.copy()\n"
        "       test_scaled_df['label']  = y_test.values\n"
        "       test_scaled_df['source'] = source_test.values   # <- add this line\n"
    )

drop_cols_train = [c for c in ["label", "source"] if c in train_smote.columns]

# Defensive shuffle: Keras's validation_split slices off the last N% of rows
# rather than sampling randomly. If train_smote.csv happens to be ordered
# (e.g. SMOTE's synthetic minority rows appended at the end), that slice can
# end up nearly single-class, making val_auc uncomputable (prints as
# 0.0000e+00) and leaving EarlyStopping/ReduceLROnPlateau blind. Shuffling
# here makes training robust to that regardless of how the CSV was produced.
train_smote = train_smote.sample(frac=1, random_state=42).reset_index(drop=True)

X_train = train_smote.drop(columns=drop_cols_train).values
y_train = train_smote["label"].values

X_test = test_scaled.drop(columns=["label", "source"]).values
y_test = test_scaled["label"].values
src_test = test_scaled["source"].values

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
n_features = X_train.shape[1]

# ---------------------------------------------------------------------------
# 2. Build DNN
# ---------------------------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8,
                                   restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                       factor=0.5, patience=4, min_lr=1e-6),
]

# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------
n_val = int(len(X_train) * 0.1)
val_labels = y_train[-n_val:]
if len(set(val_labels)) < 2:
    raise SystemExit(
        "\nAborting: the validation_split slice (last 10% of train_smote) is "
        "single-class, so val_auc cannot be computed. This should not happen "
        "now that train_smote.csv is shuffled before use -- check that "
        "preprocessing.py shuffled the data before saving, and that this "
        "script's own pre-fit shuffle above ran successfully.\n"
    )

print("\nTraining...")
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=512,
    callbacks=callbacks,
    verbose=2,
)

model.save(MODEL_OUT)
print(f"\nModel saved to {MODEL_OUT}")

# ---------------------------------------------------------------------------
# 4. Evaluate: OVERALL + PER-SOURCE
# ---------------------------------------------------------------------------
print("\nEvaluating on test set...")
proba = model.predict(X_test, batch_size=1024, verbose=0).ravel()
pred = (proba >= 0.5).astype(int)


def compute_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": len(y_true),
        "phishing_rate": float(np.mean(y_true)),
        "auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


rows = []
overall = compute_metrics(y_test, pred, proba)
overall["source"] = "OVERALL"
rows.append(overall)

for s in sorted(np.unique(src_test)):
    mask = src_test == s
    yt, yp, ypr = y_test[mask], pred[mask], proba[mask]
    if len(set(yt)) < 2:
        print(f"  Skipping '{s}': only one class present in test rows for this source.")
        continue
    m = compute_metrics(yt, yp, ypr)
    m["source"] = s
    rows.append(m)

results_df = pd.DataFrame(rows).set_index("source")
results_df = results_df[[
    "n", "phishing_rate", "auc", "accuracy", "precision", "recall", "f1",
    "false_positives", "false_negatives",
]]

pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print("\n=== Overall + per-source test performance ===")
print(results_df.to_string())

results_df.to_csv(RESULTS_OUT)
print(f"\nSaved detailed results to {RESULTS_OUT}")

# ---------------------------------------------------------------------------
# 5. Flag sources that are underperforming relative to overall
# ---------------------------------------------------------------------------
print("\n=== Warnings: sources that lag the overall numbers ===")
THRESH = 0.10  # flag if a source's metric is more than 10 points below overall
flagged = False
for metric in ["precision", "recall", "f1"]:
    overall_val = results_df.loc["OVERALL", metric]
    for s in results_df.index:
        if s == "OVERALL":
            continue
        val = results_df.loc[s, metric]
        if overall_val - val > THRESH:
            flagged = True
            print(f"  [{s}] {metric} = {val:.3f} is {overall_val - val:.3f} below "
                  f"overall ({overall_val:.3f}) -- n={int(results_df.loc[s, 'n'])}, "
                  f"natural phishing rate={results_df.loc[s, 'phishing_rate']:.3f}")
if not flagged:
    print("  None -- all sources track the overall metrics reasonably well.")

# ---------------------------------------------------------------------------
# 6. Generate & Save Visualizations
# ---------------------------------------------------------------------------
print("\nGenerating evaluation graphs and plots...")

# A. Training History Curves
epochs_range = range(1, len(history.history["loss"]) + 1)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(epochs_range, history.history["loss"], label="Train Loss", marker="o", color="#1f77b4")
axes[0].plot(epochs_range, history.history["val_loss"], label="Val Loss", marker="s", color="#ff7f0e")
axes[0].set_title("Binary Crossentropy Loss", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(epochs_range, history.history["accuracy"], label="Train Accuracy", marker="o", color="#2ca02c")
axes[1].plot(epochs_range, history.history["val_accuracy"], label="Val Accuracy", marker="s", color="#d62728")
axes[1].set_title("Model Accuracy", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# AUC
axes[2].plot(epochs_range, history.history["auc"], label="Train AUC", marker="o", color="#9467bd")
axes[2].plot(epochs_range, history.history["val_auc"], label="Val AUC", marker="s", color="#8c564b")
axes[2].set_title("Area Under ROC (AUC)", fontsize=13, fontweight="bold")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("AUC")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle("Deep Neural Network — Training & Validation History", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(HISTORY_PLOT, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved Training Graphs -> {HISTORY_PLOT}")

# B. Confusion Matrices (Overall + Each Source Dataset)
sources_list = ["OVERALL"] + sorted(np.unique(src_test).tolist())
n_plots = len(sources_list)
cols = 3
rows = (n_plots + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
axes = axes.flatten()

for idx, src in enumerate(sources_list):
    ax = axes[idx]
    if src == "OVERALL":
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        title = f"OVERALL Confusion Matrix\n(Total: {len(y_test):,})"
    else:
        mask = src_test == src
        cm = confusion_matrix(y_test[mask], pred[mask], labels=[0, 1])
        title = f"Source: {src}\n(Total: {np.sum(mask):,})"

    # Percentage normalization per true class
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.empty_like(cm).astype(str)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1f}%)"

    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Legit (0)", "Phishing (1)"],
        yticklabels=["Legit (0)", "Phishing (1)"],
        annot_kws={"size": 11, "fontweight": "bold"}
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)

# Hide any unused subplots
for idx in range(len(sources_list), len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle("Confusion Matrices (Raw Counts & Class %)", fontsize=16, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(CM_PLOT, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved Confusion Matrices -> {CM_PLOT}")

# C. ROC and Precision-Recall Curves
fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

for src in sources_list:
    if src == "OVERALL":
        y_true_s, y_prob_s = y_test, proba
        label_str = f"OVERALL (AUC = {roc_auc_score(y_true_s, y_prob_s):.4f})"
        lw = 2.5
    else:
        mask = src_test == src
        y_true_s, y_prob_s = y_test[mask], proba[mask]
        if len(set(y_true_s)) < 2:
            continue
        label_str = f"{src} (AUC = {roc_auc_score(y_true_s, y_prob_s):.4f})"
        lw = 1.5

    fpr, tpr, _ = roc_curve(y_true_s, y_prob_s)
    prec, rec, _ = precision_recall_curve(y_true_s, y_prob_s)

    ax_roc.plot(fpr, tpr, label=label_str, linewidth=lw)
    ax_pr.plot(rec, prec, label=src, linewidth=lw)

ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance")
ax_roc.set_title("ROC Curves", fontsize=13, fontweight="bold")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend(fontsize=9, loc="lower right")
ax_roc.grid(True, alpha=0.3)

ax_pr.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.legend(fontsize=9, loc="lower left")
ax_pr.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(ROC_PR_PLOT, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved ROC & PR Curves -> {ROC_PR_PLOT}")

print("\nDone. All graphs generated successfully!")
