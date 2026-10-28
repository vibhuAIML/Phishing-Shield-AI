"""
URL Phishing Detection — Exploratory Data Analysis (EDA)
==========================================================
Step 1 of the pipeline: explore the raw dataset and the extracted
feature set before preprocessing / balancing / modeling.

Inputs (place these in the same folder as this script):
    merged_raw_dataset.csv   (url, label, source)
    url_features_dnn.csv     (49 numeric features, label, source)

Outputs (written to an "eda_outputs" folder):
    class_balance.png
    source_breakdown.png
    feature_distributions.png
    correlation_heatmap.png
    top_correlated_with_label.png
    eda_summary.txt
"""

import os
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")  # safe for headless / script runs
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

RAW_PATH = "merged_raw_dataset.csv"
FEAT_PATH = "url_features_dnn.csv"
OUT_DIR = "eda_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

summary_lines = []


def log(msg):
    print(msg)
    summary_lines.append(msg)


# ----------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------
raw = pd.read_csv(RAW_PATH)
feat = pd.read_csv(FEAT_PATH)

log("=" * 60)
log("RAW DATASET")
log("=" * 60)
log(f"Shape: {raw.shape}")
log(f"Missing values:\n{raw.isnull().sum().to_string()}")
log(f"Duplicate URLs: {raw['url'].duplicated().sum()}")

log("\n" + "=" * 60)
log("FEATURE DATASET")
log("=" * 60)
log(f"Shape: {feat.shape}")
log(f"Missing values total: {feat.isnull().sum().sum()}")

# ----------------------------------------------------------------------
# 2. Class balance
# ----------------------------------------------------------------------
label_counts = raw["label"].value_counts().sort_index()
label_pct = (label_counts / label_counts.sum() * 100).round(2)
log(f"\nClass balance (0=legit, 1=phishing):\n{label_counts.to_string()}")
log(f"Class balance (%):\n{label_pct.to_string()}")

fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(x=label_counts.index.astype(str), y=label_counts.values, ax=ax,
            palette=["#4C9A2A", "#D9534F"])
ax.set_xlabel("Label (0=Legit, 1=Phishing)")
ax.set_ylabel("Count")
ax.set_title("Class Balance")
for i, v in enumerate(label_counts.values):
    ax.text(i, v, f"{v:,}\n({label_pct.values[i]}%)", ha="center", va="bottom")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/class_balance.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 3. Source breakdown (counts + label mix per source)
# ----------------------------------------------------------------------
source_counts = raw["source"].value_counts()
log(f"\nRows per source dataset:\n{source_counts.to_string()}")

source_label = pd.crosstab(raw["source"], raw["label"], normalize="index") * 100
log(f"\nLabel mix (%) per source:\n{source_label.round(2).to_string()}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(x=source_counts.values, y=source_counts.index, ax=axes[0], color="#4A90D9")
axes[0].set_title("Rows per Source Dataset")
axes[0].set_xlabel("Count")

source_label.plot(kind="barh", stacked=True, ax=axes[1], color=["#4C9A2A", "#D9534F"])
axes[1].set_title("Label Mix (%) per Source")
axes[1].set_xlabel("Percent")
axes[1].legend(["Legit (0)", "Phishing (1)"], loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/source_breakdown.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 4. Feature distributions (a representative subset), split by label
# ----------------------------------------------------------------------
dist_features = [
    "url_length", "hostname_length", "count_dots", "count_hyphens",
    "digit_ratio", "num_subdomains", "url_entropy", "suspicious_keyword_count"
]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.flat, dist_features):
    # clip extreme outliers just for readability of the plot
    upper = feat[col].quantile(0.99)
    data = feat[[col, "label"]].copy()
    data[col] = data[col].clip(upper=upper)
    sns.histplot(data=data, x=col, hue="label", bins=40, ax=ax,
                 palette=["#4C9A2A", "#D9534F"], element="step", stat="density",
                 common_norm=False)
    ax.set_title(col)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_distributions.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 5. Correlation heatmap (numeric features only)
# ----------------------------------------------------------------------
numeric_cols = feat.select_dtypes(include=[np.number]).columns.tolist()
corr = feat[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(18, 15))
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, square=True,
            cbar_kws={"shrink": 0.6})
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 6. Features most correlated with the label
# ----------------------------------------------------------------------
label_corr = corr["label"].drop("label").sort_values(key=abs, ascending=False)
log(f"\nTop 15 features correlated with label:\n{label_corr.head(15).to_string()}")

fig, ax = plt.subplots(figsize=(8, 6))
top15 = label_corr.head(15)
colors = ["#D9534F" if v > 0 else "#4A90D9" for v in top15.values]
sns.barplot(x=top15.values, y=top15.index, ax=ax, palette=colors)
ax.set_title("Top 15 Features Correlated with Label")
ax.set_xlabel("Correlation with label")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/top_correlated_with_label.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 7. Highly correlated feature pairs (candidates to drop before modeling)
# ----------------------------------------------------------------------
corr_no_label = feat[numeric_cols].drop(columns=["label"]).corr()
pairs = (
    corr_no_label.where(np.triu(np.ones(corr_no_label.shape), k=1).astype(bool))
    .stack()
    .sort_values(key=abs, ascending=False)
)
high_pairs = pairs[pairs.abs() > 0.85]
log(f"\nFeature pairs with |correlation| > 0.85 ({len(high_pairs)} pairs):")
log(high_pairs.head(20).to_string())

# ----------------------------------------------------------------------
# Save summary
# ----------------------------------------------------------------------
with open(f"{OUT_DIR}/eda_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

print(f"\nAll EDA outputs saved to ./{OUT_DIR}/")
