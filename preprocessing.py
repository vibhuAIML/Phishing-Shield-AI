"""
preprocessing.py
-----------------
Loads url_features_dnn.csv, does a stratified train/test split (stratified
on label x source so every source is represented proportionally in both
splits), scales features, applies SMOTE to the training set only, and saves:

    ./preprocessed/train_smote.csv   (scaled + SMOTE-balanced, NO source col)
    ./preprocessed/test_scaled.csv   (scaled, untouched, WITH source col)
    ./preprocessed/scaler.pkl        (fitted StandardScaler, for reuse at inference)

IMPORTANT FIX included here: the SMOTE-resampled training set is shuffled
before being saved. Without this, synthetic minority-class rows SMOTE
generates get appended at the END of the dataframe. Keras's
`validation_split=0.1` in train_dnn.py does NOT shuffle before splitting --
it just slices off the last 10% of rows -- so that slice would end up
almost single-class, making val_auc uncomputable (shows as 0.0000e+00) and
leaving EarlyStopping/ReduceLROnPlateau (both keyed on val_auc) unable to
actually monitor anything meaningful.

Requirements: pip install pandas numpy scikit-learn imbalanced-learn
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

INPUT_CSV = "./url_features_dnn.csv"
OUT_DIR = "./preprocessed"
RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
print("Loading features...")
df = pd.read_csv(INPUT_CSV)

if "label" not in df.columns or "source" not in df.columns:
    raise SystemExit("Expected 'label' and 'source' columns in the input CSV.")

feature_cols = [c for c in df.columns if c not in ("label", "source")]
X = df[feature_cols].copy()
y = df["label"].copy()
source = df["source"].copy()

# ---------------------------------------------------------------------------
# 2. Stratified train/test split
# ---------------------------------------------------------------------------
# Stratify on label x source jointly so each source's class balance is
# preserved in both train and test (plain label-only stratification could
# still let a small source land almost entirely in one split by chance).
strat_key = source.astype(str) + "_" + y.astype(str)

print(f"Splitting {len(df)} rows (test_size={TEST_SIZE})...")
X_train, X_test, y_train, y_test, source_train, source_test = train_test_split(
    X, y, source,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=strat_key,
)

print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------------------
# 3. Scale (fit on train only, apply to both)
# ---------------------------------------------------------------------------
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

joblib.dump(scaler, f"{OUT_DIR}/scaler.pkl")

# ---------------------------------------------------------------------------
# 4. SMOTE on training set only
# ---------------------------------------------------------------------------
print("Applying SMOTE to training set...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled_df, y_train)

print(f"  Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"  After SMOTE:  {pd.Series(y_train_res).value_counts().to_dict()}")

train_smote_df = pd.DataFrame(X_train_res, columns=feature_cols)
train_smote_df["label"] = y_train_res.values

# --- THE FIX: shuffle before saving --------------------------------------
# SMOTE appends synthetic rows at the end of the array. Shuffle now so that
# any downstream validation_split (which slices, not samples) sees a mix
# of original + synthetic, majority + minority rows.
train_smote_df = train_smote_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5. Save test set (scaled, WITH source column, untouched by SMOTE)
# ---------------------------------------------------------------------------
test_scaled_df = X_test_scaled_df.copy()
test_scaled_df["label"] = y_test.values
test_scaled_df["source"] = source_test.values
test_scaled_df = test_scaled_df.reset_index(drop=True)

train_smote_df.to_csv(f"{OUT_DIR}/train_smote.csv", index=False)
test_scaled_df.to_csv(f"{OUT_DIR}/test_scaled.csv", index=False)

print(f"\nSaved:")
print(f"  {OUT_DIR}/train_smote.csv  {train_smote_df.shape}  (shuffled)")
print(f"  {OUT_DIR}/test_scaled.csv  {test_scaled_df.shape}  (source preserved)")
print(f"  {OUT_DIR}/scaler.pkl")
print("\nDone.")
