"""
explore_dataset.py
==================
Displays dataset dimensions (rows and columns) and class balance analysis.
"""

import os
import pandas as pd

RAW_PATH = "merged_raw_dataset.csv"
FEAT_PATH = "url_features_dnn.csv"

def main():
    print("=" * 70)
    print("                 DATASET EXPLORATION SUMMARY")
    print("=" * 70)

    # 1. Raw Dataset Dimensions
    if os.path.exists(RAW_PATH):
        print(f"\nLoading '{RAW_PATH}'...")
        df_raw = pd.read_csv(RAW_PATH)
        n_rows, n_cols = df_raw.shape
        print("\n" + "-" * 50)
        print("1. DATASET DIMENSIONS (RAW DATASET)")
        print("-" * 50)
        print(f"  • Total Rows (URLs) : {n_rows:,}")
        print(f"  • Total Columns     : {n_cols} {list(df_raw.columns)}")
        print(f"  • Missing Values    : {df_raw.isnull().sum().sum()}")
        print(f"  • Duplicate URLs    : {df_raw['url'].duplicated().sum()}")

        # 2. Balance Check
        print("\n" + "-" * 50)
        print("2. CLASS BALANCE ANALYSIS")
        print("-" * 50)
        counts = df_raw["label"].value_counts()
        pcts = df_raw["label"].value_counts(normalize=True) * 100

        n_legit = counts.get(0, 0)
        n_phish = counts.get(1, 0)
        pct_legit = pcts.get(0, 0.0)
        pct_phish = pcts.get(1, 0.0)

        print(f"  • Legitimate URLs (0) : {n_legit:>8,}  ({pct_legit:.2f}%)")
        print(f"  • Phishing URLs   (1) : {n_phish:>8,}  ({pct_phish:.2f}%)")
        print(f"  • Imbalance Ratio     : 1 : {n_phish / n_legit:.2f}")

        # Verdict on Balance
        print("\n  >> BALANCE STATUS:")
        if abs(pct_legit - pct_phish) < 5:
            print("     [x] The dataset is BALANCED (~50/50 split).")
        elif abs(pct_legit - pct_phish) < 15:
            print("     [!] The dataset is MODERATELY BALANCED (~56% legit vs ~44% phishing).")
        else:
            print("     [!] The dataset is SIGNIFICANTLY IMBALANCED.")

        # 3. Source Breakdown
        if "source" in df_raw.columns:
            print("\n" + "-" * 50)
            print("3. BALANCE BREAKDOWN BY DATA SOURCE")
            print("-" * 50)
            source_summary = df_raw.groupby(["source", "label"]).size().unstack(fill_value=0)
            source_summary["Total"] = source_summary.sum(axis=1)
            source_summary["% Phishing"] = (source_summary.get(1, 0) / source_summary["Total"]) * 100
            source_summary["Status"] = source_summary["% Phishing"].apply(
                lambda p: "Balanced (50/50)" if 45 <= p <= 55 else ("Phish-Heavy" if p > 55 else "Legit-Heavy")
            )
            source_summary = source_summary.rename(columns={0: "Legit (0)", 1: "Phish (1)"})
            print(source_summary[["Total", "Legit (0)", "Phish (1)", "% Phishing", "Status"]].to_string())

    # 4. Feature Dataset Dimensions
    if os.path.exists(FEAT_PATH):
        print("\n" + "-" * 50)
        print("4. FEATURE MATRIX DIMENSIONS (AFTER FEATURE EXTRACTION)")
        print("-" * 50)
        df_feat = pd.read_csv(FEAT_PATH, nrows=5)
        # Fast row count check
        total_feat_cols = len(df_feat.columns)
        print(f"  • Extracted Features File : {FEAT_PATH}")
        print(f"  • Total Columns           : {total_feat_cols} (49 numeric features + label + source)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
