"""
generate_results.py
===================
Evaluates the trained DNN model on test_scaled.csv and generates comprehensive 
results files with all classification metrics (F1-score, Precision, Recall, Accuracy,
Specificity, Balanced Accuracy, ROC-AUC, PR-AUC, MCC, Confusion Matrix elements).

Outputs:
  - model_results_comprehensive.csv
  - model_results_report.md
  - model_results_summary.txt
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, balanced_accuracy_score, log_loss, classification_report
)

TEST_PATH = "preprocessed/test_scaled.csv"
MODEL_PATH = "preprocessed/dnn_model.keras"
OUT_CSV = "model_results_comprehensive.csv"
OUT_MD = "model_results_report.md"
OUT_TXT = "model_results_summary.txt"


def evaluate_slice(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = len(y_true)
    
    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # False Positive Rate = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # False Negative Rate = FN / (FN + TP)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    has_both_classes = len(set(y_true)) > 1
    roc_auc = roc_auc_score(y_true, y_prob) if has_both_classes else np.nan
    pr_auc = average_precision_score(y_true, y_prob) if has_both_classes else np.nan
    mcc = matthews_corrcoef(y_true, y_pred) if has_both_classes else np.nan
    bacc = balanced_accuracy_score(y_true, y_pred) if has_both_classes else np.nan
    loss = log_loss(y_true, y_prob, labels=[0, 1]) if has_both_classes else np.nan

    return {
        "Total_Samples (n)": int(total),
        "Phishing_Ratio": float(np.mean(y_true)),
        "Accuracy (%)": float(accuracy_score(y_true, y_pred) * 100),
        "Precision (%)": float(precision_score(y_true, y_pred, zero_division=0) * 100),
        "Recall_Sensitivity (%)": float(recall_score(y_true, y_pred, zero_division=0) * 100),
        "Specificity (%)": float(specificity * 100),
        "F1_Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "Balanced_Accuracy (%)": float(bacc * 100) if not np.isnan(bacc) else np.nan,
        "ROC_AUC": float(roc_auc),
        "PR_AUC": float(pr_auc),
        "MCC": float(mcc),
        "Log_Loss": float(loss),
        "True_Positives (TP)": int(tp),
        "True_Negatives (TN)": int(tn),
        "False_Positives (FP)": int(fp),
        "False_Negatives (FN)": int(fn),
        "False_Positive_Rate (%)": float(fpr * 100),
        "False_Negative_Rate (%)": float(fnr * 100),
    }


def main():
    print("Loading test data and trained model...")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Missing '{TEST_PATH}'")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing '{MODEL_PATH}'")

    test_df = pd.read_csv(TEST_PATH)
    drop_cols = [c for c in ["label", "source"] if c in test_df.columns]
    X_test = test_df.drop(columns=drop_cols).values
    y_test = test_df["label"].values
    src_test = test_df["source"].values if "source" in test_df.columns else np.array(["all"] * len(test_df))

    model = keras.models.load_model(MODEL_PATH)
    print("Predicting probabilities on test set...")
    y_prob = model.predict(X_test, batch_size=1024, verbose=0).ravel()
    y_pred = (y_prob >= 0.50).astype(int)

    # 1. Overall Evaluation
    records = []
    overall_dict = evaluate_slice(y_test, y_pred, y_prob)
    overall_dict["Source"] = "OVERALL"
    records.append(overall_dict)

    # 2. Per-Source Evaluation
    for src in sorted(np.unique(src_test)):
        mask = src_test == src
        yt, yp, ypr = y_test[mask], y_pred[mask], y_prob[mask]
        res = evaluate_slice(yt, yp, ypr)
        res["Source"] = src
        records.append(res)

    results_df = pd.DataFrame(records)
    # Reorder columns with Source first
    cols = ["Source"] + [c for c in results_df.columns if c != "Source"]
    results_df = results_df[cols]

    # Save to CSV
    results_df.to_csv(OUT_CSV, index=False)
    print(f"\n[✓] Saved Comprehensive CSV Results: {OUT_CSV}")

    # Generate Text Summary
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("                    DEEP LEARNING MODEL EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. OVERALL CLASSIFICATION REPORT\n")
        f.write("-" * 60 + "\n")
        f.write(classification_report(y_test, y_pred, target_names=["Legitimate (0)", "Phishing (1)"], digits=4))
        f.write("\n\n2. COMPREHENSIVE METRICS BREAKDOWN\n")
        f.write("-" * 80 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n" + "=" * 80 + "\n")

    print(f"[✓] Saved Text Summary: {OUT_TXT}")

    # Generate Markdown Report
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# URL Phishing Deep Neural Network — Comprehensive Evaluation Results\n\n")
        f.write(f"**Total Test URLs Evaluated**: `{len(y_test):,}`  \n")
        f.write(f"**Model File**: `preprocessed/dnn_model.keras`  \n\n")

        f.write("## 1. Key Performance Highlights\n\n")
        f.write(f"- **Overall Accuracy**: **`{overall_dict['Accuracy (%)']:.2f}%`**\n")
        f.write(f"- **Overall F1-Score**: **`{overall_dict['F1_Score']:.4f}`**\n")
        f.write(f"- **Overall ROC-AUC**: **`{overall_dict['ROC_AUC']:.4f}`**\n")
        f.write(f"- **Overall PR-AUC (Avg Precision)**: **`{overall_dict['PR_AUC']:.4f}`**\n")
        f.write(f"- **Overall Precision (PPV)**: **`{overall_dict['Precision (%)']:.2f}%`**\n")
        f.write(f"- **Overall Recall (Sensitivity)**: **`{overall_dict['Recall_Sensitivity (%)']:.2f}%`**\n")
        f.write(f"- **Overall Specificity (TNR)**: **`{overall_dict['Specificity (%)']:.2f}%`**\n")
        f.write(f"- **Matthews Correlation Coefficient (MCC)**: **`{overall_dict['MCC']:.4f}`**\n\n")

        f.write("## 2. Overall Confusion Matrix Breakdown\n\n")
        f.write("| Metric | Sample Count | Percentage |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write(f"| **True Negatives (TN)** - Legit correctly classified | `{overall_dict['True_Negatives (TN)']:,}` | `{overall_dict['Specificity (%)']:.2f}%` of Legit |\n")
        f.write(f"| **True Positives (TP)** - Phish correctly classified | `{overall_dict['True_Positives (TP)']:,}` | `{overall_dict['Recall_Sensitivity (%)']:.2f}%` of Phishing |\n")
        f.write(f"| **False Positives (FP)** - Legit mistakenly flagged as Phish | `{overall_dict['False_Positives (FP)']:,}` | `{overall_dict['False_Positive_Rate (%)']:.2f}%` of Legit |\n")
        f.write(f"| **False Negatives (FN)** - Phish missed as Legit | `{overall_dict['False_Negatives (FN)']:,}` | `{overall_dict['False_Negative_Rate (%)']:.2f}%` of Phishing |\n\n")

        f.write("## 3. Full Benchmark Table Across Datasets\n\n")
        f.write("| Source Dataset | Samples ($n$) | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC | PR-AUC | MCC |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for _, r in results_df.iterrows():
            f.write(f"| **{r['Source']}** | {int(r['Total_Samples (n)']):,} | {r['Accuracy (%)']:.2f}% | {r['Precision (%)']:.2f}% | {r['Recall_Sensitivity (%)']:.2f}% | {r['Specificity (%)']:.2f}% | **{r['F1_Score']:.4f}** | {r['ROC_AUC']:.4f} | {r['PR_AUC']:.4f} | {r['MCC']:.4f} |\n")

        f.write("\n## 4. Confusion Matrix Counts Per Dataset\n\n")
        f.write("| Source Dataset | TP | TN | FP | FN | FPR (%) | FNR (%) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for _, r in results_df.iterrows():
            f.write(f"| **{r['Source']}** | {int(r['True_Positives (TP)']):,} | {int(r['True_Negatives (TN)']):,} | {int(r['False_Positives (FP)']):,} | {int(r['False_Negatives (FN)']):,} | {r['False_Positive_Rate (%)']:.2f}% | {r['False_Negative_Rate (%)']:.2f}% |\n")

    print(f"[✓] Saved Markdown Report: {OUT_MD}")
    print("\n" + "=" * 70)
    print("                    FINAL RESULTS TABLE")
    print("=" * 70)
    summary_table = results_df[["Source", "Total_Samples (n)", "Accuracy (%)", "Precision (%)", "Recall_Sensitivity (%)", "F1_Score", "ROC_AUC", "MCC"]]
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
