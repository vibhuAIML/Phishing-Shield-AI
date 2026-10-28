"""
threshold_search_reweighted.py
-------------------------------
Finds a SINGLE global decision threshold (no source lookup needed at
inference time -- source isn't available for a real incoming URL anyway).

The catch: the test set's "natural" blended phishing rate (~44%) is an
artifact of combining 5 datasets, most of which are artificially balanced
benchmark sets (raw_dataset, dataset_phishing ~50/50, phiusiil ~42%). Real
deployment traffic looks nothing like that -- most URLs anyone encounters
are legitimate, with phishing a small minority. mendeley_url_phish's
natural rate (~8.8%) is the closest proxy in this data to that reality, but
even that may not match your actual deployment traffic.

Rather than physically resampling the test set (which throws away data and
adds sampling noise), this script uses SAMPLE WEIGHTING: it reweights the
existing test predictions so that, in aggregate, they behave as if drawn
from a population with a chosen phishing prevalence. This lets precision/
recall/F1 be computed as they WOULD look at that prevalence, without
discarding any rows.

It sweeps thresholds under several assumed real-world prevalences (1%, 2%,
5%, 8.8% [mendeley's natural rate], 10%, 20%) so you can see how the
"right" threshold shifts as the assumed base rate drops, and pick one
deliberately rather than defaulting to whatever the blended test set implies.

Requirements: pip install pandas numpy scikit-learn tensorflow
"""

import numpy as np
import pandas as pd
from tensorflow import keras

PREPROCESSED_DIR = "./preprocessed"
MODEL_PATH = f"{PREPROCESSED_DIR}/dnn_model.keras"
OUT_CSV = f"{PREPROCESSED_DIR}/reweighted_threshold_search.csv"

# Prevalences to evaluate under. Edit this list if you have a better
# estimate of your actual deployment phishing rate.
ASSUMED_PREVALENCES = [0.01, 0.02, 0.05, 0.088, 0.10, 0.20]

# The prevalence to use for the single FINAL recommended threshold.
# 0.088 = mendeley_url_phish's natural rate, the closest thing in this data
# to "real" traffic. Change this if you have domain knowledge of your own
# deployment's true phishing rate.
TARGET_PREVALENCE_FOR_RECOMMENDATION = 0.088

THRESHOLDS_TO_SWEEP = np.linspace(0.01, 0.99, 197)


def weighted_confusion_counts(y_true, proba, threshold, sample_weight):
    pred = (proba >= threshold).astype(int)
    tp = np.sum(sample_weight * ((pred == 1) & (y_true == 1)))
    fp = np.sum(sample_weight * ((pred == 1) & (y_true == 0)))
    fn = np.sum(sample_weight * ((pred == 0) & (y_true == 1)))
    tn = np.sum(sample_weight * ((pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def weighted_precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def make_prevalence_weights(y_true, target_prevalence):
    """
    Returns per-row weights such that the WEIGHTED prevalence of the
    positive class equals target_prevalence, while weights within each
    class stay uniform (so no individual row is up/down-weighted based on
    anything except its class). Total weight mass is preserved at len(y_true).
    """
    n = len(y_true)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return np.ones(n)

    w_pos = target_prevalence * n / n_pos
    w_neg = (1 - target_prevalence) * n / n_neg

    weights = np.where(y_true == 1, w_pos, w_neg)
    return weights


def best_threshold_for_f1_weighted(y_true, proba, sample_weight):
    best_f1, best_t, best_p, best_r = -1, 0.5, 0.0, 0.0
    for t in THRESHOLDS_TO_SWEEP:
        tp, fp, fn, tn = weighted_confusion_counts(y_true, proba, t, sample_weight)
        p, r, f1 = weighted_precision_recall_f1(tp, fp, fn)
        if f1 > best_f1:
            best_f1, best_t, best_p, best_r = f1, t, p, r
    return best_t, best_p, best_r, best_f1


print("Loading model and test set...")
model = keras.models.load_model(MODEL_PATH)
test_scaled = pd.read_csv(f"{PREPROCESSED_DIR}/test_scaled.csv")

X_test = test_scaled.drop(columns=["label", "source"]).values
y_test = test_scaled["label"].values

proba = model.predict(X_test, batch_size=1024, verbose=0).ravel()

blended_rate = float(np.mean(y_test))
print(f"\nBlended (actual) test set phishing rate: {blended_rate:.3f}")
print("This is an artifact of combining 5 datasets, most artificially "
      "balanced -- NOT a real-world estimate.\n")

rows = []
for prev in ASSUMED_PREVALENCES:
    weights = make_prevalence_weights(y_test, prev)
    t, p, r, f1 = best_threshold_for_f1_weighted(y_test, proba, weights)
    rows.append({
        "assumed_prevalence": prev,
        "best_threshold": round(t, 4),
        "precision_at_that_prevalence": p,
        "recall_at_that_prevalence": r,
        "f1_at_that_prevalence": f1,
    })

results_df = pd.DataFrame(rows).set_index("assumed_prevalence")
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print("=== Best global threshold under different assumed real-world prevalences ===")
print(results_df.to_string())
results_df.to_csv(OUT_CSV)
print(f"\nSaved to {OUT_CSV}")

# --- Final single recommendation ---
weights = make_prevalence_weights(y_test, TARGET_PREVALENCE_FOR_RECOMMENDATION)
t, p, r, f1 = best_threshold_for_f1_weighted(y_test, proba, weights)

print(f"\n=== Recommended single global threshold "
      f"(assuming {TARGET_PREVALENCE_FOR_RECOMMENDATION:.1%} real-world phishing rate) ===")
print(f"  threshold = {t:.4f}")
print(f"  expected precision = {p:.4f}")
print(f"  expected recall    = {r:.4f}")
print(f"  expected f1        = {f1:.4f}")

print("""
IMPORTANT CAVEATS:
1. This threshold is fit and evaluated on the same test set it's chosen
   from -- treat it as directionally right, not a number to trust blindly.
   Ideally repeat this on a separate held-out validation set before locking
   in a production threshold.
2. The "assumed prevalence" is a guess (mendeley_url_phish's natural rate,
   the closest thing to real traffic in this data). If you know or can
   estimate your actual deployment's phishing rate, change
   TARGET_PREVALENCE_FOR_RECOMMENDATION at the top of this script and rerun.
3. This produces ONE threshold with no dependency on 'source' -- it's safe
   to use directly at inference time on a URL with unknown origin.
""")
