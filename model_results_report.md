# URL Phishing Deep Neural Network — Comprehensive Evaluation Results

**Total Test URLs Evaluated**: `126,511`  
**Model File**: `preprocessed/dnn_model.keras`  

## 1. Key Performance Highlights

- **Overall Accuracy**: **`96.23%`**
- **Overall F1-Score**: **`0.9576`**
- **Overall ROC-AUC**: **`0.9918`**
- **Overall PR-AUC (Avg Precision)**: **`0.9907`**
- **Overall Precision (PPV)**: **`95.35%`**
- **Overall Recall (Sensitivity)**: **`96.18%`**
- **Overall Specificity (TNR)**: **`96.27%`**
- **Matthews Correlation Coefficient (MCC)**: **`0.9237`**

## 2. Overall Confusion Matrix Breakdown

| Metric | Sample Count | Percentage |
| :--- | :--- | :--- |
| **True Negatives (TN)** - Legit correctly classified | `67,863` | `96.27%` of Legit |
| **True Positives (TP)** - Phish correctly classified | `53,879` | `96.18%` of Phishing |
| **False Positives (FP)** - Legit mistakenly flagged as Phish | `2,630` | `3.73%` of Legit |
| **False Negatives (FN)** - Phish missed as Legit | `2,139` | `3.82%` of Phishing |

## 3. Full Benchmark Table Across Datasets

| Source Dataset | Samples ($n$) | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC | PR-AUC | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OVERALL** | 126,511 | 96.23% | 95.35% | 96.18% | 96.27% | **0.9576** | 0.9918 | 0.9907 | 0.9237 |
| **dataset_phishing** | 2,280 | 81.67% | 78.30% | 87.52% | 75.83% | **0.8266** | 0.9162 | 0.9228 | 0.6378 |
| **mendeley_url_phish** | 21,584 | 94.39% | 62.06% | 93.59% | 94.47% | **0.7463** | 0.9853 | 0.9224 | 0.7353 |
| **phishbd_2026** | 26,227 | 94.16% | 93.91% | 98.01% | 85.17% | **0.9592** | 0.9838 | 0.9926 | 0.8592 |
| **phiusiil** | 46,518 | 97.85% | 99.97% | 94.95% | 99.98% | **0.9739** | 0.9987 | 0.9988 | 0.9566 |
| **raw_dataset** | 29,902 | 97.96% | 99.37% | 96.54% | 99.38% | **0.9793** | 0.9933 | 0.9954 | 0.9597 |

## 4. Confusion Matrix Counts Per Dataset

| Source Dataset | TP | TN | FP | FN | FPR (%) | FNR (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OVERALL** | 53,879 | 67,863 | 2,630 | 2,139 | 3.73% | 3.82% |
| **dataset_phishing** | 996 | 866 | 276 | 142 | 24.17% | 12.48% |
| **mendeley_url_phish** | 1,781 | 18,592 | 1,089 | 122 | 5.53% | 6.41% |
| **phishbd_2026** | 17,995 | 6,700 | 1,167 | 365 | 14.83% | 1.99% |
| **phiusiil** | 18,674 | 26,845 | 6 | 993 | 0.02% | 5.05% |
| **raw_dataset** | 14,433 | 14,860 | 92 | 517 | 0.62% | 3.46% |
