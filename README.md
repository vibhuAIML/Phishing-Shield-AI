# 🛡️ Multimodal Phishing Detection System
### Dual-Engine Deep Learning Defense: Email Transformer (DistilBERT) + URL Deep Neural Network (DNN)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow / Keras 3](https://img.shields.io/badge/TensorFlow%20%2F%20Keras%203-FF6F00.svg?logo=tensorflow&logoColor=white)](https://keras.io/)
[![HuggingFace DistilBERT](https://img.shields.io/badge/HuggingFace-DistilBERT-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co/distilbert-base-uncased)
[![Flask REST API](https://img.shields.io/badge/Flask-Web%20Dashboard%20%26%20API-000000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end cybersecurity system designed to identify, analyze, and neutralize sophisticated email and URL phishing attacks in real time. The system couples a fine-tuned **DistilBERT NLP Transformer** (for semantic intent and social engineering detection) with an optimized **49-feature Deep Neural Network (DNN)** (for structural, statistical, and lexical URL inspection), fused via **Bayesian Noisy-OR Threat Fusion** with zero-trust malicious URL overrides.

It includes a full-featured **Cybersecurity Operations Center (SOC) Web Dashboard**, a **REST API**, and an automated pipeline CLI.

---

## 📑 Table of Contents

- [Key Capabilities](#-key-capabilities)
- [System Architecture & Flow](#-system-architecture--flow)
- [Model Performance & Benchmarks](#-model-performance--benchmarks)
  - [Overall Multi-Engine Comparison](#overall-multi-engine-comparison)
  - [URL DNN Benchmark Across 5 Datasets](#url-dnn-benchmark-across-5-datasets-n--126511)
  - [Email DistilBERT Performance](#email-distilbert-performance-n--8395)
  - [Multimodal Fusion Results](#multimodal-fusion-results-n--182364)
- [Neural Network & Transformer Architectures](#-neural-network--transformer-architectures)
  - [1. URL Lexical & Structural DNN](#1-url-lexical--structural-dnn)
  - [2. Email Intent DistilBERT](#2-email-intent-distilbert)
  - [3. Decision Fusion Engine](#3-decision-fusion-engine)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Quick Start](#-installation--quick-start)
  - [1. Clone and Set Up Virtual Environment](#1-clone-and-set-up-virtual-environment)
  - [2. Install Dependencies](#2-install-dependencies)
- [Running the SOC Web Dashboard](#-running-the-soc-web-dashboard)
- [REST API Reference](#-rest-api-reference)
- [CLI & Pipeline Execution (`main.py`)](#-cli--pipeline-execution-mainpy)
- [Hardware & Optimization](#-hardware--optimization)
- [License](#-license)

---

## ⚡ Key Capabilities

- **Multimodal Threat Fusion**: Jointly analyzes email language (urgency, credential intimidation, pretexting) and deep lexical URL features, resolving cross-domain evasion attacks.
- **High-Throughput Offline URL Inspection**: Extracts **49 hand-crafted statistical and lexical features** in microseconds without external DNS/WHOIS round-trips, eliminating attacker probe awareness and latency bottlenecks.
- **Quishing & Attachment Scanner**: Decodes 2D QR codes (Quishing) inside image attachments with OpenCV and dissects PDF streams for `/URI` actions and `/JavaScript` execution.
- **Email Header Security & Spoofing Defense**: Audits RFC 5322 MIME headers for **SPF, DKIM, and DMARC** compliance, flagging Display Name brand impersonation (e.g. PayPal, Microsoft).
- **Enterprise Mail Flow Gateways**: Native connectors for **Microsoft 365** (MS Graph API) and **Google Workspace** (Gmail REST API) with automated threat quarantine and category labeling.
- **State-of-the-Art NLP Semantics**: Fine-tuned on **182,364 real-world emails** using HuggingFace DistilBERT with sub-word tokenization and attention mechanisms.
- **Zero-Trust URL Override**: Employs a Bayesian Noisy-OR probabilistic fusion with an emergency safety trigger: any high-confidence malicious URL ($P_{\text{URL}} \ge 0.85$) immediately flags the email as Phishing even if text mimics legitimate correspondence.
- **SOC Analyst Feedback & Continuous Retraining**: SQLite-backed human-in-the-loop triage system (`feedback_store.db`) allowing security analysts to report false positives/negatives and trigger automated retraining.
- **Production Containerization & Task Queue**: Distributed asynchronous worker pipeline with **Celery and Redis**, fully deployable via **Docker Compose**.
- **Interactive SOC-Style Web Application**: Glassmorphism dark-mode UI with live animated risk gauges, structural breakdown charts, 5 inspection tabs, 13 pre-loaded real-world threat scenarios, and actionable remediation advice.
- **Production REST API**: Scalable endpoints (`/api/predict/email`, `/api/predict/url`, `/api/scan/headers`, `/api/scan/attachment`, `/api/gateway/scan`, `/api/feedback`, `/api/retrain/trigger`, `/api/system/status`).

---

## 🏗️ System Architecture & Flow

```mermaid
flowchart TD
    In([User / Security Gateway Input]) --> InputType{Input Type}
    
    %% Email Branch
    InputType -->|Email Subject & Body| ExtractURLs[Regex URL Extractor]
    ExtractURLs --> DistilBERT[DistilBERT Transformer\nTokenization & Sequence Classification]
    DistilBERT -->|Text Phishing Prob: P_email| Fusion[Multimodal Bayesian\nNoisy-OR Fusion Engine]
    
    ExtractURLs -->|Extracted Link 1..N| Extractor[49-Feature URL Extractor\nLexical, Structural & Entropy]
    
    %% URL Direct Branch
    InputType -->|Single URL| Extractor
    
    %% Feature Processing & DNN
    Extractor --> Scaler[StandardScaler Normalization]
    Scaler --> DNN[URL Deep Neural Network\nDense 128 -> 64 -> 32 -> Sigmoid]
    DNN -->|URL Phishing Prob: P_url| Fusion
    
    %% Decision Logic
    Fusion --> OverrideCheck{P_url >= 0.85\nOverride?}
    OverrideCheck -->|Yes| OverrideVerdict[Verdict: PHISHING\nConfidence: MAX P_url]
    OverrideCheck -->|No| ProbFormula["P_combined = 1 - (1 - P_email) * (1 - max(P_url))"]
    
    OverrideVerdict --> Verdict[Unified Verdict & SOC Risk Level]
    ProbFormula --> Verdict
    
    Verdict --> Output[Web Dashboard & REST API Output\n- Risk Score (0-100%)\n- Severity: Safe / Suspicious / Phishing\n- Feature Breakdown & Explanations\n- Security Recommendations]
```

---

## 📊 Model Performance & Benchmarks

### Overall Multi-Engine Comparison

| Model / Subsystem | Target Modality | Test Samples ($n$) | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Email DistilBERT** | Subject & Body NLP | 18,237 | **99.71%** | **99.65%** | **99.78%** | **0.9972** | **0.9999** |
| **URL DNN (49 Features)** | Lexical / Structural | 126,511 | **96.23%** | **95.35%** | **96.18%** | **0.9576** | **0.9918** |
| **Merged Dual-Engine** | Multimodal Fusion | 182,364 | **99.60%** | **98.40%** | **99.60%** | **0.9919** | **0.9972** |

> [!NOTE]
> In the merged dual-engine, **Recall is 100.00%** on the combined evaluation test set — eliminating false negatives for phishing threats.

---

### URL DNN Benchmark Across 5 Datasets ($n = 126,511$)

Evaluated on **126,511 unseen hold-out test URLs** spanning 5 diverse benchmark distributions:

| Dataset Source | Samples ($n$) | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC | PR-AUC | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OVERALL COMPOSITE** | **126,511** | **96.23%** | **95.35%** | **96.18%** | **96.27%** | **0.9576** | **0.9918** | **0.9907** | **0.9237** |
| `raw_dataset` | 29,902 | **97.96%** | 99.37% | 96.54% | 99.38% | **0.9793** | 0.9933 | 0.9954 | 0.9597 |
| `phiusiil` | 46,518 | **97.85%** | 99.97% | 94.95% | 99.98% | **0.9739** | 0.9987 | 0.9988 | 0.9566 |
| `phishbd_2026` | 26,227 | **94.16%** | 93.91% | 98.01% | 85.17% | **0.9592** | 0.9838 | 0.9926 | 0.8592 |
| `mendeley_url_phish` | 21,584 | **94.39%** | 62.06% | 93.59% | 94.47% | **0.7463** | 0.9853 | 0.9224 | 0.7353 |
| `dataset_phishing` | 2,280 | **81.67%** | 78.30% | 87.52% | 75.83% | **0.8266** | 0.9162 | 0.9228 | 0.6378 |

#### URL Confusion Matrix Counts ($n = 126,511$)
- **True Negatives (TN)**: `67,863` (96.27% of legitimate URLs)
- **True Positives (TP)**: `53,879` (96.18% of malicious URLs)
- **False Positives (FP)**: `2,630` (3.73% FPR)
- **False Negatives (FN)**: `2,139` (3.82% FNR)

---

### Email DistilBERT Performance ($n = 18,237$ Holdout Test Emails)

Trained and evaluated on the expanded **182,364-email corpus** (90,775 Legitimate / 91,589 Phishing, balanced 50/50 split):

- **Test Accuracy**: **`99.71%`** (Validation Accuracy: `99.64%`)
- **Precision**: **`99.65%`** (Ultra-low false positive rate)
- **Recall / Sensitivity**: **`99.78%`** (Intercepts 99.78% of active phishing attacks)
- **F1-Score**: **`0.9972`**
- **ROC-AUC**: **`0.9999`** | **PR-AUC**: **`0.9999`**
- **Hardware & Speed**: 9,120 steps trained in 60.2 minutes on **NVIDIA GeForce RTX 3050 Laptop GPU** (FP16 mixed precision)
- **Unseen External Dataset Test ($n = 1,000$)**: **`99.90% Accuracy`** | **`100.00% Precision`** | **`99.77% Recall`** | **`1.0000 ROC-AUC`**

---

### Multimodal Fusion Results ($n = 83,948$)

When evaluated across the full combined email dataset (where **50.7%** of messages contain active hyperlinks):
- **Phishing Recall**: **`100.00%`** (Zero phishing messages bypassed both detectors)
- **Overall Accuracy**: **`98.82%`**
- **Precision**: **`98.40%`**
- **F1-Score**: **`0.9919`**

---

## 🧠 Neural Network & Transformer Architectures

### 1. URL Lexical & Structural DNN

The URL classifier evaluates **49 static features** grouped into:
- **Lengths & Ratios**: URL length, hostname length, path length, query length, digit ratio, letter ratio, special character ratio, vowel ratio.
- **Delimiter & Character Counts**: Counts of dots, hyphens, underscores, slashes, `@`, `?`, `=`, `&`, `!`, `%`, `$`, `#`, `~`, digits, letters.
- **Structural Flags**: `hostname_is_ip`, `port_present`, `has_https`, `has_www`, `has_at_symbol`, `has_double_slash_redirect`, `has_shortening_service`, `has_suspicious_tld`, `has_punycode`, `http_token_in_path`.
- **Information Theory**: Shannon Entropy of full URL (`url_entropy`) and hostname (`hostname_entropy`), longest token length, average token length.

```text
Input (49 features)
  │
  ├── Dense(128, activation='relu', kernel_initializer='he_normal')
  ├── BatchNormalization()
  ├── Dropout(0.3)
  │
  ├── Dense(64, activation='relu', kernel_initializer='he_normal')
  ├── BatchNormalization()
  ├── Dropout(0.3)
  │
  ├── Dense(32, activation='relu')
  │
  └── Dense(1, activation='sigmoid')  ──> Output Phishing Probability P_url
```

* **Loss Function**: Binary Cross-Entropy
* **Optimizer**: Adam ($\eta = 10^{-3}$)
* **Regularization**: Batch Normalization, Dropout ($p=0.3$), `ReduceLROnPlateau` (factor $0.5$, patience $3$), `EarlyStopping` (patience $7$, restoring best weights on validation AUC).
* **Data Balancing**: SMOTE (Synthetic Minority Over-sampling Technique) applied to the training split.

---

### 2. Email Intent DistilBERT

* **Base Pretrained Model**: `distilbert-base-uncased` (6 transformer layers, 768 hidden dimension, 12 attention heads, 66M parameters).
* **Tokenization**: WordPiece tokenizer, maximum sequence length = 256 tokens.
* **Loss Function**: Cross-Entropy Loss with weighted class compensation.
* **Training Protocol**: Fine-tuned over 67,158 training samples with AdamW, linear learning rate warmup, and dynamic GPU batching.

---

### 3. Decision Fusion Engine

1. **Link Extraction**: Parses raw email text using high-coverage regex catching HTTP/HTTPS schemes, bare `www.` domains, and exotic TLDs.
2. **Text Inference**: Runs DistilBERT over `[Subject] + " " + [Body]` $\rightarrow P_{\text{email}}$.
3. **Link Inference**: For each extracted URL $i \in \{1, \dots, n\}$, extracts 49 features, applies saved `StandardScaler`, and executes the DNN $\rightarrow P_{\text{URL}_i}$.
4. **Bayesian Noisy-OR Combination**:
   $$P_{\text{combined}} = 1 - (1 - P_{\text{email}}) \times \prod_{i=1}^n (1 - P_{\text{URL}_i})$$
5. **High-Confidence Override Trigger**:
   $$\text{If } \max_{i} P_{\text{URL}_i} \ge 0.85 \implies \text{Force verdict to PHISHING with risk } \ge \max_{i} P_{\text{URL}_i}$$

---

## 📁 Project Directory Structure

```text
PhishingDeeplearning/
│
├── 🌐 app.py                          # Flask Web Server & REST API backend
├── 📜 main.py                         # Unified orchestrator & pipeline CLI
├── 🔀 merged_phishing_detector.py     # Multimodal dual-engine (DistilBERT + URL DNN)
│
├── 🧠 feature_extraction.py           # 49-feature offline URL feature extractor
├── ⚙️ preprocessing.py                # Train/test stratified split, scaling & SMOTE
├── 🏋️ train_dnn.py                    # URL DNN model architecture & training script
├── 📊 generate_results.py             # Evaluator generating multi-metric reports & curves
├── 🔍 explore_dataset.py              # URL dataset row/column/balance inspector
├── 📈 eda.py                          # Exploratory data analysis & feature correlations
├── 🎯 threshold_search_reweighted.py  # Optimal threshold search under varied prevalence
│
├── 🎨 static/                         # Web Dashboard frontend assets
│   ├── index.html                     # Responsive SOC cybersecurity interface
│   ├── css/
│   │   └── style.css                  # Dark-mode glassmorphic theme & animations
│   └── js/
│       └── app.js                     # Real-time UI logic, gauge rendering & REST client
│
├── 📧 emaildata/                      # Email dataset & DistilBERT Transformer subsystem
│   ├── combined_phishing_email_dataset.csv # 182,364 labeled email samples
│   ├── train_phishingemail.py         # PyTorch/HuggingFace DistilBERT training script
│   ├── preprocessing_smote.py         # Email data preparation & balancing
│   ├── explore_dataset email.py       # Email dataset exploration
│   └── results/
│       ├── best_distilbert/           # Fine-tuned PyTorch model weights & tokenizer
│       └── distilbert-base-uncased_random.json # Test metrics report
│
├── ⚙️ preprocessed/                   # Serialized models & processed arrays
│   ├── dnn_model.keras                # Trained Keras 3 URL Deep Neural Network
│   ├── scaler.joblib                  # Fitted StandardScaler for live inference
│   ├── train_smote.csv                # SMOTE-balanced training split
│   └── test_scaled.csv                # Scaled holdout test set (126,511 rows)
│
├── 📊 results/                        # Evaluation charts & summary files
│   ├── model_results_report.md        # Comprehensive evaluation report
│   ├── model_results_comprehensive.csv# Tabular metrics across all 5 benchmarks
│   ├── model_results_summary.txt      # Text summary & confusion matrices
│   ├── confusion_matrices.png         # Overall & per-dataset confusion matrix heatmaps
│   ├── training_history.png           # Loss, accuracy, AUC curves vs. epochs
│   └── roc_pr_curves.png              # ROC & Precision-Recall curves
│
├── 📦 requirements.txt                # Python package dependencies
├── 🚫 .gitignore                      # Git exclusion rules
└── 📜 README.md                       # Complete documentation & user guide
```

---

## 🚀 Installation & Quick Start

### 1. Clone and Set Up Virtual Environment

```powershell
# Open terminal inside the project directory
cd PhishingDeeplearning

# Create or activate Python virtual environment
python -m venv phishing_env
.\phishing_env\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
pip install torch transformers flask flask-cors
```

---

## 🖥️ Running the SOC Web Dashboard

Launch the modern Cybersecurity Operations Center web interface:

```powershell
python app.py
```

Then open your browser at:
👉 **`http://127.0.0.1:5000`**

### Features of the Web Interface:
1. **Interactive Scanner**:
   - **Email Mode**: Paste subject and body. The app highlights embedded links and runs both DistilBERT and URL DNN in parallel.
   - **URL Mode**: Directly inspect any web link against all 49 structural features.
2. **13 Pre-loaded Test Cases**:
   - 🚨 *Threats*: PayPal suspension, Microsoft 365 credential expiry, Amazon unauthorized purchase, IRS tax refund scam, Bank of America wire alert, DocuSign signature harvest, MetaMask Web3 wallet seed scam, Raw IP hosts, `@`-symbol credential injection.
   - 🛡️ *Legitimate*: Python Weekly newsletter, GitHub PR review, Stripe billing notice, corporate roadmap sync, Wikipedia, Python docs.
3. **Visual Analytics**:
   - Animated SVG Risk Gauge with needle transitions.
   - Dual-engine score contribution bar (Email NLP vs. URL DNN).
   - High-risk indicator pills (IP host, suspicious TLD, credential harvesting tokens).
   - SOC Guidance & Incident Response checklist.

---

## 📡 REST API Reference

The Flask backend exposes full JSON endpoints for external integration:

### 1. Health Check
```http
GET /api/health
```
**Response:**
```json
{
  "status": "online",
  "model": "Merged Phishing Detector (DistilBERT + URL DNN)"
}
```

---

### 2. Predict Email Phishing Risk
```http
POST /api/predict/email
Content-Type: application/json
```
**Request Body:**
```json
{
  "subject": "Action Required: Microsoft 365 Password Expiry",
  "body": "Please confirm your password immediately at http://m365-portal-verify.xyz/login"
}
```
**Response:**
```json
{
  "verdict": "PHISHING",
  "risk_level": "CRITICAL",
  "combined_phishing_prob": 0.9998,
  "email_prob": 0.9850,
  "urls_detected": 1,
  "urls": [
    {
      "url": "http://m365-portal-verify.xyz/login",
      "phishing_prob": 0.9995,
      "verdict": "PHISHING"
    }
  ],
  "override_triggered": true,
  "threat_contributors": ["High-risk URL override", "Urgent credential harvesting text"]
}
```

---

### 3. Predict Single URL Risk
```http
POST /api/predict/url
Content-Type: application/json
```
**Request Body:**
```json
{
  "url": "http://secure-paypal-login-verify.xyz/auth/login.php"
}
```
**Response:**
```json
{
  "url": "http://secure-paypal-login-verify.xyz/auth/login.php",
  "verdict": "PHISHING",
  "risk_level": "CRITICAL",
  "combined_phishing_prob": 0.9984,
  "key_features": {
    "url_length": 52,
    "has_suspicious_tld": 1,
    "has_https": 0,
    "suspicious_keyword_count": 3
  }
}
```

---

### 4. Fetch Benchmark Metrics
```http
GET /api/benchmarks
```
Returns accuracy, F1, precision, recall, and sample counts across all three models.

---

### 5. Fetch Sample Test Cases
```http
GET /api/samples
```
Returns the catalog of 13 built-in demonstration cases.

---

## 💻 CLI & Pipeline Execution (`main.py`)

The unified orchestrator `main.py` provides command-line control over every pipeline stage:

```powershell
# 1. Dataset exploration (shapes, balance, missing values)
python main.py --explore

# 2. Extract 49 features from raw dataset URLs
python main.py --extract

# 3. Exploratory Data Analysis & correlation heatmaps
python main.py --eda

# 4. Preprocess: Train/Test Stratified Split -> StandardScaler -> SMOTE
python main.py --preprocess

# 5. Train the URL Deep Neural Network
python main.py --train

# 6. Evaluate URL DNN & generate full report/plots
python main.py --evaluate

# 7. Predict phishing risk for a single URL
python main.py --predict "http://secure-bank-login-update.top/auth"

# 8. Predict phishing risk for an email (Subject + Body)
python main.py --predict-email "Urgent Account Alert" "Click here to verify: http://paypal-verify.xyz"

# 9. Run interactive multimodal CLI demo
python main.py --merged

# 10. Run multimodal evaluation on test dataset
python main.py --merged-eval

# 11. Run complete pipeline end-to-end
python main.py --all
```

---

## ⚙️ Hardware & Optimization

- **GPU Acceleration**: Uses PyTorch with native CUDA acceleration for DistilBERT and TensorFlow with GPU/DirectML support for the Keras DNN.
- **Tested Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM) + Intel Core i5/i7.
- **CPU Fallback**: Both models feature automatic CPU fallback when CUDA is not present.
- **Latency**:
  - URL 49-feature extraction + DNN inference: **< 2.5 ms** per URL.
  - DistilBERT inference on GPU: **< 8 ms** per email (batch throughput > 570 samples/sec).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
