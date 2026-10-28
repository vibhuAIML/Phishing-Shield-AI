# 🚀 How to Run the Multimodal Phishing Detection System

This guide provides step-by-step instructions to run the **Cybersecurity Operations Center (SOC) Web Dashboard**, execute **CLI predictions**, run **benchmark evaluations**, and replicate the **data preprocessing and model training pipeline**.

---

## 📋 Table of Contents
1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Quick Start: Launching the SOC Web Dashboard](#2-quick-start-launching-the-soc-web-dashboard)
3. [Running Dual-Engine Predictions via CLI](#3-running-dual-engine-predictions-via-cli)
4. [Running Benchmark Evaluations](#4-running-benchmark-evaluations)
5. [Using the REST API](#5-using-the-rest-api)
6. [Data Preprocessing & Model Training Workflow](#6-data-preprocessing--model-training-workflow)

---

## 1. Prerequisites & Environment Setup

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.10+ (Tested on Python 3.10 / 3.11 / 3.12 / 3.14)
- **Hardware Acceleration (Recommended)**: NVIDIA GPU with CUDA (e.g. RTX 3050 Laptop GPU / RTX 3060+) or CPU fallback.

### Activate the Project Virtual Environment

Open PowerShell or Command Prompt inside the `PhishingDeeplearning` directory:

```powershell
# Navigate to project directory
cd C:\Users\Vibhu-Kavi\Downloads\PhishingDeeplearning

# Activate existing virtual environment
.\phishing_env\Scripts\Activate.ps1
```

*(If you are on Linux / macOS, use: `source phishing_env/bin/activate`)*

---

## 2. Quick Start: Launching the SOC Web Dashboard

Start the integrated Flask web server hosting both the REST API and the interactive dark-mode SOC dashboard:

```powershell
& ".\phishing_env\Scripts\python.exe" app.py
```

### Accessing the Web Interface
Open your web browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### Dashboard Features:
- **Real-Time Dual-Engine Inference**: Type or paste any email subject, body text, or standalone URL.
- **Embedded Link Detection**: Automatically parses hyperlinks inside the email body and feeds them to the 49-feature URL DNN.
- **Visual Threat Breakdown**: Live animated gauge meters, severity tags (`SAFE`, `LOW`, `MODERATE`, `HIGH`, `CRITICAL`), and feature-level explanations.
- **13 Interactive Preloaded Scenarios**: Click any sample (PayPal suspension, M365 password expiry, Amazon order fraud, benign company newsletters) to audition detection capabilities instantly.
- **5 SOC Modules**:
  1. *Email Threat Analyzer* (with 1-click Human-in-the-Loop Analyst Triage)
  2. *URL Threat Scanner* (49-feature extraction profile)
  3. *Quishing & Attachments* (Image QR code decoder & PDF scanner)
  4. *Header Security* (SPF/DKIM/DMARC & display-name brand spoofing)
  5. *Enterprise & Retraining* (Live M365 & Gmail mailbox scanner + continuous retraining triggers)

---

## 3. Running CLI Commands & Pipeline Execution (`main.py`)

You can execute end-to-end scanning, pipeline stages, or mailbox audits directly via `main.py`:

### A. Full Email Multimodal Scan (Subject + Body + Embedded Links)
```powershell
& ".\phishing_env\Scripts\python.exe" main.py --predict-email `
    "Urgent: Your account is locked" `
    "Please verify your credentials immediately at http://login-secure-banking-update.xyz to prevent suspension."
```

### B. Standalone URL Inspection (49-Feature URL DNN)
```powershell
& ".\phishing_env\Scripts\python.exe" main.py --predict "http://paypal-verification-account-login.top/auth/index.php"
```

### C. Enterprise Mailbox Gateway Scan & Remediation
Scans Microsoft 365 and Google Workspace inboxes, executing automated quarantine on threats:
```powershell
& ".\phishing_env\Scripts\python.exe" main.py --gateway-sync
```

### D. Scan Email Attachment / Quishing QR Flyer or PDF
```powershell
& ".\phishing_env\Scripts\python.exe" main.py --scan-attachment "invoice.pdf"
```

### E. Audit Raw Email Headers (SPF / DKIM / DMARC / Spoofing)
```powershell
& ".\phishing_env\Scripts\python.exe" main.py --scan-headers "headers_sample.txt"
```

### F. Trigger Incremental Retraining on Analyst Feedback
```powershell
& ".\phishing_env\Scripts\python.exe" main.py --retrain-feedback
```

---

## 4. Running the Automated Test Suite

Run the comprehensive end-to-end test suite verifying all 6 security subsystems:

```powershell
& ".\phishing_env\Scripts\python.exe" test_suite_all.py
```

*Verifies: HeaderAuthScanner, Attachment/Quishing Scanner, Enterprise Connectors (M365 & Gmail), Feedback Retraining Database, Multimodal Decision Fusion, and all Flask REST endpoints.*

---

## 5. Using the REST API

When `app.py` is running, the REST API endpoints are exposed on `http://127.0.0.1:5000`:

### System Status & Health
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/system/status" -Method GET
```

### Analyze Email
```powershell
$payload = @{
    subject = "Account Password Expiration Notice"
    body    = "Your session will expire in 1 hour. Log in now at http://portal-update-m365.xyz"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict/email" -Method POST -Body $payload -ContentType "application/json"
```

### Analyze URL
```powershell
$payload = @{
    url = "http://login-account-verification.xyz/auth"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/predict/url" -Method POST -Body $payload -ContentType "application/json"
```

### Audit Email Headers (SPF / DKIM / DMARC / Spoofing)
```powershell
$payload = @{
    headers = "From: `"PayPal Security`" <service@paypal-fake-alert.xyz>`nAuthentication-Results: spf=fail; dkim=fail; dmarc=fail"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/scan/headers" -Method POST -Body $payload -ContentType "application/json"
```

### Trigger Enterprise Gateway Mailbox Scan (M365 & Gmail)
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/gateway/scan" -Method GET
```

### Submit Analyst Triage Feedback
```powershell
$payload = @{
    sample_type      = "email"
    target_content   = "Subject: Fake invoice alert..."
    predicted_prob   = 0.99
    predicted_verdict= "PHISHING"
    analyst_verdict  = "PHISHING"
    analyst_notes    = "Confirmed credential harvesting attack"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/feedback" -Method POST -Body $payload -ContentType "application/json"
```

### Trigger Continuous Retraining
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/retrain/trigger" -Method POST
```

---

## 6. Docker & Celery / Redis Containerized Deployment

The system is fully containerized using **Docker** and **Docker Compose** for production microservice deployment:

### Multi-Container Topology:
1. `phishing_redis`: In-memory message broker (`redis:7-alpine`) on port `6379`.
2. `phishing_soc_web`: Gunicorn WSGI production web server serving the dashboard & REST API on port `5000`.
3. `phishing_celery_worker`: Distributed task worker consuming asynchronous email scans and gateway sync jobs.

### Running via Docker Compose:
> **Note**: Ensure Docker Desktop is installed and running before executing compose commands.

```bash
docker compose up --build
```

To run in the background (detached mode):
```bash
docker compose up -d
```

To stop containers:
```bash
docker compose down
```

---

## 7. Data Preprocessing & Model Training Workflow

If you ever wish to re-merge new data, re-run SMOTE, or retrain the Transformer:

### Step 1: Merge Additional Datasets
```powershell
& ".\phishing_env\Scripts\python.exe" emaildata/merge_datasets.py
```
*(Produces `emaildata/combined_phishing_email_dataset.csv` with 182,364 deduplicated rows).*

### Step 2: Exploratory Data Analysis (EDA)
```powershell
& ".\phishing_env\Scripts\python.exe" "emaildata/explore_dataset email.py" `
    --data "emaildata/combined_phishing_email_dataset.csv" `
    --out "emaildata/eda_output"
```

### Step 3: Preprocessing & SMOTE Balancing
```powershell
& ".\phishing_env\Scripts\python.exe" emaildata/preprocessing_smote.py `
    --input "emaildata/combined_phishing_email_dataset.csv" `
    --out "emaildata/output" `
    --max-features 30000 `
    --k 5
```

### Step 4: Retrain the Email DistilBERT Transformer (GPU)
```powershell
& ".\phishing_env\Scripts\python.exe" emaildata/train_phishingemail.py `
    --data "emaildata/combined_phishing_email_dataset.csv" `
    --model distilbert-base-uncased `
    --epochs 2 `
    --bs 16 `
    --accum 2 `
    --grad_ckpt `
    --out "emaildata/results"
```
*(Automatically saves the fine-tuned model and tokenizer to `emaildata/results/best_distilbert/` for instant deployment).*

---

### 🎉 All Systems Ready!
Launch `app.py` anytime to run the complete defense system.
