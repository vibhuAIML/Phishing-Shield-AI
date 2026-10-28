"""
merged_phishing_detector.py
===========================
Unified Multimodal Phishing Detection Engine.

Merges both trained models:
  1. train_phishingemail: Fine-tuned DistilBERT (Transformer for email subject & body text semantics)
  2. train_dnn: Deep Neural Network (49 lexical/structural features for embedded URLs)

Capabilities:
  • Single URL threat scoring (DNN)
  • Single Email text threat scoring (DistilBERT)
  • Full Multimodal Email Phishing Detection (DistilBERT + URL extraction + DNN threat fusion)
  • Comparative benchmark evaluation on the phishing dataset
"""

import os
import re
import sys
import argparse
from typing import Dict, List, Any, Optional
from urllib.parse import urlsplit

import numpy as np
import joblib
import torch
import keras
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Import existing offline lexical URL feature extractor
try:
    import feature_extraction
except ImportError:
    from . import feature_extraction

# URL regex pattern matching HTTP(S), www, and common domain structures
URL_REGEX = re.compile(
    r"(?:https?://|www\.)[^\s<>'\"`]+|[a-zA-Z0-9.-]+\.(?:com|org|net|xyz|top|ru|cn|work|biz|info|cc|tk|ga|cf|gq|club|online|site|vip|loan|racing)[^\s<>'\"`]*",
    re.IGNORECASE
)

# Ordered feature columns exactly matching scaler and DNN input
URL_FEATURE_COLS = [
    'url_length', 'hostname_length', 'path_length', 'query_length', 'fragment_length',
    'count_dots', 'count_hyphens', 'count_underscore', 'count_slash', 'count_question',
    'count_equal', 'count_at', 'count_and', 'count_exclamation', 'count_space',
    'count_tilde', 'count_comma', 'count_plus', 'count_asterisk', 'count_hash',
    'count_dollar', 'count_percent', 'count_semicolon', 'count_digits', 'count_letters',
    'digit_ratio', 'letter_ratio', 'special_char_ratio', 'vowel_ratio', 'num_subdomains',
    'hostname_is_ip', 'port_present', 'tld_length', 'domain_length', 'digits_in_domain',
    'hyphen_in_domain', 'has_https', 'has_www', 'has_at_symbol', 'has_double_slash_redirect',
    'has_shortening_service', 'has_suspicious_tld', 'has_punycode', 'http_token_in_path',
    'suspicious_keyword_count', 'url_entropy', 'hostname_entropy', 'longest_word_length',
    'avg_word_length'
]


class MergedPhishingDetector:
    """
    Unified dual-engine detector combining:
      - DistilBERT for Email Natural Language Intent
      - Deep Neural Network for URL Structural & Lexical Threats
    """

    def __init__(
        self,
        dnn_path: str = "preprocessed/dnn_model.keras",
        scaler_path: str = "preprocessed/scaler.joblib",
        distilbert_path: str = "emaildata/results/best_distilbert",
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Resolve paths relative to project root if needed
        dnn_file = os.path.join(base_dir, dnn_path) if not os.path.isabs(dnn_path) else dnn_path
        scaler_file = os.path.join(base_dir, scaler_path) if not os.path.isabs(scaler_path) else scaler_path
        distilbert_dir = os.path.join(base_dir, distilbert_path) if not os.path.isabs(distilbert_path) else distilbert_path

        print("[*] Initializing Merged Phishing Detector...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  • Hardware Backend: {self.device.type.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

        # 1. Load URL DNN & Scaler
        print("  • Loading URL Phishing DNN model & feature scaler...")
        self.dnn_model = keras.models.load_model(dnn_file)
        self.scaler = joblib.load(scaler_file)

        # 2. Load Email DistilBERT
        print("  • Loading Email Phishing DistilBERT Transformer...")
        self.tokenizer = AutoTokenizer.from_pretrained(distilbert_dir)
        self.email_model = AutoModelForSequenceClassification.from_pretrained(distilbert_dir)
        self.email_model.to(self.device)
        self.email_model.eval()

        print("[✓] Both models loaded and ready for dual-engine inference.\n")

    def extract_urls(self, text: str) -> List[str]:
        """Extract and clean URLs embedded in email subject/body text."""
        if not text or not isinstance(text, str):
            return []
        matches = URL_REGEX.findall(text)
        cleaned = []
        for m in matches:
            clean_url = m.strip(".,;:!?)>\"'[]{}<>\n\r\t`")
            if len(clean_url) > 3 and "." in clean_url and clean_url not in cleaned:
                cleaned.append(clean_url)
        return cleaned

    TRUSTED_DOMAINS = {
        "google.com", "github.com", "wikipedia.org", "python.org",
        "microsoft.com", "apple.com", "stripe.com", "amazon.com",
        "linkedin.com", "youtube.com", "kernel.org", "w3.org", "apache.org",
        "nih.gov", "stackoverflow.com", "medium.com", "cloudflare.com"
    }

    def predict_url(self, url: str) -> Dict[str, Any]:
        """
        Analyze a URL using feature extraction + URL DNN.
        Returns phishing probability, verdict, risk level, and key signals.
        """
        try:
            raw_feats = feature_extraction.extract_features(url)
            vector = np.array([raw_feats.get(c, 0.0) for c in URL_FEATURE_COLS], dtype=np.float32).reshape(1, -1)
            scaled_vector = self.scaler.transform(vector)
            prob = float(self.dnn_model.predict(scaled_vector, verbose=0).ravel()[0])
        except Exception:
            prob = 0.5
            raw_feats = {}

        signals = []
        if raw_feats.get("hostname_is_ip", 0):
            signals.append("IP address used as hostname")
        if raw_feats.get("has_suspicious_tld", 0):
            signals.append("High-risk phishing Top-Level Domain (TLD)")
        if raw_feats.get("has_shortening_service", 0):
            signals.append("URL shortening / obfuscation service detected")
        if raw_feats.get("suspicious_keyword_count", 0) > 0:
            signals.append(f"{raw_feats['suspicious_keyword_count']} security/credential lure keyword(s)")
        if raw_feats.get("has_at_symbol", 0):
            signals.append("@ symbol embedded in URL (credential injection trick)")
        if raw_feats.get("url_entropy", 0.0) > 4.5:
            signals.append(f"High Shannon entropy ({raw_feats['url_entropy']:.2f}) indicating randomized path/token")
        if not raw_feats.get("has_https", 1):
            signals.append("Plain unencrypted HTTP connection")

        # Global Authority Domain Reputation Check
        try:
            full = url if "://" in url else "http://" + url
            netloc = urlsplit(full).netloc.lower().split(":")[0]
            is_trusted = any(netloc == t or netloc.endswith("." + t) for t in self.TRUSTED_DOMAINS)
            has_spoof = bool(raw_feats.get("has_suspicious_tld", 0) or raw_feats.get("hostname_is_ip", 0) or raw_feats.get("has_at_symbol", 0))
            if is_trusted and not has_spoof:
                prob = min(prob * 0.08, 0.12)
                signals.append("Verified high-reputation domain")
        except Exception:
            pass

        if prob >= 0.85:
            risk_level = "CRITICAL"
        elif prob >= 0.65:
            risk_level = "HIGH"
        elif prob >= 0.50:
            risk_level = "MODERATE"
        elif prob >= 0.25:
            risk_level = "LOW"
        else:
            risk_level = "SAFE"

        return {
            "url": url,
            "phishing_prob": round(prob, 4),
            "verdict": "PHISHING" if prob >= 0.5 else "LEGITIMATE",
            "risk_level": risk_level,
            "key_signals": signals,
        }

    def predict_email_text(self, subject: str, body: str) -> Dict[str, Any]:
        """
        Analyze email subject + body text semantics using DistilBERT.
        """
        subject = subject or ""
        body = body or ""
        text = (subject + " [SEP] " + body[:3000]).strip()

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding=True,
            return_token_type_ids=False,
            return_tensors="pt"
        ).to(self.device)
        inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

        with torch.no_grad():
            outputs = self.email_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        phish_prob = float(probs[1])
        return {
            "phishing_prob": round(phish_prob, 4),
            "verdict": "PHISHING" if phish_prob >= 0.5 else "LEGITIMATE",
            "legit_prob": round(float(probs[0]), 4),
        }

    def predict(
        self,
        subject: str = "",
        body: str = "",
        url: Optional[str] = None,
        headers: Optional[Any] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Unified Multimodal Phishing Evaluation.
        Merges DistilBERT (Email Language), DNN (URL Security),
        Email Header Authentication (SPF/DKIM/DMARC), and Quishing Attachment Scanning.
        """
        # If user directly queries a single standalone URL
        if url and not subject and not body and not headers and not attachments:
            url_res = self.predict_url(url)
            return {
                "mode": "URL_ONLY",
                "verdict": url_res["verdict"],
                "combined_phishing_prob": url_res["phishing_prob"],
                "risk_level": url_res["risk_level"],
                "url_analysis": [url_res],
                "explanation": f"Evaluated via URL Deep Neural Network: {url_res['verdict']} ({url_res['phishing_prob']*100:.1f}%)"
            }

        reasons = []

        # 1. Analyze Email Text with DistilBERT
        text_res = self.predict_email_text(subject, body)
        p_email = text_res["phishing_prob"]

        # 2. Extract and analyze all embedded URLs from text
        combined_text = (subject or "") + " " + (body or "")
        detected_urls = self.extract_urls(combined_text)
        if url and url not in detected_urls:
            detected_urls.insert(0, url)

        # 3. Optional Attachment & Quishing QR scanning
        attachment_reports = []
        if attachments:
            try:
                from attachment_scanner import AttachmentScanner
                attach_scanner = AttachmentScanner()
                for att in attachments:
                    fname = att.get("filename", "attachment")
                    fdata = att.get("data")
                    att_rep = attach_scanner.scan_attachment(fname, fdata)
                    attachment_reports.append(att_rep)
                    for discovered_url in att_rep.get("extracted_urls", []):
                        if discovered_url not in detected_urls:
                            detected_urls.append(discovered_url)
                    for flag in att_rep.get("threat_flags", []):
                        reasons.append(f"[Attachment Threat] {flag}")
            except Exception as e:
                print(f"[!] Attachment scanning warning: {e}")

        # 4. Optional Header Authentication & Spoofing Inspection
        header_report = None
        p_header = 0.0
        if headers:
            try:
                from header_auth_scanner import HeaderAuthScanner
                hdr_scanner = HeaderAuthScanner()
                header_report = hdr_scanner.scan(headers)
                p_header = header_report.get("header_risk_score", 0.0)
                for anom in header_report.get("anomalies", []):
                    reasons.append(f"[Header Security] {anom}")
            except Exception as e:
                print(f"[!] Header inspection warning: {e}")

        url_analyses = [self.predict_url(u) for u in detected_urls]

        # 5. Multimodal Decision Fusion Engine
        if not url_analyses:
            p_combined = p_email
            p_url_max = 0.0
            reasons.append(f"Classified purely via Natural Language Intent (DistilBERT: {p_email*100:.1f}% phishing probability).")
        else:
            p_url_max = max(u["phishing_prob"] for u in url_analyses)

            # Calibrated Multimodal Decision Fusion:
            # Case 1: URL is definitively malicious (>= 0.85) -> immediate override to high threat
            if p_url_max >= 0.85:
                p_combined = max(p_email, p_url_max)
            # Case 2: Email text is strongly benign (< 0.20) and URL is only mildly suspicious (< 0.85)
            elif p_email < 0.20:
                p_combined = 0.85 * p_email + 0.15 * p_url_max
            # Case 3: Email text exhibits phishing/social engineering or ambiguous signals
            else:
                p_combined = 1.0 - (1.0 - p_email) * (1.0 - p_url_max)

            if p_email >= 0.5:
                reasons.append(f"Email text exhibits phishing/social-engineering signals (DistilBERT: {p_email*100:.1f}%).")
            else:
                reasons.append(f"Email text appears conversational/benign (DistilBERT: {p_email*100:.1f}%).")

            if p_url_max >= 0.65:
                worst_url = max(url_analyses, key=lambda x: x["phishing_prob"])
                reasons.append(f"High-threat link detected: '{worst_url['url']}' (DNN risk: {p_url_max*100:.1f}%).")
                if worst_url["key_signals"]:
                    reasons.append(f"Signals: {', '.join(worst_url['key_signals'])}")
            elif detected_urls:
                reasons.append(f"{len(detected_urls)} link(s) inspected; all scored low-risk by DNN.")

        # Boost risk if header spoofing or critical authentication failures are confirmed
        if p_header >= 0.50:
            p_combined = max(p_combined, p_header)
            reasons.append(f"Authentication failure: Sender spoofing / SPF / DKIM verification failed ({p_header*100:.1f}% risk).")

        # Determine final verdict and overall threat severity
        verdict = "PHISHING" if p_combined >= 0.5 else "LEGITIMATE"
        if p_combined >= 0.85:
            risk_level = "CRITICAL"
        elif p_combined >= 0.65:
            risk_level = "HIGH"
        elif p_combined >= 0.50:
            risk_level = "MODERATE"
        elif p_combined >= 0.20:
            risk_level = "LOW"
        else:
            risk_level = "SAFE"

        return {
            "mode": "MULTIMODAL_FUSION",
            "verdict": verdict,
            "combined_phishing_prob": round(float(p_combined), 4),
            "risk_level": risk_level,
            "email_text_phishing_prob": p_email,
            "max_url_phishing_prob": round(float(p_url_max), 4) if url_analyses else None,
            "urls_detected_count": len(url_analyses),
            "urls_analysis": url_analyses,
            "header_security": header_report,
            "attachment_analysis": attachment_reports,
            "explanation": " ".join(reasons)
        }

    def evaluate_benchmark(self, data_path: str = "emaildata/combined_phishing_email_dataset.csv", n_samples: int = 500):
        """
        Evaluates and compares:
          1. DistilBERT Email-Only
          2. DNN URL-Only (on emails containing URLs)
          3. Merged Dual-Engine (DistilBERT + URL DNN)
        """
        import pandas as pd
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file = os.path.join(base_dir, data_path) if not os.path.isabs(data_path) else data_path

        if not os.path.exists(csv_file):
            print(f"[!] Evaluation file '{csv_file}' not found.")
            return

        print(f"\n[*] Running Benchmark Evaluation on {n_samples} samples from '{os.path.basename(csv_file)}'...")
        df = pd.read_csv(csv_file, low_memory=False)
        sample = df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(drop=True)

        y_true = sample["label"].values.astype(int)
        preds_email = []
        preds_url = []
        preds_merged = []
        probs_merged = []
        has_url_flags = []

        for idx, (_, row) in enumerate(sample.iterrows()):
            if (idx + 1) % 50 == 0 or idx == len(sample) - 1:
                print(f"  • Scoring samples: [{idx + 1}/{len(sample)}]...", end="\r", flush=True)
            subj = str(row.get("subject", "") or "")
            body = str(row.get("body", "") or "")
            res = self.predict(subject=subj, body=body)

            p_comb = res["combined_phishing_prob"]
            p_txt = res["email_text_phishing_prob"]
            p_url = res["max_url_phishing_prob"]

            preds_merged.append(int(p_comb >= 0.5))
            probs_merged.append(p_comb)
            preds_email.append(int(p_txt >= 0.5))

            if p_url is not None:
                has_url_flags.append(True)
                preds_url.append(int(p_url >= 0.5))
            else:
                has_url_flags.append(False)
                preds_url.append(0)

        # Compute Metrics
        def get_metrics(yt, yp, proba=None):
            acc = accuracy_score(yt, yp)
            prec = precision_score(yt, yp, zero_division=0)
            rec = recall_score(yt, yp, zero_division=0)
            f1 = f1_score(yt, yp, zero_division=0)
            auc = roc_auc_score(yt, proba) if proba is not None and len(set(yt)) > 1 else None
            return acc, prec, rec, f1, auc

        acc_m, prec_m, rec_m, f1_m, auc_m = get_metrics(y_true, preds_merged, probs_merged)
        acc_e, prec_e, rec_e, f1_e, _ = get_metrics(y_true, preds_email)

        print("\n" + "=" * 76)
        print("                 BENCHMARK PERFORMANCE COMPARISON")
        print("=" * 76)
        print(f"{'Model Architecture':<32} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}")
        print("-" * 76)
        print(f"{'1. Email-Only (DistilBERT)':<32} | {acc_e*100:>7.2f}% | {prec_e*100:>7.2f}% | {rec_e*100:>7.2f}% | {f1_e:>9.4f}")
        print(f"{'2. Merged Dual-Engine (Fusion)':<32} | {acc_m*100:>7.2f}% | {prec_m*100:>7.2f}% | {rec_m*100:>7.2f}% | {f1_m:>9.4f}")
        print("=" * 76)
        if auc_m:
            print(f"  • Merged Model ROC-AUC : {auc_m:.4f}")
        print(f"  • Emails with URLs     : {sum(has_url_flags)} / {len(sample)} ({sum(has_url_flags)/len(sample)*100:.1f}%)")
        print(f"  • Test Samples Scored  : {len(sample)}")
        print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="Merged Phishing Detector (DistilBERT + URL DNN)")
    parser.add_argument("--url", type=str, help="Single URL to inspect via DNN")
    parser.add_argument("--subject", type=str, default="", help="Email Subject")
    parser.add_argument("--body", type=str, default="", help="Email Body text")
    parser.add_argument("--file", type=str, help="Text file containing email content")
    parser.add_argument("--evaluate", action="store_true", help="Run comparative benchmark evaluation")
    parser.add_argument("--samples", type=int, default=300, help="Number of benchmark samples to evaluate")

    args = parser.parse_args()
    detector = MergedPhishingDetector()

    if args.evaluate:
        detector.evaluate_benchmark(n_samples=args.samples)
        return

    if args.file and os.path.exists(args.file):
        with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        res = detector.predict(subject="Email from File", body=content)
    elif args.subject or args.body:
        res = detector.predict(subject=args.subject, body=args.body, url=args.url)
    elif args.url:
        res = detector.predict(url=args.url)
    else:
        # Default interactive demo example
        print("\n[i] No arguments provided. Running demonstration on a representative phishing sample...")
        sample_subject = "URGENT: Your PayPal Account Has Been Suspended"
        sample_body = (
            "Dear Customer, We detected unusual login attempts to your account. "
            "Please verify your credentials immediately to avoid permanent suspension: "
            "http://secure-paypal-login-verify.xyz/auth/login.php. Thank you, Security Team."
        )
        res = detector.predict(subject=sample_subject, body=sample_body)

    # Print Formatted Result
    print("\n" + "=" * 70)
    print("                     PHISHING DETECTION REPORT")
    print("=" * 70)
    verdict_badge = "🚨 [PHISHING ATTACK DETECTED]" if res["verdict"] == "PHISHING" else "🛡️ [LEGITIMATE / SAFE]"
    print(f"  VERDICT         : {verdict_badge}")
    print(f"  COMBINED RISK   : {res['combined_phishing_prob'] * 100:.1f}% ({res['risk_level']})")

    if "email_text_phishing_prob" in res and res["email_text_phishing_prob"] is not None:
        print(f"  EMAIL NLP SCORE : {res['email_text_phishing_prob'] * 100:.1f}% (DistilBERT)")

    if res.get("urls_detected_count", 0) > 0:
        print(f"  URLS INSPECTED  : {res['urls_detected_count']} link(s) found")
        for i, u_res in enumerate(res["urls_analysis"], 1):
            print(f"\n    [Link #{i}] {u_res['url']}")
            print(f"      • Threat Score : {u_res['phishing_prob'] * 100:.1f}% ({u_res['verdict']} - {u_res['risk_level']})")
            if u_res["key_signals"]:
                print(f"      • Signals      : {', '.join(u_res['key_signals'])}")
    else:
        print("  URLS INSPECTED  : None found in message")

    print("\n  EXPLANATION     :")
    print(f"  {res['explanation']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
