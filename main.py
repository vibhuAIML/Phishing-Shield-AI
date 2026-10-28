"""
main.py
=======
Unified Pipeline Orchestrator & CLI for URL Phishing Deep Learning Project.

Usage:
  python main.py --explore       # Show dataset overview & class balance
  python main.py --eda           # Run exploratory data analysis & correlations
  python main.py --extract       # Extract 49 numeric features from raw URLs
  python main.py --preprocess    # Stratified split, scale & SMOTE balance
  python main.py --train         # Train Deep Neural Network with callbacks & plots
  python main.py --evaluate      # Generate comprehensive F1, ROC, & metric reports
  python main.py --predict <URL> # Predict phishing probability for a live URL
  python main.py --all           # Run full pipeline end-to-end
"""

import sys
import subprocess
import argparse


def run_step(script_name, description):
    print("\n" + "=" * 70)
    print(f"  STEP: {description}")
    print(f"  RUNNING: python {script_name}")
    print("=" * 70)
    cmd = [sys.executable, script_name]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n[Error] {script_name} failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="URL Phishing Detection Deep Learning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--explore", action="store_true", help="Inspect dataset shapes and class balance")
    parser.add_argument("--extract", action="store_true", help="Extract lexical/structural features from raw dataset")
    parser.add_argument("--eda", action="store_true", help="Run exploratory data analysis")
    parser.add_argument("--preprocess", action="store_true", help="Split, scale, and SMOTE-balance data")
    parser.add_argument("--train", action="store_true", help="Train the Deep Neural Network")
    parser.add_argument("--evaluate", action="store_true", help="Generate comprehensive evaluation reports")
    parser.add_argument("--predict", type=str, metavar="URL", help="Predict phishing risk for a URL (via URL DNN)")
    parser.add_argument("--predict-email", nargs=2, metavar=("SUBJECT", "BODY"), help="Predict email phishing risk via Merged Detector")
    parser.add_argument("--scan-attachment", type=str, metavar="FILE", help="Scan image/QR flyer or PDF attachment for threats")
    parser.add_argument("--scan-headers", type=str, metavar="HEADER_FILE", help="Audit email headers from file for SPF/DKIM/DMARC/spoofing")
    parser.add_argument("--gateway-sync", action="store_true", help="Execute Microsoft 365 and Google Workspace mailbox scan")
    parser.add_argument("--retrain-feedback", action="store_true", help="Trigger incremental retraining on logged analyst feedback")
    parser.add_argument("--merged", action="store_true", help="Run merged multimodal phishing detection demo")
    parser.add_argument("--merged-eval", action="store_true", help="Run merged model benchmark evaluation on dataset")
    parser.add_argument("--all", action="store_true", help="Execute complete pipeline end-to-end")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        print("\n>> Quick Start: Run 'python main.py --explore' or 'python main.py --merged'")
        return

    if args.explore:
        run_step("explore_dataset.py", "Dataset Exploration")

    elif args.extract:
        run_step("feature_extraction.py", "Lexical Feature Extraction")

    elif args.eda:
        run_step("eda.py", "Exploratory Data Analysis")

    elif args.preprocess:
        run_step("preprocessing.py", "Preprocessing (Split, Scale, SMOTE)")

    elif args.train:
        run_step("train_dnn.py", "DNN Model Training")

    elif args.evaluate:
        run_step("generate_results.py", "Comprehensive Performance Evaluation")

    elif args.predict:
        from merged_phishing_detector import MergedPhishingDetector
        detector = MergedPhishingDetector()
        res = detector.predict(url=args.predict)
        print("\n" + "=" * 60)
        print(f"URL: {args.predict}")
        print(f"VERDICT: {res['verdict']} | RISK: {res['combined_phishing_prob']*100:.1f}% ({res['risk_level']})")
        print("=" * 60)

    elif args.predict_email:
        from merged_phishing_detector import main as run_detector_cli
        sys.argv = ["merged_phishing_detector.py", "--subject", args.predict_email[0], "--body", args.predict_email[1]]
        run_detector_cli()

    elif args.scan_attachment:
        from attachment_scanner import AttachmentScanner
        scanner = AttachmentScanner()
        res = scanner.scan_attachment(args.scan_attachment, args.scan_attachment)
        import json
        print("\n" + "=" * 60)
        print("ATTACHMENT & QUISHING SCAN RESULT:")
        print(json.dumps(res, indent=2))
        print("=" * 60)

    elif args.scan_headers:
        from header_auth_scanner import HeaderAuthScanner
        scanner = HeaderAuthScanner()
        with open(args.scan_headers, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        res = scanner.scan(content)
        import json
        print("\n" + "=" * 60)
        print("HEADER SECURITY AUDIT RESULT:")
        print(json.dumps(res, indent=2))
        print("=" * 60)

    elif args.gateway_sync:
        from enterprise_connectors import EnterpriseGatewayManager
        from merged_phishing_detector import MergedPhishingDetector
        detector = MergedPhishingDetector()
        mgr = EnterpriseGatewayManager()
        res = mgr.run_gateway_inspection(detector)
        import json
        print("\n" + "=" * 60)
        print("ENTERPRISE GATEWAY SCAN RESULT:")
        print(json.dumps(res, indent=2))
        print("=" * 60)

    elif args.retrain_feedback:
        from feedback_retraining import FeedbackRetrainingManager
        mgr = FeedbackRetrainingManager()
        res = mgr.trigger_retraining()
        print("\n" + "=" * 60)
        print("CONTINUOUS RETRAINING RESULT:")
        print(res)
        print("=" * 60)

    elif args.merged:
        from merged_phishing_detector import main as run_detector_cli
        sys.argv = ["merged_phishing_detector.py"]
        run_detector_cli()

    elif args.merged_eval:
        from merged_phishing_detector import main as run_detector_cli
        sys.argv = ["merged_phishing_detector.py", "--evaluate", "--samples", "300"]
        run_detector_cli()

    elif args.all:
        print("Starting full pipeline execution...")
        run_step("explore_dataset.py", "Dataset Exploration")
        run_step("preprocessing.py", "Preprocessing (Split, Scale, SMOTE)")
        run_step("train_dnn.py", "DNN Model Training")
        run_step("generate_results.py", "Comprehensive Performance Evaluation")
        print("\n[✓] Full pipeline completed successfully!")


if __name__ == "__main__":
    main()
