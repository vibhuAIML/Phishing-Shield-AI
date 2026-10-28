"""
test_suite_all.py
=================
Comprehensive End-to-End Verification Test Suite for All New & Existing Modules:
  1. Header Authentication Scanner (SPF / DKIM / DMARC / Brand Spoofing)
  2. Attachment & Quishing Scanner (QR Code Decoding & PDF /URI Inspection)
  3. Enterprise Mail Flow Connectors (Microsoft 365 & Google Workspace)
  4. Analyst Feedback & Continuous Retraining System
  5. Unified Multimodal Detector (Text + URL + Headers + Attachments)
  6. Flask REST API & Web Server Endpoints
"""

import sys
import os
import json
import base64
import unittest
import numpy as np

# Ensure working directory is project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


class TestPhishingSystemEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 70)
        print("  STARTING PHISHING DETECTION COMPREHENSIVE TEST SUITE")
        print("=" * 70)

    # --------------------------------------------------------------------------
    # TEST 1: Header Authentication Scanner
    # --------------------------------------------------------------------------
    def test_01_header_auth_scanner(self):
        print("\n[TEST 1] Testing HeaderAuthScanner...")
        from header_auth_scanner import HeaderAuthScanner
        scanner = HeaderAuthScanner()

        # Spoofed PayPal header
        spoofed_header = """
From: "PayPal Security Center" <service@paypal-security-alert.xyz>
Return-Path: <bounce@unregistered-node.net>
Reply-To: <credential-collector@gmail.com>
Authentication-Results: mx.google.com;
    spf=fail (google.com: domain of service@paypal-security-alert.xyz does not designate 192.0.2.1 as permitted sender);
    dkim=fail header.i=@paypal-security-alert.xyz;
    dmarc=fail (p=REJECT sp=REJECT) header.from=paypal.com
Subject: URGENT: Your PayPal Account Has Been Suspended
"""
        res = scanner.scan(spoofed_header)
        self.assertEqual(res["header_verdict"], "SUSPICIOUS")
        self.assertGreaterEqual(res["header_risk_score"], 0.5)
        self.assertEqual(res["spf"]["status"], "fail")
        self.assertEqual(res["dkim"]["status"], "fail")
        self.assertEqual(res["dmarc"]["status"], "fail")
        self.assertEqual(res["spoofed_brand"], "Paypal")
        self.assertTrue(any("Display Name Impersonation" in a for a in res["anomalies"]))
        print(f"  [✓] Spoofed Header Correctly Flagged: Risk Score {res['header_risk_score']} (Spoofed: {res['spoofed_brand']})")

        # Legitimate header
        legit_header = """
From: "Microsoft Security" <no-reply@microsoft.com>
Return-Path: <ms-bounces@microsoft.com>
Authentication-Results: spf=pass smtp.mailfrom=microsoft.com;
    dkim=pass header.d=microsoft.com;
    dmarc=pass header.from=microsoft.com
Subject: Security notice
"""
        res_legit = scanner.scan(legit_header)
        self.assertEqual(res_legit["header_verdict"], "VERIFIED")
        self.assertLess(res_legit["header_risk_score"], 0.25)
        print(f"  [✓] Legitimate Header Correctly Verified: Risk Score {res_legit['header_risk_score']}")

    # --------------------------------------------------------------------------
    # TEST 2: Attachment & Quishing Scanner
    # --------------------------------------------------------------------------
    def test_02_attachment_and_quishing_scanner(self):
        print("\n[TEST 2] Testing AttachmentScanner (Quishing QR & PDF)...")
        from attachment_scanner import AttachmentScanner
        import cv2

        scanner = AttachmentScanner()

        # 2a. Synthetic QR Quishing Detection
        enc = cv2.QRCodeEncoder.create()
        qr_matrix = enc.encode("https://login-quishing-threat.xyz/verify")
        scaled_qr = cv2.resize(qr_matrix, (330, 330), interpolation=cv2.INTER_NEAREST)
        padded_qr = cv2.copyMakeBorder(scaled_qr, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        _, buf = cv2.imencode(".png", padded_qr)
        qr_bytes = buf.tobytes()

        qr_res = scanner.scan_attachment("quishing_flyer.png", qr_bytes)
        self.assertTrue(qr_res["has_threat"])
        self.assertIn("https://login-quishing-threat.xyz/verify", qr_res["extracted_urls"])
        self.assertTrue(any("Quishing" in f for f in qr_res["threat_flags"]))
        print(f"  [✓] Quishing QR Decoded Successfully: '{qr_res['extracted_urls'][0]}'")

        # 2b. PDF /URI and JavaScript Stream Detection
        sample_pdf_bytes = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /Annots [4 0 R] >> endobj
4 0 obj << /Type /Annot /Subtype /Link /A << /S /URI /URI (http://malicious-invoice-portal.xyz/pay) >> >> endobj
5 0 obj << /JavaScript (app.alert('Malware');) >> endobj
trailer << /Root 1 0 R >> %%EOF
"""
        pdf_res = scanner.scan_attachment("invoice.pdf", sample_pdf_bytes)
        self.assertTrue(pdf_res["has_threat"])
        self.assertIn("http://malicious-invoice-portal.xyz/pay", pdf_res["extracted_urls"])
        self.assertTrue(any("JavaScript" in f for f in pdf_res["threat_flags"]))
        print(f"  [✓] PDF Threat Successfully Detected: '{pdf_res['extracted_urls'][0]}' + JS stream")

    # --------------------------------------------------------------------------
    # TEST 3: Enterprise Mail Flow Connectors
    # --------------------------------------------------------------------------
    def test_03_enterprise_connectors(self):
        print("\n[TEST 3] Testing Enterprise Mail Flow Connectors (M365 & Gmail)...")
        from enterprise_connectors import M365Connector, GoogleWorkspaceConnector, EnterpriseGatewayManager

        m365 = M365Connector(simulate=True)
        self.assertTrue(m365.authenticate())
        inbox = m365.fetch_inbox_messages()
        self.assertGreater(len(inbox), 0)
        self.assertTrue(m365.quarantine_message(inbox[0]["id"], verdict_score=0.99))
        print(f"  [✓] M365 Connector Operational: Polled {len(inbox)} messages, quarantine verified.")

        gmail = GoogleWorkspaceConnector(simulate=True)
        g_inbox = gmail.fetch_inbox_messages()
        self.assertGreater(len(g_inbox), 0)
        self.assertTrue(gmail.apply_phishing_label(g_inbox[0]["id"]))
        print(f"  [✓] Google Workspace Connector Operational: Polled {len(g_inbox)} messages, label verified.")

    # --------------------------------------------------------------------------
    # TEST 4: Analyst Feedback & Continuous Retraining
    # --------------------------------------------------------------------------
    def test_04_feedback_and_retraining(self):
        print("\n[TEST 4] Testing FeedbackRetrainingManager...")
        from feedback_retraining import FeedbackRetrainingManager
        mgr = FeedbackRetrainingManager()

        # Submit verified analyst feedback
        fb = mgr.submit_feedback(
            sample_type="url",
            target_content="https://internal-dev-portal.corp.test",
            predicted_prob=0.82,
            predicted_verdict="PHISHING",
            analyst_verdict="LEGITIMATE",
            analyst_notes="Verified internal dev sandbox domain."
        )
        self.assertEqual(fb["feedback_type"], "FALSE_POSITIVE")
        self.assertEqual(fb["status"], "success")

        metrics = mgr.get_feedback_metrics()
        self.assertGreater(metrics["total_feedback"], 0)
        self.assertGreater(metrics["false_positives"], 0)
        print(f"  [✓] Feedback Logged & Classified: Type={fb['feedback_type']} (ID #{fb['feedback_id']})")

        # Trigger incremental retraining
        retrain_res = mgr.trigger_retraining()
        self.assertEqual(retrain_res["status"], "completed")
        print(f"  [✓] Retraining Trigger Completed: {retrain_res['message']}")

    # --------------------------------------------------------------------------
    # TEST 5: Unified Multimodal Phishing Detector
    # --------------------------------------------------------------------------
    def test_05_multimodal_detector_fusion(self):
        print("\n[TEST 5] Testing MergedPhishingDetector (Full Fusion with Headers & Attachments)...")
        from merged_phishing_detector import MergedPhishingDetector
        detector = MergedPhishingDetector()

        # Test case: Polite pretexting email + spoofed header + malicious payload link
        test_headers = """
From: "PayPal Billing Department" <billing-alert@fake-paypal-verify.xyz>
Authentication-Results: spf=fail; dkim=fail; dmarc=fail
"""
        test_subject = "Important account information regarding your recent order"
        test_body = (
            "Hello, thank you for being a valued customer. Please review your recent order invoice: "
            "http://secure-paypal-login-verify.xyz/auth/login.php. Have a great day!"
        )

        res = detector.predict(
            subject=test_subject,
            body=test_body,
            headers=test_headers
        )

        self.assertEqual(res["verdict"], "PHISHING")
        self.assertGreaterEqual(res["combined_phishing_prob"], 0.85)
        self.assertEqual(res["risk_level"], "CRITICAL")
        self.assertIsNotNone(res["header_security"])
        self.assertEqual(res["header_security"]["spoofed_brand"], "Paypal")
        self.assertGreater(res["urls_detected_count"], 0)
        print(f"  [✓] Full Fusion Verdict: {res['verdict']} | Combined Risk: {res['combined_phishing_prob']*100:.1f}% ({res['risk_level']})")
        print(f"      Reasons: {res['explanation']}")

    # --------------------------------------------------------------------------
    # TEST 6: Flask REST API Endpoints
    # --------------------------------------------------------------------------
    def test_06_flask_api_endpoints(self):
        print("\n[TEST 6] Testing Flask REST API Endpoints...")
        from app import app
        client = app.test_client()

        # 6a. GET /api/health
        r_health = client.get("/api/health")
        self.assertEqual(r_health.status_code, 200)
        self.assertEqual(r_health.get_json()["status"], "online")
        print("  [✓] Endpoint GET /api/health passed.")

        # 6b. GET /api/system/status
        r_sys = client.get("/api/system/status")
        self.assertEqual(r_sys.status_code, 200)
        sys_data = r_sys.get_json()
        self.assertIn("modules", sys_data)
        self.assertEqual(sys_data["modules"]["quishing_qr_pdf_scanner"], "ready")
        self.assertEqual(sys_data["modules"]["enterprise_m365_gmail_connectors"], "ready")
        print(f"  [✓] Endpoint GET /api/system/status passed: Device={sys_data['device']}.")

        # 6c. POST /api/predict/url
        r_url = client.post("/api/predict/url", json={"url": "http://paypal-verification-account.xyz/login"})
        self.assertEqual(r_url.status_code, 200)
        self.assertEqual(r_url.get_json()["verdict"], "PHISHING")
        print("  [✓] Endpoint POST /api/predict/url passed.")

        # 6d. POST /api/scan/headers
        r_hdr = client.post("/api/scan/headers", json={
            "headers": "From: \"PayPal Alert\" <test@phishing.xyz>\nAuthentication-Results: spf=fail"
        })
        self.assertEqual(r_hdr.status_code, 200)
        self.assertEqual(r_hdr.get_json()["spf"]["status"], "fail")
        print("  [✓] Endpoint POST /api/scan/headers passed.")

        # 6e. GET /api/gateway/scan
        r_gw = client.get("/api/gateway/scan")
        self.assertEqual(r_gw.status_code, 200)
        gw_data = r_gw.get_json()
        self.assertGreater(gw_data["m365_scanned"], 0)
        print(f"  [✓] Endpoint GET /api/gateway/scan passed: {gw_data['m365_scanned']} messages audited.")

        # 6f. GET /api/feedback/stats
        r_fb = client.get("/api/feedback/stats")
        self.assertEqual(r_fb.status_code, 200)
        print(f"  [✓] Endpoint GET /api/feedback/stats passed.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
