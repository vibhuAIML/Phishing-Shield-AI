"""
celery_tasks.py
===============
Asynchronous Celery Task Queue for Enterprise-Scale Phishing Inspection.
Integrates with Redis message broker for distributed worker processing.
Tasks:
  • task_predict_email: Multimodal background scan of incoming corporate messages
  • task_predict_url: Microsecond URL feature extraction and DNN scoring
  • task_scan_attachment: Asynchronous Quishing QR and PDF payload extraction
  • task_scan_headers: SPF/DKIM/DMARC and display name spoofing validation
  • task_gateway_sync: Automated enterprise mailbox polling & remediation
"""

import os
import base64
from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Initialize Celery app instance
celery_app = Celery(
    "phishing_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# Celery task configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=120,
)

# Lazy worker detector singletons
_detector = None
_header_scanner = None
_attach_scanner = None


def get_detector():
    global _detector
    if _detector is None:
        from merged_phishing_detector import MergedPhishingDetector
        _detector = MergedPhishingDetector()
    return _detector


def get_header_scanner():
    global _header_scanner
    if _header_scanner is None:
        from header_auth_scanner import HeaderAuthScanner
        _header_scanner = HeaderAuthScanner()
    return _header_scanner


def get_attach_scanner():
    global _attach_scanner
    if _attach_scanner is None:
        from attachment_scanner import AttachmentScanner
        _attach_scanner = AttachmentScanner()
    return _attach_scanner


@celery_app.task(name="tasks.predict_email")
def task_predict_email(subject: str, body: str, raw_headers: str = None) -> dict:
    """Asynchronous email scan task."""
    detector = get_detector()
    result = detector.predict(subject=subject, body=body)

    if raw_headers:
        hdr_scanner = get_header_scanner()
        hdr_res = hdr_scanner.scan(raw_headers)
        result["header_security"] = hdr_res

    return result


@celery_app.task(name="tasks.predict_url")
def task_predict_url(url: str) -> dict:
    """Asynchronous URL DNN classification task."""
    detector = get_detector()
    return detector.predict_url(url)


@celery_app.task(name="tasks.scan_attachment")
def task_scan_attachment(filename: str, file_b64: str) -> dict:
    """Asynchronous Quishing QR and PDF payload detection task."""
    scanner = get_attach_scanner()
    file_bytes = base64.b64decode(file_b64)
    res = scanner.scan_attachment(filename, file_bytes)

    # Route any discovered URLs through the URL DNN
    if res.get("extracted_urls"):
        detector = get_detector()
        url_scores = [detector.predict_url(u) for u in res["extracted_urls"]]
        res["url_assessments"] = url_scores

    return res


@celery_app.task(name="tasks.gateway_sync")
def task_gateway_sync() -> dict:
    """Asynchronous batch synchronization with M365 and Gmail mail gateways."""
    from enterprise_connectors import EnterpriseGatewayManager
    detector = get_detector()
    manager = EnterpriseGatewayManager()
    return manager.run_gateway_inspection(detector)


@celery_app.task(name="tasks.trigger_retrain")
def task_trigger_retrain() -> dict:
    """Asynchronous background retraining execution."""
    from feedback_retraining import FeedbackRetrainingManager
    mgr = FeedbackRetrainingManager()
    return mgr.trigger_retraining()


if __name__ == "__main__":
    print("[*] Celery Task definitions compiled successfully.")
    print("  • Tasks registered:", list(celery_app.tasks.keys()))
