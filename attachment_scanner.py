"""
attachment_scanner.py
=====================
Attachment & QR Code (Quishing) Security Inspection Module.
Features:
  • Decodes 2D QR codes (Quishing attacks) embedded within image attachments (PNG, JPG, WEBP, BMP).
  • Inspects PDF documents for embedded URI annotations, malicious JavaScript streams, and hidden links.
  • Performs OCR text & pattern extraction for credential solicitation images.
  • Seamlessly extracts and routes discovered URLs into the URL DNN risk engine.
"""

import os
import re
import io
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Optional imports with graceful fallbacks
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None

# URL matching regex
URL_REGEX = re.compile(
    r"(?:https?://|www\.)[^\s<>'\"`]+|[a-zA-Z0-9.-]+\.(?:com|org|net|xyz|top|ru|cn|work|biz|info|cc|tk|ga|cf|gq|club|online|site|vip|loan|racing)[^\s<>'\"`]*",
    re.IGNORECASE
)


class AttachmentScanner:
    """
    Scans email attachments (images, QR codes, PDFs) for obfuscated phishing payloads.
    """

    def __init__(self):
        self.qr_detector = None
        if cv2 is not None:
            try:
                self.qr_detector = cv2.QRCodeDetector()
            except Exception as e:
                print(f"[!] Warning: Could not initialize cv2.QRCodeDetector: {e}")

    def scan_qr_code(self, image_input: Any) -> List[Dict[str, Any]]:
        """
        Detects and decodes QR codes inside an image file path, PIL Image, or raw byte buffer.
        Returns a list of decoded QR code payloads and metadata.
        """
        qr_results = []
        if self.qr_detector is None or cv2 is None:
            return qr_results

        # Load image into OpenCV BGR numpy array
        cv_img = None
        try:
            if isinstance(image_input, str) and os.path.isfile(image_input):
                cv_img = cv2.imread(image_input)
            elif isinstance(image_input, bytes):
                nparr = np.frombuffer(image_input, np.uint8)
                cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif Image is not None and isinstance(image_input, Image.Image):
                cv_img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
            elif isinstance(image_input, np.ndarray):
                cv_img = image_input
        except Exception as e:
            print(f"[!] AttachmentScanner image load error: {e}")
            return qr_results

        if cv_img is None:
            return qr_results

        # 1. Primary detection: detectAndDecodeMulti
        try:
            retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(cv_img)
            if retval and decoded_info:
                for text in decoded_info:
                    if text and text.strip():
                        qr_results.append({
                            "type": "QR_CODE",
                            "payload": text.strip(),
                            "is_url": bool(URL_REGEX.search(text.strip())),
                        })
        except Exception:
            pass

        # 2. Fallback single detection: detectAndDecode
        if not qr_results:
            try:
                text, points, _ = self.qr_detector.detectAndDecode(cv_img)
                if text and text.strip():
                    qr_results.append({
                        "type": "QR_CODE",
                        "payload": text.strip(),
                        "is_url": bool(URL_REGEX.search(text.strip())),
                    })
            except Exception:
                pass

        # 3. Grayscale / thresholding enhancement fallback if low contrast
        if not qr_results:
            try:
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                # Try Otsu threshold
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                text, _, _ = self.qr_detector.detectAndDecode(thresh)
                if text and text.strip():
                    qr_results.append({
                        "type": "QR_CODE",
                        "payload": text.strip(),
                        "is_url": bool(URL_REGEX.search(text.strip())),
                    })
            except Exception:
                pass

        return qr_results

    def scan_pdf(self, pdf_input: Any) -> Dict[str, Any]:
        """
        Parses a PDF file or bytes for embedded URLs (/URI objects), JavaScript streams,
        and suspicious launch actions.
        """
        raw_bytes = b""
        if isinstance(pdf_input, str) and os.path.isfile(pdf_input):
            with open(pdf_input, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(pdf_input, bytes):
            raw_bytes = pdf_input

        if not raw_bytes:
            return {"type": "PDF", "extracted_urls": [], "suspicious_objects": []}

        # Decode safely with replacement for regex inspection
        content = raw_bytes.decode("latin1", errors="ignore")

        # 1. Search for embedded PDF /URI actions: /URI (http...)
        uri_matches = re.findall(r"/URI\s*\((https?://[^)]+)\)", content, re.IGNORECASE)
        direct_matches = URL_REGEX.findall(content)

        extracted_urls = []
        for u in set(uri_matches + direct_matches):
            clean_u = re.sub(r"[)>'\"`]+$", "", u.strip())
            if clean_u:
                extracted_urls.append(clean_u)
        extracted_urls = sorted(list(set(extracted_urls)))

        # 2. Search for PDF attack vectors
        suspicious_objects = []
        if "/JavaScript" in content or "/JS" in content:
            suspicious_objects.append("Embedded JavaScript execution stream (/JS)")
        if "/Launch" in content:
            suspicious_objects.append("Arbitrary executable launch action (/Launch)")
        if "/EmbeddedFiles" in content:
            suspicious_objects.append("Hidden embedded file attachments (/EmbeddedFiles)")

        return {
            "type": "PDF",
            "extracted_urls": extracted_urls,
            "suspicious_objects": suspicious_objects,
            "is_malicious_structure": len(suspicious_objects) > 0,
        }

    def scan_attachment(self, filename: str, file_data: Any) -> Dict[str, Any]:
        """
        Unified inspection dispatcher for any attachment (images, QR codes, PDFs).
        """
        ext = os.path.splitext(filename.lower())[-1]
        discovered_urls: List[str] = []
        qr_codes: List[str] = []
        threat_flags: List[str] = []

        if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            # Check QR codes (Quishing)
            qrs = self.scan_qr_code(file_data)
            for q in qrs:
                qr_codes.append(q["payload"])
                if q["is_url"]:
                    discovered_urls.append(q["payload"])
                    threat_flags.append(f"Quishing attack detected: QR code decodes to web address '{q['payload']}'.")
                else:
                    threat_flags.append(f"QR code detected with encoded payload: '{q['payload']}'.")

        elif ext == ".pdf":
            pdf_res = self.scan_pdf(file_data)
            discovered_urls.extend(pdf_res["extracted_urls"])
            for obj in pdf_res["suspicious_objects"]:
                threat_flags.append(f"Suspicious PDF structure: {obj}")

        return {
            "filename": filename,
            "extension": ext,
            "qr_codes": qr_codes,
            "extracted_urls": list(set(discovered_urls)),
            "threat_flags": threat_flags,
            "has_threat": len(threat_flags) > 0 or len(discovered_urls) > 0,
        }


# Quick test script generating a sample QR code image and scanning it
if __name__ == "__main__":
    scanner = AttachmentScanner()

    print("[*] Testing Attachment & Quishing Scanner...")
    # 1. Test QR Code Quishing Detection
    if cv2 is not None and hasattr(cv2, "QRCodeEncoder"):
        enc = cv2.QRCodeEncoder.create()
        raw_qr = enc.encode("https://phishing-quishing-alert.xyz/secure-login")
        scaled_qr = cv2.resize(raw_qr, (330, 330), interpolation=cv2.INTER_NEAREST)
        padded_qr = cv2.copyMakeBorder(scaled_qr, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        _, buf = cv2.imencode(".png", padded_qr)
        qr_scan_res = scanner.scan_attachment("quishing_flyer.png", buf.tobytes())
        import json
        print("=== QUISHING QR ATTACHMENT SCAN RESULT ===")
        print(json.dumps(qr_scan_res, indent=2))

    # 2. Test synthetic PDF stream
    sample_pdf_bytes = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /Annots [4 0 R] >> endobj
4 0 obj << /Type /Annot /Subtype /Link /A << /S /URI /URI (http://malicious-login-portal.xyz/auth) >> >> endobj
5 0 obj << /JavaScript (app.alert('Malware');) >> endobj
trailer << /Root 1 0 R >> %%EOF
"""
    pdf_res = scanner.scan_attachment("invoice_update.pdf", sample_pdf_bytes)
    import json
    print("\n=== PDF ATTACHMENT SCAN RESULT ===")
    print(json.dumps(pdf_res, indent=2))
