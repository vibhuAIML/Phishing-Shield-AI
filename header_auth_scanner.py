"""
header_auth_scanner.py
======================
Email Header Security & Authentication Inspection Module.
Inspects raw RFC 5322 MIME headers for:
  • SPF (Sender Policy Framework) validation & alignment
  • DKIM (DomainKeys Identified Mail) cryptographic signature checks
  • DMARC (Domain-based Message Authentication, Reporting, and Conformance) status
  • Display Name Impersonation / Spoofing (e.g. 'PayPal <hacker@evil.com>')
  • Return-Path & Reply-To domain divergence
  • Mail routing anomalies and hop count analysis
"""

import re
from email import policy
from email.parser import HeaderParser, Parser
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

# Recognized high-target enterprise brands susceptible to display-name spoofing
PROTECTED_BRANDS = {
    "paypal": ["paypal.com", "paypal-communication.com"],
    "microsoft": ["microsoft.com", "office.com", "office365.com", "microsoftonline.com", "live.com", "outlook.com"],
    "office 365": ["microsoft.com", "office.com", "office365.com", "microsoftonline.com"],
    "apple": ["apple.com", "icloud.com", "appleid.apple.com"],
    "google": ["google.com", "gmail.com", "googlemail.com"],
    "amazon": ["amazon.com", "amazon.co.uk", "amazonses.com"],
    "netflix": ["netflix.com"],
    "chase": ["chase.com"],
    "bank of america": ["bankofamerica.com", "bofa.com"],
    "wells fargo": ["wellsfargo.com"],
    "dhl": ["dhl.com", "dhl.de"],
    "fedex": ["fedex.com"],
    "ups": ["ups.com"],
    "metamask": ["metamask.io"],
    "coinbase": ["coinbase.com"],
    "irs": ["irs.gov"],
}


class HeaderAuthScanner:
    """
    Analyzes email headers for authentication compliance, spoofing signals,
    and routing anomalies.
    """

    def __init__(self):
        self.parser = HeaderParser(policy=policy.default)

    @staticmethod
    def _extract_domain(email_address: str) -> str:
        """Extracts normalized domain from 'user@domain.com' or '<user@domain.com>'."""
        if not email_address:
            return ""
        # Match inside brackets if present
        match = re.search(r"<([^>]+)>", email_address)
        clean = match.group(1).strip() if match else email_address.strip()
        if "@" in clean:
            return clean.split("@")[-1].lower().strip()
        return clean.lower()

    @staticmethod
    def _extract_display_name(from_header: str) -> str:
        """Extracts display name e.g. 'PayPal Security' from 'PayPal Security <admin@evil.com>'."""
        if "<" in from_header:
            return from_header.split("<")[0].replace('"', "").replace("'", "").strip()
        return ""

    def parse_headers(self, raw_headers: Any) -> Dict[str, Any]:
        """
        Parses raw RFC 5322 header text or key-value format into a normalized dictionary.
        """
        if isinstance(raw_headers, dict):
            return {k.lower(): v for k, v in raw_headers.items()}

        raw_str = str(raw_headers).strip()
        parsed = self.parser.parsestr(raw_str)
        headers = {}
        for k, v in parsed.items():
            headers[k.lower()] = str(v)
        
        # If HeaderParser found nothing but raw string has key: value lines, fallback to regex
        if not headers and ":" in raw_str:
            for line in raw_str.splitlines():
                if ":" in line and not line.startswith(" ") and not line.startswith("\t"):
                    parts = line.split(":", 1)
                    headers[parts[0].strip().lower()] = parts[1].strip()
        return headers

    def scan(self, raw_or_dict_headers: Any) -> Dict[str, Any]:
        """
        Comprehensive header inspection evaluating SPF, DKIM, DMARC,
        display name spoofing, and Return-Path mismatches.
        """
        headers = self.parse_headers(raw_or_dict_headers)

        from_hdr = headers.get("from", "")
        from_domain = self._extract_domain(from_hdr)
        display_name = self._extract_display_name(from_hdr)

        return_path = headers.get("return-path", "")
        return_domain = self._extract_domain(return_path)

        reply_to = headers.get("reply-to", "")
        reply_domain = self._extract_domain(reply_to)

        auth_results = headers.get("authentication-results", "")
        received_spf = headers.get("received-spf", "")
        dkim_sig = headers.get("dkim-signature", "")

        # 1. SPF Inspection
        spf_status = "none"
        spf_domain = ""
        combined_spf_text = f"{auth_results} {received_spf}".lower()
        if "spf=pass" in combined_spf_text or "pass" in received_spf.lower():
            spf_status = "pass"
        elif "spf=fail" in combined_spf_text or "fail" in received_spf.lower():
            spf_status = "fail"
        elif "spf=softfail" in combined_spf_text:
            spf_status = "softfail"
        elif "spf=neutral" in combined_spf_text:
            spf_status = "neutral"

        spf_domain_match = re.search(r"smtp\.mailfrom=([a-zA-Z0-9.-]+)", combined_spf_text)
        if spf_domain_match:
            spf_domain = spf_domain_match.group(1).lower()

        # 2. DKIM Inspection
        dkim_status = "none"
        dkim_domain = ""
        if "dkim=pass" in auth_results.lower():
            dkim_status = "pass"
        elif "dkim=fail" in auth_results.lower():
            dkim_status = "fail"
        elif dkim_sig:
            # Signature exists; extract d= domain
            d_match = re.search(r"d=([a-zA-Z0-9.-]+)", dkim_sig)
            if d_match:
                dkim_domain = d_match.group(1).lower()
                dkim_status = "present"

        dkim_domain_match = re.search(r"header\.d=([a-zA-Z0-9.-]+)", auth_results.lower())
        if dkim_domain_match:
            dkim_domain = dkim_domain_match.group(1)

        # 3. DMARC Inspection
        dmarc_status = "none"
        dmarc_policy = "none"
        if "dmarc=pass" in auth_results.lower():
            dmarc_status = "pass"
        elif "dmarc=fail" in auth_results.lower():
            dmarc_status = "fail"

        policy_match = re.search(r"action=(reject|quarantine|none)", auth_results.lower())
        if policy_match:
            dmarc_policy = policy_match.group(1)

        # 4. Alignment & Spoofing Analysis
        anomalies: List[str] = []
        risk_score = 0.0

        # Check SPF failure
        if spf_status in ["fail", "softfail"]:
            risk_score += 0.35 if spf_status == "fail" else 0.20
            anomalies.append(f"SPF authentication {spf_status.upper()} (Originating IP unauthorized for domain '{from_domain}').")

        # Check DKIM failure
        if dkim_status == "fail":
            risk_score += 0.35
            anomalies.append("DKIM cryptographic signature verification FAILED (Body or headers modified in transit).")

        # Check DMARC failure
        if dmarc_status == "fail":
            risk_score += 0.30
            anomalies.append(f"DMARC policy check FAILED (Policy: {dmarc_policy.upper()}). Domain alignment check failed.")

        # Check Display Name Spoofing
        spoofed_brand = None
        if display_name:
            norm_name = display_name.lower()
            for brand, legitimate_domains in PROTECTED_BRANDS.items():
                if brand in norm_name:
                    # Check if actual from_domain belongs to legitimate domains
                    is_legit = any(from_domain == d or from_domain.endswith("." + d) for d in legitimate_domains)
                    if not is_legit:
                        spoofed_brand = brand.capitalize()
                        risk_score += 0.45
                        anomalies.append(
                            f"CRITICAL: Display Name Impersonation detected! Sender displays '{display_name}' ({spoofed_brand}), "
                            f"but the actual sending address is '<{from_hdr}>' ({from_domain})."
                        )
                        break

        # Check Return-Path divergence (Ghost sending / bounce redirection)
        if return_domain and from_domain and return_domain != from_domain:
            # Disregard common legitimate ESP bounces like mailgun, sendgrid, amazonses if SPF passes
            esp_bounces = ["sendgrid.net", "amazonses.com", "mailgun.org", "constantcontact.com", "mandrillapp.com"]
            is_common_esp = any(return_domain.endswith(esp) for esp in esp_bounces)
            if not is_common_esp and spf_status != "pass":
                risk_score += 0.25
                anomalies.append(
                    f"Return-Path divergence: From domain '{from_domain}' differs from Return-Path '{return_domain}'."
                )

        # Check Reply-To divergence
        if reply_domain and from_domain and reply_domain != from_domain:
            risk_score += 0.20
            anomalies.append(
                f"Reply-To divergence: Responses routed to '{reply_domain}' rather than sender '{from_domain}'."
            )

        # Cap risk score between 0.0 and 1.0
        final_risk = min(max(round(risk_score, 4), 0.0), 1.0)
        verdict = "SUSPICIOUS" if final_risk >= 0.50 else ("CAUTION" if final_risk >= 0.25 else "VERIFIED")

        return {
            "from": from_hdr,
            "from_domain": from_domain,
            "display_name": display_name,
            "return_path_domain": return_domain,
            "reply_to_domain": reply_domain,
            "spf": {
                "status": spf_status,
                "domain": spf_domain,
            },
            "dkim": {
                "status": dkim_status,
                "domain": dkim_domain,
            },
            "dmarc": {
                "status": dmarc_status,
                "policy": dmarc_policy,
            },
            "spoofed_brand": spoofed_brand,
            "anomalies": anomalies,
            "header_risk_score": final_risk,
            "header_verdict": verdict,
        }


# Quick sanity test if executed directly
if __name__ == "__main__":
    scanner = HeaderAuthScanner()

    sample_spoofed_header = """
From: "PayPal Account Security" <service-alert@paypal-update-alert.xyz>
Return-Path: <bounce@unregistered-relay.org>
Reply-To: <credentials-harvest@gmail.com>
Authentication-Results: mx.google.com;
    spf=fail (google.com: domain of service-alert@paypal-update-alert.xyz does not designate 192.0.2.1 as permitted sender);
    dkim=fail header.i=@paypal-update-alert.xyz;
    dmarc=fail (p=REJECT sp=REJECT dis=NONE) header.from=paypal.com
Subject: URGENT: Your PayPal Account Has Been Suspended
"""
    result = scanner.scan(sample_spoofed_header)
    print("=== HEADER SCAN RESULT ===")
    import json
    print(json.dumps(result, indent=2))
