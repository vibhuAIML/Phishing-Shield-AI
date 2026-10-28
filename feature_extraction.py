"""
URL Phishing Detection — Feature Extraction for Deep Learning (DNN)
=====================================================================
Extracts purely lexical & structural numeric features from raw URLs
(no live network / WHOIS / DNS lookups needed), so it works offline
and scales to 600k+ rows.

Input : merged_raw_dataset.csv   (columns: url, label, source)
Output: url_features_dnn.csv     (all numeric feature columns + label)
"""

import re
import math
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from urllib.parse import urlsplit

# ----------------------------------------------------------------------
# Reference lists used by several features
# ----------------------------------------------------------------------
SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "shorte.st", "cutt.ly", "rebrand.ly", "tiny.cc", "bit.do",
    "soo.gd", "s2r.co", "qrco.de", "rb.gy", "v.gd"
}

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "loan",
    "win", "bid", "party", "review", "download", "racing", "date", "faith",
    "science", "trade", "webcam", "men", "stream", "gdn", "kim"
}

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "bank", "password", "pay", "webscr", "wp-admin", "suspend", "billing",
    "security", "invoice", "unlock", "alert", "recover", "authenticate"
]

IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
VOWELS = set("aeiouAEIOU")


def shannon_entropy(s: str) -> float:
    """Shannon entropy of a string (0 for empty string)."""
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def longest_word_len(s: str) -> int:
    words = re.split(r"[^A-Za-z0-9]+", s)
    return max((len(w) for w in words if w), default=0)


def avg_word_len(s: str) -> float:
    words = [w for w in re.split(r"[^A-Za-z0-9]+", s) if w]
    return float(np.mean([len(w) for w in words])) if words else 0.0


def extract_features(url: str) -> dict:
    url = str(url).strip()

    full_url = url if "://" in url else "http://" + url
    try:
        parts = urlsplit(full_url)
    except ValueError:
        sanitized = full_url.replace("[", "").replace("]", "")
        try:
            parts = urlsplit(sanitized)
        except Exception:
            parts = urlsplit("http://malformed-url.invalid")
    scheme = (parts.scheme or "").lower()
    path = parts.path or ""
    query = parts.query or ""
    fragment = parts.fragment or ""
    netloc = parts.netloc or ""

    # hostname: parse manually (tolerant of malformed IPv6/port strings that
    # make urllib's strict .hostname/.port properties raise ValueError)
    host_part = netloc.split("@")[-1]  # drop userinfo, if any
    if host_part.startswith("["):  # IPv6 literal, e.g. [::1]:8080
        hostname = host_part.split("]")[0].lstrip("[").lower()
        port_present = int("]:" in host_part)
    else:
        host_pieces = host_part.split(":")
        hostname = host_pieces[0].lower()
        port_present = int(len(host_pieces) > 1 and host_pieces[1] != "")

    # registered domain / suffix approximation (no tldextract available offline)
    host_labels = hostname.split(".") if hostname else []
    tld = host_labels[-1] if len(host_labels) >= 2 else ""
    domain = host_labels[-2] if len(host_labels) >= 2 else (host_labels[0] if host_labels else "")
    num_subdomains = max(len(host_labels) - 2, 0)

    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    total_len = len(url) if len(url) > 0 else 1

    feats = {
        # --- length features ---
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "fragment_length": len(fragment),

        # --- raw character counts ---
        "count_dots": url.count("."),
        "count_hyphens": url.count("-"),
        "count_underscore": url.count("_"),
        "count_slash": url.count("/"),
        "count_question": url.count("?"),
        "count_equal": url.count("="),
        "count_at": url.count("@"),
        "count_and": url.count("&"),
        "count_exclamation": url.count("!"),
        "count_space": url.count(" ") + url.count("%20"),
        "count_tilde": url.count("~"),
        "count_comma": url.count(","),
        "count_plus": url.count("+"),
        "count_asterisk": url.count("*"),
        "count_hash": url.count("#"),
        "count_dollar": url.count("$"),
        "count_percent": url.count("%"),
        "count_semicolon": url.count(";"),
        "count_digits": digits,
        "count_letters": letters,

        # --- ratios ---
        "digit_ratio": digits / total_len,
        "letter_ratio": letters / total_len,
        "special_char_ratio": (total_len - digits - letters) / total_len,
        "vowel_ratio": sum(c in VOWELS for c in url) / total_len,

        # --- domain / structural features ---
        "num_subdomains": num_subdomains,
        "hostname_is_ip": int(bool(IPV4_RE.match(hostname))),
        "port_present": port_present,
        "tld_length": len(tld),
        "domain_length": len(domain),
        "digits_in_domain": sum(c.isdigit() for c in domain),
        "hyphen_in_domain": int("-" in domain),

        # --- security / trick indicators ---
        "has_https": int(scheme == "https"),
        "has_www": int(hostname.startswith("www.")),
        "has_at_symbol": int("@" in url),
        "has_double_slash_redirect": int("//" in url[7:]),
        "has_shortening_service": int(hostname in SHORTENERS),
        "has_suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "has_punycode": int("xn--" in hostname),
        "http_token_in_path": int("http" in (path + query).lower()),
        "suspicious_keyword_count": sum(kw in url.lower() for kw in SUSPICIOUS_KEYWORDS),

        # --- entropy / word-shape ---
        "url_entropy": shannon_entropy(url),
        "hostname_entropy": shannon_entropy(hostname),
        "longest_word_length": longest_word_len(url),
        "avg_word_length": avg_word_len(url),
    }
    return feats


def build_feature_dataframe(df: pd.DataFrame, url_col: str = "url") -> pd.DataFrame:
    records = df[url_col].apply(extract_features)
    feat_df = pd.DataFrame(records.tolist())
    return feat_df


if __name__ == "__main__":
    import os
    import sys

    IN_PATH = sys.argv[1] if len(sys.argv) > 1 else ("merged_raw_dataset.csv" if os.path.exists("merged_raw_dataset.csv") else "/mnt/user-data/uploads/merged_raw_dataset.csv")
    OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "url_features_dnn.csv"

    print(f"Loading raw dataset from {IN_PATH}...")
    df = pd.read_csv(IN_PATH)

    print(f"Extracting features for {len(df):,} URLs...")
    feat_df = build_feature_dataframe(df, url_col="url")

    # attach label (and keep source for optional stratification / auditing)
    if "label" in df.columns:
        feat_df["label"] = df["label"].values
    if "source" in df.columns:
        feat_df["source"] = df["source"].values

    print("Feature matrix shape:", feat_df.shape)
    feat_df.to_csv(OUT_PATH, index=False)
    print(f"Saved -> {OUT_PATH}")
