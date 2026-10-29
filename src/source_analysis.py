"""
News Source Credibility Analysis Module.
Analyzes publisher domain, metadata, known editorial records, and fact-checking accreditation.
Adheres strictly to the principle: Never classify a publisher as fake without evidence.
"""

import os
import json
from urllib.parse import urlparse
from typing import Dict, Any, Optional

DEFAULT_SOURCES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "reputable_sources.json"
)


class SourceAnalyzer:
    """
    Evaluates news domain credibility against curated, verifiable publisher registries.
    """

    def __init__(self, sources_file: str = DEFAULT_SOURCES_PATH):
        self.sources_file = sources_file
        self.sources_db = self._load_database()

    def _load_database(self) -> Dict[str, Any]:
        """Loads the domain reputation database safely."""
        if os.path.exists(self.sources_file):
            try:
                with open(self.sources_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("domains", {})
            except Exception:
                return {}
        return {}

    def extract_clean_domain(self, url_or_domain: str) -> str:
        """
        Normalizes a URL or domain string to lower-case base domain without www.
        """
        if not url_or_domain:
            return ""

        url_or_domain = url_or_domain.strip().lower()
        if "://" in url_or_domain:
            parsed = urlparse(url_or_domain)
            domain = parsed.hostname or ""
        else:
            domain = url_or_domain.split("/")[0]

        # Strip www.
        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    def analyze_source(
        self,
        url_or_domain: str,
        extracted_author: Optional[str] = None,
        extracted_publisher: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates domain credibility.
        Returns detailed publisher profile and credibility indicators.
        """
        clean_domain = self.extract_clean_domain(url_or_domain)
        if not clean_domain:
            return {
                "has_source_info": False,
                "domain": "Not Provided",
                "publisher_name": extracted_publisher or "Not Specified",
                "author": extracted_author or "Not Specified",
                "credibility_tier": "No Source Data",
                "status_badge": "⚪ Unspecified Source",
                "description": "Article text was entered without a corresponding publisher URL or domain.",
                "fact_check_certified": False,
                "is_satire": False,
                "warning": None,
            }

        # Check exact or parent domain match
        matched_info = None
        if clean_domain in self.sources_db:
            matched_info = self.sources_db[clean_domain]
        else:
            # Check subdomain matching e.g., hindi.news18.com -> news18.com
            parts = clean_domain.split(".")
            if len(parts) > 2:
                parent_domain = ".".join(parts[-2:])
                if parent_domain in self.sources_db:
                    matched_info = self.sources_db[parent_domain]

        if matched_info:
            is_satire = "Satire" in matched_info.get("category", "")
            return {
                "has_source_info": True,
                "domain": clean_domain,
                "publisher_name": matched_info.get("name", clean_domain),
                "author": extracted_author if extracted_author else "Not explicitly stated in article metadata",
                "category": matched_info.get("category", "General Publisher"),
                "credibility_tier": matched_info.get("credibility_rating", "Verified Record"),
                "status_badge": (
                    "🎭 Known Satire / Parody" if is_satire
                    else "🛡️ Verified Fact-Checker" if matched_info.get("fact_check_certified")
                    else "✅ High Credibility Source"
                ),
                "country": matched_info.get("country", "Global"),
                "fact_check_certified": matched_info.get("fact_check_certified", False),
                "is_satire": is_satire,
                "description": matched_info.get("notes", ""),
                "warning": "Warning: This source publishes satire/parody fiction for comedic purposes." if is_satire else None,
            }

        # If domain is not in curated DB: DO NOT brand as fake!
        return {
            "has_source_info": True,
            "domain": clean_domain,
            "publisher_name": extracted_publisher if extracted_publisher else clean_domain.capitalize(),
            "author": extracted_author if extracted_author else "Not found in HTML meta tags",
            "category": "Unindexed / Independent Publisher",
            "credibility_tier": "Unverified Record",
            "status_badge": "⚠️ Unindexed Domain (Requires Verification)",
            "country": "Unknown / International",
            "fact_check_certified": False,
            "is_satire": False,
            "description": (
                f"The domain '{clean_domain}' is not in the curated high-trust publisher directory. "
                "This does NOT mean the article is fake; independent fact-checking or cross-referencing is recommended."
            ),
            "warning": "Domain is not in known-reliable registry. Cross-reference claims with primary reporting.",
        }
