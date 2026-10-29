"""
Claim Extraction and Evidence Verification Module.
Extracts factual claims from text and retrieves evidence from verified records
or trusted external sources. Categorizes claims into: Supported, Contradicted,
or Insufficient Evidence.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CLAIMS_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "verified_claims.json"
)

ASSERTIVE_VERBS = {
    "announced", "confirmed", "discovered", "revealed", "proved", "declared",
    "claimed", "reported", "stated", "found", "found that", "causes", "cure",
    "destroyed", "killed", "launched", "banned", "सांगितले", "दावा", "घोषणा",
    "केले", "सांगण्यात आले", "खुलासा"
}


class ClaimVerifier:
    """
    Extracts key claims from an article and verifies them against curated
    evidence databases and fact-checking registries.
    """

    def __init__(self, claims_path: str = CLAIMS_DB_PATH):
        self.claims_path = claims_path
        self.evidence_db = self._load_claims()

    def _load_claims(self) -> List[Dict[str, Any]]:
        """Loads verified fact-checks from disk."""
        if os.path.exists(self.claims_path):
            try:
                with open(self.claims_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("verified_database", [])
            except Exception:
                return []
        return []

    def extract_salient_claims(self, text: str, max_claims: int = 4) -> List[str]:
        """
        Extracts salient assertive sentences likely to contain factual assertions.
        Uses sentence boundary splitting, length filtering, and verb cues.
        """
        if not text:
            return []

        # Sentence splitting handling Latin and Indic punctuation
        raw_sentences = re.split(r'[\.\n\r?!।॥]+', text)
        candidates = []

        for s in raw_sentences:
            s_clean = s.strip()
            # Ignore very short phrases or extremely long run-on passages
            if 30 <= len(s_clean) <= 280:
                s_lower = s_clean.lower()
                has_verb = any(v in s_lower for v in ASSERTIVE_VERBS)
                has_digit = bool(re.search(r'\d+', s_clean))
                # Prioritize assertive sentences or statistical statements
                score = (2 if has_verb else 0) + (1 if has_digit else 0)
                candidates.append((score, s_clean))

        # Sort by relevance score
        candidates.sort(key=lambda x: x[0], reverse=True)
        extracted = [c[1] for c in candidates[:max_claims]]

        # Fallback to first non-empty sentences if no assertive keywords matched
        if not extracted and raw_sentences:
            for s in raw_sentences:
                if len(s.strip()) > 30:
                    extracted.append(s.strip())
                    if len(extracted) >= max_claims:
                        break

        return extracted

    def verify_claim(self, claim_text: str) -> Dict[str, Any]:
        """
        Matches an individual claim against the verified evidence repository.
        Returns categorization: 'Supported', 'Contradicted', or 'Insufficient Evidence'.
        """
        if not self.evidence_db or not claim_text.strip():
            return {
                "claim": claim_text,
                "status": "Insufficient Evidence",
                "badge": "⚪ Insufficient Evidence",
                "similarity_score": 0.0,
                "evidence_summary": "No verified peer-reviewed or fact-check records match this statement in the archive.",
                "source_title": "Independent Review Required",
                "source_url": None,
                "date": None,
                "disclaimer": "Lack of existing fact-check evidence does not mean the claim is false."
            }

        claim_corpus = [entry["claim"] for entry in self.evidence_db]
        all_texts = [claim_text] + claim_corpus

        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                token_pattern=r'(?u)[^\s,.:;!?\"\'()\[\]{}।॥]+',
                min_df=1
            )
            tfidf_mat = vectorizer.fit_transform(all_texts)
            sims = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:])[0]

            best_idx = int(sims.argmax())
            best_sim = float(sims[best_idx])

            # Match threshold: semantic similarity or keyword overlap
            matched_entry = self.evidence_db[best_idx]
            claim_lower = claim_text.lower()
            keyword_matches = sum(1 for kw in matched_entry.get("keywords", []) if kw.lower() in claim_lower)

            if best_sim >= 0.18 or keyword_matches >= 2 or (keyword_matches >= 1 and best_sim >= 0.12):
                status = matched_entry.get("status", "Contradicted")
                badge = (
                    "🟢 Supported by Evidence" if status == "Supported"
                    else "🔴 Contradicted by Evidence" if status == "Contradicted"
                    else "⚪ Insufficient Evidence"
                )

                return {
                    "claim": claim_text,
                    "matched_claim": matched_entry["claim"],
                    "status": status,
                    "badge": badge,
                    "similarity_score": round(best_sim, 3),
                    "evidence_summary": matched_entry["verdict_explanation"],
                    "source_title": matched_entry["source"],
                    "source_url": matched_entry.get("source_url"),
                    "date": matched_entry.get("date"),
                    "disclaimer": "Matched against accredited fact-check archives."
                }

        except Exception:
            pass

        return {
            "claim": claim_text,
            "status": "Insufficient Evidence",
            "badge": "⚪ Insufficient Evidence",
            "similarity_score": 0.0,
            "evidence_summary": "No verified fact-check record found in the database. Independent manual verification is recommended.",
            "source_title": "Unverified Claim",
            "source_url": None,
            "date": None,
            "disclaimer": "Insufficient evidence does NOT imply the claim is false. It simply indicates that no matching verified fact-check was located."
        }

    def verify_article(self, text: str, headline: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extracts salient claims from headline and article body and verifies each against evidence.
        """
        claims = []
        if headline and len(headline.strip()) > 15:
            claims.append(headline.strip())

        body_claims = self.extract_salient_claims(text)
        for bc in body_claims:
            if bc not in claims and len(claims) < 4:
                claims.append(bc)

        results = []
        for claim in claims:
            results.append(self.verify_claim(claim))
        return results
