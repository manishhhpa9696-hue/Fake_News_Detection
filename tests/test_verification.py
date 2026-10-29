"""
Unit Tests for Claim Verification and Evidence Matching Engine.
"""

import pytest
from src.verification import ClaimVerifier


def test_claim_verifier_supported_english():
    verifier = ClaimVerifier()
    res = verifier.verify_claim("NASA confirms James Webb Telescope spots oldest galaxies ever observed")
    assert res["status"] == "Supported"
    assert "Nature Astronomy" in res["source_title"] or "NASA" in res["source_title"]
    assert res["similarity_score"] > 0.15


def test_claim_verifier_contradicted_english():
    verifier = ClaimVerifier()
    res = verifier.verify_claim("Boiled lemon hot water destroys all cancer cells and viruses immediately")
    assert res["status"] == "Contradicted"
    assert "Snopes" in res["source_title"] or "American Cancer Society" in res["source_title"]


def test_claim_verifier_hindi_supported():
    verifier = ClaimVerifier()
    res = verifier.verify_claim("इसरो ने मौसम उपग्रह इनसैट-3डीएस का सफल प्रक्षेपण किया")
    assert res["status"] == "Supported"
    assert "ISRO" in res["source_title"]


def test_claim_verifier_hindi_contradicted():
    verifier = ClaimVerifier()
    res = verifier.verify_claim("नमक की भारी कमी होने वाली है और नमक की कीमत हजार रुपये प्रति किलो हो जाएगी")
    assert res["status"] == "Contradicted"
    assert "PIB" in res["source_title"] or "Alt News" in res["source_title"]


def test_claim_verifier_marathi_supported():
    verifier = ClaimVerifier()
    res = verifier.verify_claim("भारतीय अंतराळ संशोधन संस्थेने चंद्रयान ३ मोहीम यशस्वीपणे पूर्ण केली")
    assert res["status"] == "Supported"
    assert "ISRO" in res["source_title"] or "बूम" in res["source_title"]


def test_claim_verifier_marathi_contradicted():
    verifier = ClaimVerifier()
    res = verifier.verify_claim("दोन हजार रुपयांच्या नव्या नोटेत नॅनो जीपीएस चिप बसवण्यात आली आहे")
    assert res["status"] == "Contradicted"
    assert "RBI" in res["source_title"] or "बूम" in res["source_title"]


def test_claim_verifier_insufficient_evidence():
    verifier = ClaimVerifier()
    res = verifier.verify_claim("Local committee reviewed municipal water distribution schedule for next Tuesday")
    assert res["status"] == "Insufficient Evidence"
    assert res["similarity_score"] == 0.0
    assert "Insufficient evidence does NOT imply the claim is false" in res["disclaimer"]
