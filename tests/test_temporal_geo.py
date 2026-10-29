"""
Unit Tests for Temporal Specificity and Geographic Verification.
"""

import pytest
from src.temporal_geo_analysis import (
    analyze_temporal_specificity,
    analyze_geographic_verification,
    analyze_temporal_and_geographic,
)


def test_temporal_specificity_high():
    text = (
        "On August 23, 2023 at 6:04 PM IST, the Indian Space Research Organisation "
        "confirmed that the Chandrayaan-3 lander made contact with the lunar surface."
    )
    result = analyze_temporal_specificity(text)
    assert result["score"] >= 70.0
    assert "Verified Calendar" in result["badge"] or "High Temporal Precision" in result["rating"]
    assert len(result["explicit_dates_found"]) > 0


def test_temporal_specificity_vague():
    text = (
        "Breaking viral message: Just now leaked reports reveal shocking truth from yesterday. "
        "In coming weeks all citizens must act immediately."
    )
    result = analyze_temporal_specificity(text)
    assert result["score"] <= 45.0
    assert len(result["vague_markers_found"]) > 0
    assert "Vague" in result["badge"] or "Vague" in result["rating"]


def test_geographic_verification_real():
    text = (
        "The scientific payload was assembled at Bengaluru and transported to the "
        "Satish Dhawan Space Centre in Sriharikota, Andhra Pradesh, for pre-launch checks."
    )
    result = analyze_geographic_verification(text)
    assert result["score"] >= 70.0
    assert "Real-World" in result["badge"] or "Strong" in result["rating"]
    assert any("Sriharikota" in loc or "Bengaluru" in loc for loc in result["detected_locations"])


def test_geographic_verification_suspicious():
    text = (
        "Whistleblower claims experimental chemicals are synthesized in a secret laboratory "
        "and an unnamed underground bunker before being transported to an undisclosed location."
    )
    result = analyze_geographic_verification(text)
    assert result["score"] <= 45.0
    assert len(result["suspicious_markers"]) > 0


def test_temporal_geo_combined():
    text = "On 15 January 2024, officials in New Delhi and Mumbai released economic guidelines."
    combined = analyze_temporal_and_geographic(text)
    assert combined["temporal"]["score"] >= 60.0
    assert combined["geographic"]["score"] >= 60.0
    assert combined["grounding_index"] >= 60.0
