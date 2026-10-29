"""
Unit Tests for Prediction Engine, Uncertainty Gating, and Claims.
"""

import pytest
from src.predict import NewsPredictor
from src.clickbait_analysis import analyze_clickbait_and_sentiment


def test_predictor_basic_inference():
    predictor = NewsPredictor()
    result = predictor.predict_article(
        headline="Astronomers discover distant galaxy with James Webb Space Telescope",
        text="NASA researchers confirmed observational details regarding stellar evolution and redshift.",
        url="https://www.reuters.com/science/jwst",
        save_to_history=False,
    )
    assert result["success"] is True
    assert "prediction" in result
    assert "confidence" in result
    assert result["confidence"] >= 0.0
    assert result["source"]["domain"] == "reuters.com"

    # Test .predict() alias
    alias_res = predictor.predict(
        headline="Astronomers discover distant galaxy with James Webb Space Telescope",
        text="NASA researchers confirmed observational details regarding stellar evolution and redshift.",
        save_to_history=False,
    )
    assert alias_res["success"] is True
    assert alias_res["prediction"] == result["prediction"]


def test_uncertainty_threshold_flagging():
    predictor = NewsPredictor(uncertainty_margin=0.49)  # Wide margin forces uncertainty
    result = predictor.predict_article(
        headline="Ambiguous headline regarding market fluctuations",
        text="Traders observed mixed reactions across international commodities today.",
        save_to_history=False,
    )
    assert result["success"] is True
    assert result["is_uncertain"] is True
    assert "Uncertain" in result["prediction"]


def test_clickbait_detection():
    sensational_headline = "SHOCKING TRUTH REVEALED!! YOU WON'T BELIEVE THIS MIRACLE CURE NOW!!!"
    body = "Ordinary news about seasonal weather."
    analysis = analyze_clickbait_and_sentiment(headline=sensational_headline, body=body)
    assert analysis["clickbait_score"] > 40.0
    assert analysis["punctuation_flags"] is True
    assert len(analysis["sensational_words_found"]) > 0
