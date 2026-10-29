"""
Unit Tests for Source Credibility Analysis.
"""

import pytest
from src.source_analysis import SourceAnalyzer


def test_known_high_credibility_source():
    analyzer = SourceAnalyzer()
    res = analyzer.analyze_source("https://www.reuters.com/world/article")
    assert res["has_source_info"] is True
    assert res["domain"] == "reuters.com"
    assert "High Credibility" in res["credibility_tier"]
    assert res["is_satire"] is False


def test_known_satire_source():
    analyzer = SourceAnalyzer()
    res = analyzer.analyze_source("https://theonion.com/some-funny-post")
    assert res["has_source_info"] is True
    assert res["is_satire"] is True
    assert "Satire" in res["credibility_tier"]
    assert res["warning"] is not None


def test_unknown_domain_never_assumes_fake():
    analyzer = SourceAnalyzer()
    res = analyzer.analyze_source("https://independent-local-journal.org/report")
    assert res["has_source_info"] is True
    # Verify that unindexed source is NOT classified as fake
    assert res["is_satire"] is False
    assert "Unverified" in res["credibility_tier"]
    assert "does NOT mean the article is fake" in res["description"]


def test_empty_source():
    analyzer = SourceAnalyzer()
    res = analyzer.analyze_source("")
    assert res["has_source_info"] is False
