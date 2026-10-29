"""
Unit tests for evaluation metrics and circular visualization charts.
"""

import numpy as np
import pytest
from src.evaluation import (
    evaluate_model_performance,
    create_confusion_matrix_figure,
    create_circular_metric_chart,
    create_circular_matrix_radar,
)


def test_evaluate_model_performance_basic():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.85, 0.3])

    metrics = evaluate_model_performance(y_true, y_pred, y_prob, model_name="TestModel")
    assert metrics["model_name"] == "TestModel"
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["roc_auc"] is not None


def test_create_circular_metric_chart():
    fig = create_circular_metric_chart(
        title="Fake Probability",
        value=82.4,
        color="#ef4444",
        icon="🔴",
        subtitle="High Risk",
    )
    assert fig is not None
    assert len(fig.data) == 1
    # Check that values are [82.4, 17.6]
    assert fig.data[0].values[0] == 82.4
    assert fig.data[0].hole == 0.74

    # Test title-less chart (for dashboard cards)
    fig_no_title = create_circular_metric_chart(
        title="",
        value=50.0,
        color="#10b981",
        subtitle="Neutral",
    )
    assert fig_no_title is not None
    assert fig_no_title.data[0].values[0] == 50.0
    assert fig_no_title.layout.title.text is None or fig_no_title.layout.title.text == ""


def test_create_circular_matrix_radar():
    fig = create_circular_matrix_radar(
        fake_prob=75.0,
        real_prob=25.0,
        temporal_score=80.0,
        geo_score=90.0,
        title="Test Radar",
    )
    assert fig is not None
    assert len(fig.data) == 1
    # Polar chart has 5 points (4 categories + closing loop)
    assert len(fig.data[0].r) == 5
    assert fig.data[0].r[0] == 75.0
