"""
Model Evaluation and Performance Metrics Module.
Calculates Accuracy, Precision, Recall, Macro F1, ROC-AUC, and Confusion Matrices.
Generates publication-quality interactive charts via Plotly.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
import plotly.graph_objects as go
import plotly.express as px


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Computes rigorous classification metrics across binary and macro averages.
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_binary = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
    rec_binary = float(recall_score(y_true, y_pred, average="binary", zero_division=0))
    f1_binary = float(f1_score(y_true, y_pred, average="binary", zero_division=0))

    cm = confusion_matrix(y_true, y_pred).tolist()

    roc_auc = None
    fpr_list, tpr_list = None, None
    if y_prob is not None:
        try:
            # Handle 2D probability matrix or 1D array of positive class probs
            probs_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            roc_auc = float(roc_auc_score(y_true, probs_pos))
            fpr, tpr, _ = roc_curve(y_true, probs_pos)
            fpr_list = fpr.tolist()
            tpr_list = tpr.tolist()
        except Exception:
            roc_auc = None

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_fake": round(prec_binary, 4),
        "recall_fake": round(rec_binary, 4),
        "f1_fake": round(f1_binary, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm,
        "fpr": fpr_list,
        "tpr": tpr_list,
    }


def create_confusion_matrix_figure(
    cm: List[List[int]],
    labels: List[str] = ["Reliable-like (0)", "Fake-like (1)"],
    title: str = "Confusion Matrix",
) -> go.Figure:
    """
    Renders an interactive heatmap of the confusion matrix.
    """
    z = cm
    x = labels
    y = labels

    z_text = [[str(val) for val in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 16, "family": "Inter, Arial, sans-serif"},
            colorscale="Blues",
            colorbar=dict(title="Sample Count"),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, family="Inter, Arial, sans-serif", color="#f8fafc")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        xaxis=dict(title="Predicted Label", tickfont=dict(size=13, color="#94a3b8")),
        yaxis=dict(title="True Actual Label", tickfont=dict(size=13, color="#94a3b8"), autorange="reversed"),
        margin=dict(l=60, r=40, t=60, b=60),
        height=380,
    )
    return fig


def create_model_comparison_bar_chart(metrics_list: List[Dict[str, Any]]) -> go.Figure:
    """
    Creates a grouped bar chart comparing Accuracy, Precision, Recall, Macro F1 across models.
    """
    df = pd.DataFrame(metrics_list)
    models = df["model_name"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accuracy", x=models, y=df["accuracy"], marker_color="#6366f1"))
    fig.add_trace(go.Bar(name="Macro Precision", x=models, y=df["precision_macro"], marker_color="#38bdf8"))
    fig.add_trace(go.Bar(name="Macro Recall", x=models, y=df["recall_macro"], marker_color="#f59e0b"))
    fig.add_trace(go.Bar(name="Macro F1-Score", x=models, y=df["f1_macro"], marker_color="#10b981"))

    fig.update_layout(
        barmode="group",
        title=dict(text="Comparative Benchmark Metrics", x=0.5, font=dict(size=18, color="#f8fafc")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        yaxis=dict(
            title="Score (0.0 to 1.0)",
            range=[0.0, 1.05],
            tickformat=".2f",
            gridcolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#94a3b8"),
        ),
        xaxis=dict(title="Model Architecture", tickfont=dict(color="#e2e8f0")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#cbd5e1"),
        ),
        height=420,
        margin=dict(l=50, r=30, t=80, b=50),
    )
    return fig


def create_roc_curve_figure(
    models_roc_data: List[Dict[str, Any]],
    title: str = "Receiver Operating Characteristic (ROC) Curves",
) -> go.Figure:
    """
    Plots interactive ROC Curves for multiple trained models.
    """
    fig = go.Figure()

    # Diagonal baseline for random classifier
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#64748b", width=1.5),
            name="Random Chance (AUC = 0.50)",
        )
    )

    colors = ["#6366f1", "#38bdf8", "#10b981", "#f43f5e", "#a855f7"]

    for idx, model_data in enumerate(models_roc_data):
        fpr = model_data.get("fpr")
        tpr = model_data.get("tpr")
        auc_val = model_data.get("roc_auc", 0.0)
        name = model_data.get("model_name", f"Model {idx+1}")

        if fpr and tpr:
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    line=dict(color=colors[idx % len(colors)], width=2.5),
                    name=f"{name} (AUC = {auc_val:.3f})",
                )
            )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, color="#f8fafc")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        xaxis=dict(
            title="False Positive Rate (1 - Specificity)",
            range=[0, 1],
            gridcolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            title="True Positive Rate (Sensitivity / Recall)",
            range=[0, 1.02],
            gridcolor="rgba(255, 255, 255, 0.08)",
            tickfont=dict(color="#94a3b8"),
        ),
        legend=dict(
            x=0.55,
            y=0.15,
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            font=dict(color="#e2e8f0"),
        ),
        height=420,
        margin=dict(l=50, r=30, t=60, b=50),
    )
    return fig


def format_circular_title(title: str, icon: str = "") -> str:
    """
    Formats the metric title with clean multi-line wrapping (<br>)
    so that long titles never clip or cut off in narrow multi-column layouts.
    """
    prefix = f"{icon} " if icon else ""
    t = title.strip()

    known_splits = {
        "Fake News Detection": f"{prefix}Fake News<br>Detection",
        "Fake News Probability": f"{prefix}Fake News<br>Detection",
        "Real News Authenticity": f"{prefix}Real News<br>Authenticity",
        "Temporal Specificity": f"{prefix}Temporal<br>Specificity",
        "Geographic Verification": f"{prefix}Geographic<br>Verification",
        "Geographic Grounding": f"{prefix}Geographic<br>Verification",
    }
    if t in known_splits:
        return known_splits[t]

    words = t.split()
    if len(words) <= 1:
        return f"{prefix}{t}"
    elif len(words) == 2:
        return f"{prefix}{words[0]}<br>{words[1]}"
    elif len(words) == 3:
        return f"{prefix}{words[0]} {words[1]}<br>{words[2]}"
    else:
        mid = len(words) // 2
        return f"{prefix}{' '.join(words[:mid])}<br>{' '.join(words[mid:])}"


def create_circular_metric_chart(
    title: str = "",
    value: float = 0.0,
    color: str = "#6366f1",
    icon: str = "",
    subtitle: str = "",
) -> go.Figure:
    """
    Renders an interactive circular progress gauge (donut ring) for single metric presentation.
    Uses multi-line title wrapping and domain centering to prevent text clipping across screen sizes.
    """
    val = max(0.0, min(100.0, float(value)))
    remainder = round(100.0 - val, 2)

    has_title = bool(title and title.strip())
    display_title = format_circular_title(title, icon) if has_title else ""
    sub_text = f"<br><span style='font-size:11px; color:#94a3b8;'>{subtitle}</span>" if subtitle else ""

    pie_domain = dict(x=[0, 1], y=[0, 0.82]) if has_title else dict(x=[0, 1], y=[0, 1.0])
    annotation_y = 0.41 if has_title else 0.50

    fig = go.Figure(
        data=[
            go.Pie(
                values=[val, remainder],
                hole=0.74,
                marker=dict(colors=[color, "rgba(255, 255, 255, 0.08)"]),
                textinfo="none",
                hoverinfo="none",
                sort=False,
                direction="clockwise",
                showlegend=False,
                domain=pie_domain,
            )
        ]
    )

    fig.update_layout(
        height=210 if has_title else 170,
        margin=dict(l=6, r=6, t=46 if has_title else 8, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text=display_title,
            x=0.5,
            xanchor="center",
            y=0.94,
            yanchor="top",
            font=dict(size=12.5, color="#f1f5f9", family="Plus Jakarta Sans, Inter, sans-serif"),
        ) if has_title else None,
        annotations=[
            dict(
                text=f"<b style='font-size:22px; color:#ffffff;'>{int(round(val))}%</b>{sub_text}",
                x=0.5,
                y=annotation_y,
                showarrow=False,
                align="center",
                font=dict(size=20, color="#ffffff", family="Plus Jakarta Sans, Inter, sans-serif"),
            )
        ],
    )
    return fig


def create_circular_matrix_radar(
    fake_prob: float,
    real_prob: float,
    temporal_score: float,
    geo_score: float,
    title: str = "Multi-Signal Verification Matrix",
) -> go.Figure:
    """
    Generates a 4-dimensional circular polar radar chart linking model probabilities,
    temporal anchoring, and geographic grounding.
    """
    categories = [
        "Fake News Detection",
        "Real News Authenticity",
        "Temporal Specificity",
        "Geographic Verification",
    ]
    values = [
        float(fake_prob),
        float(real_prob),
        float(temporal_score),
        float(geo_score),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(99, 102, 241, 0.25)",
            line=dict(color="#818cf8", width=2.5),
            marker=dict(size=7, color="#6366f1"),
            name="Article Signals",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10, color="#94a3b8"),
                gridcolor="rgba(255, 255, 255, 0.1)",
                linecolor="rgba(255, 255, 255, 0.1)",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#e2e8f0"),
                gridcolor="rgba(255, 255, 255, 0.1)",
                linecolor="rgba(255, 255, 255, 0.1)",
            ),
            bgcolor="rgba(15, 23, 42, 0.4)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Plus Jakarta Sans, Inter, sans-serif"),
        title=dict(text=title, x=0.5, font=dict(size=16, color="#f8fafc")),
        height=320,
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=False,
    )
    return fig

