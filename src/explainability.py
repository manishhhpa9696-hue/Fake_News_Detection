"""
Explainable AI (XAI) Module.
Provides LIME local surrogate explanations, model feature weight attribution,
interactive token highlighting, and Plotly visual explainability charts.
"""

import re
import html
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import plotly.graph_objects as go
from lime.lime_text import LimeTextExplainer


class ModelExplainer:
    """
    Manages Explainable AI routines for text classification models,
    including LIME local explanations and linear coefficient attribution.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        if class_names is None:
            self.class_names = ["Reliable-like", "Fake-like"]
        else:
            self.class_names = class_names

        # Initialize LIME text explainer with whitespace splitting for multilingual Unicode word preservation
        self.lime_explainer = LimeTextExplainer(
            class_names=self.class_names,
            bow=True,
            split_expression=r'\s+',
            random_state=42,
        )

    def explain_with_lime(
        self,
        text: str,
        predict_proba_fn,
        num_features: int = 10,
    ) -> Dict[str, Any]:
        """
        Generates local explanation using LIME surrogate model.
        Returns list of (feature, weight) tuples and explanation metadata.
        """
        if not text or not text.strip():
            return {
                "features": [],
                "intercept": 0.0,
                "score": 0.0,
                "disclaimer": "Insufficient text for LIME explanation.",
            }

        try:
            # LIME perturbation
            exp = self.lime_explainer.explain_instance(
                text_instance=text,
                classifier_fn=predict_proba_fn,
                num_features=num_features,
                labels=[1],  # Focus on class 1: Fake-like
            )

            # Extract weights for class 1
            # Positive weight -> pushes toward Fake-like (Class 1)
            # Negative weight -> pushes toward Reliable-like (Class 0)
            feature_weights = exp.as_list(label=1)

            return {
                "features": feature_weights,
                "intercept": float(exp.intercept.get(1, 0.0)) if hasattr(exp, "intercept") else 0.0,
                "score": float(exp.score) if hasattr(exp, "score") else 1.0,
                "disclaimer": (
                    "LIME features indicate local word influences on the model's prediction. "
                    "These highlights are mathematical correlations in training data, not proof of falsity."
                ),
            }
        except Exception as e:
            # Fallback if LIME fails on short text
            return {
                "features": [],
                "error": str(e),
                "disclaimer": "LIME explanation generation encountered an issue; fallback to global token weights.",
            }

    def explain_with_linear_weights(
        self,
        text: str,
        vectorizer,
        classifier,
        top_n: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Direct coefficient attribution for linear models (Logistic Regression / Linear SVM).
        Term Contribution = TF-IDF value * Model Weight
        """
        if not hasattr(classifier, "coef_"):
            return []

        try:
            vec = vectorizer.transform([text])
            feature_names = np.array(vectorizer.get_feature_names_out())
            coefs = classifier.coef_[0]

            # Non-zero indices in input sample
            nonzero_indices = vec.nonzero()[1]
            if len(nonzero_indices) == 0:
                return []

            contributions = []
            for idx in nonzero_indices:
                word = feature_names[idx]
                tfidf_val = vec[0, idx]
                weight = coefs[idx]
                impact = float(tfidf_val * weight)
                contributions.append((word, impact))

            # Sort by absolute impact
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            return contributions[:top_n]

        except Exception:
            return []

    def generate_highlighted_html(
        self,
        text: str,
        feature_weights: List[Tuple[str, float]],
    ) -> str:
        """
        Generates readable HTML snippet with color-coded token highlights:
        Red highlight for terms increasing Fake probability.
        Green highlight for terms increasing Reliable probability.
        """
        if not text:
            return ""

        weight_dict = {word.lower(): weight for word, weight in feature_weights}
        if not weight_dict:
            return f"<p style='line-height:1.7;'>{html.escape(text)}</p>"

        # Split text while preserving punctuation and spacing
        tokens = re.split(r'(\s+|[.,!?;:"()\[\]{}]+)', text)
        highlighted_parts = []

        for token in tokens:
            cleaned_token = token.strip().lower()
            if cleaned_token in weight_dict:
                weight = weight_dict[cleaned_token]
                # Scale opacity based on weight magnitude
                opacity = min(max(abs(weight) * 2.5, 0.15), 0.75)
                if weight > 0:
                    # Pushes toward Fake -> Coral / Red
                    bg_color = f"rgba(239, 68, 68, {opacity:.2f})"
                    border_color = "rgba(239, 68, 68, 0.8)"
                    text_color = "#fca5a5"
                    tooltip = f"Pushes toward Fake-like (Impact: +{weight:.3f})"
                else:
                    # Pushes toward Reliable -> Green / Emerald
                    bg_color = f"rgba(16, 185, 129, {opacity:.2f})"
                    border_color = "rgba(16, 185, 129, 0.8)"
                    text_color = "#6ee7b7"
                    tooltip = f"Pushes toward Reliable-like (Impact: {weight:.3f})"

                span = (
                    f"<span title='{tooltip}' style='background-color:{bg_color}; "
                    f"border-bottom: 2px solid {border_color}; color: {text_color}; padding: 2px 5px; "
                    f"border-radius: 4px; font-weight: 600; cursor: help;'>"
                    f"{html.escape(token)}</span>"
                )
                highlighted_parts.append(span)
            else:
                highlighted_parts.append(html.escape(token))

        html_result = "".join(highlighted_parts)
        return (
            f"<div style='line-height:1.9; font-size: 15px; color: #f1f5f9; "
            f"background: rgba(15, 23, 42, 0.85); padding: 20px; border-radius: 12px; "
            f"border: 1px solid rgba(255, 255, 255, 0.12); box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35); "
            f"max-height: 400px; overflow-y: auto;'>"
            f"{html_result}</div>"
        )

    def create_feature_importance_plot(
        self,
        feature_weights: List[Tuple[str, float]],
        title: str = "Most Influential Words (XAI Attribution)",
    ) -> go.Figure:
        """
        Plots a horizontal bar chart showing words pushing towards Fake (red) vs Reliable (green).
        """
        if not feature_weights:
            fig = go.Figure()
            fig.update_layout(
                title=dict(text="No prominent feature weights to display.", font=dict(color="#94a3b8")),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig

        # Sort features so most negative is at bottom, most positive at top
        sorted_weights = sorted(feature_weights, key=lambda x: x[1])
        words = [w[0] for w in sorted_weights]
        scores = [w[1] for w in sorted_weights]
        colors = ["#10b981" if s < 0 else "#ef4444" for s in scores]

        fig = go.Figure(
            go.Bar(
                x=scores,
                y=words,
                orientation="h",
                marker=dict(color=colors, line=dict(width=1, color="rgba(255,255,255,0.2)")),
                text=[f"{s:+.3f}" for s in scores],
                textposition="auto",
                textfont=dict(color="#ffffff", size=11, family="Inter, sans-serif"),
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=17, family="Inter, sans-serif", color="#f8fafc")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Inter, sans-serif"),
            xaxis=dict(
                title="← Pushes toward Reliable (Green) | Pushes toward Fake (Red) →",
                zeroline=True,
                zerolinecolor="#94a3b8",
                zerolinewidth=2,
                gridcolor="rgba(255, 255, 255, 0.08)",
                tickfont=dict(color="#94a3b8"),
            ),
            yaxis=dict(
                title="Token / Term",
                tickfont=dict(color="#e2e8f0"),
            ),
            height=380,
            margin=dict(l=80, r=40, t=60, b=50),
        )
        return fig
