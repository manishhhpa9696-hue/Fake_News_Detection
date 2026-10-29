"""
Inference and Composite Prediction Engine.
Coordinates text featurization, calibrated probabilistic inference,
configurable uncertainty gating, clickbait scoring, domain credibility checking,
and explainable AI token attribution.
"""

import os
import json
from typing import Dict, Any, Optional
import joblib
import numpy as np

from .preprocessing import clean_text
from .multilingual import detect_language
from .clickbait_analysis import analyze_clickbait_and_sentiment
from .source_analysis import SourceAnalyzer
from .verification import ClaimVerifier
from .explainability import ModelExplainer
from .temporal_geo_analysis import analyze_temporal_and_geographic
from .database import save_prediction_record

DEFAULT_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models"
)


class NewsPredictor:
    """
    Production-grade inference pipeline combining calibrated ML models,
    probabilistic uncertainty thresholds, source credibility, and Explainable AI.
    """

    def __init__(
        self,
        models_dir: str = DEFAULT_MODELS_DIR,
        uncertainty_margin: float = 0.12,  # Window around 0.50 flagged as uncertain
    ):
        self.models_dir = models_dir
        self.uncertainty_margin = uncertainty_margin

        self.model = None
        self.vectorizer = None
        self.all_models = {}
        self.metadata = {}

        self._load_artifacts()
        self.source_analyzer = SourceAnalyzer()
        self.claim_verifier = ClaimVerifier()
        self.explainer = ModelExplainer()

    def _load_artifacts(self) -> None:
        """Loads serialized model and vectorizer from disk."""
        model_path = os.path.join(self.models_dir, "best_model.joblib")
        vec_path = os.path.join(self.models_dir, "vectorizer.joblib")
        all_path = os.path.join(self.models_dir, "all_models.joblib")
        meta_path = os.path.join(self.models_dir, "metadata.json")

        if os.path.exists(model_path) and os.path.exists(vec_path):
            try:
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vec_path)
                if os.path.exists(all_path):
                    self.all_models = joblib.load(all_path)
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
            except Exception as e:
                print(f"[!] Warning: Error loading model artifacts: {e}")

    def predict_article(
        self,
        text: str,
        headline: Optional[str] = None,
        url: Optional[str] = None,
        model_name: Optional[str] = None,
        uncertainty_threshold: Optional[float] = None,
        save_to_history: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end analysis on news input:
        1. Language detection
        2. Text cleaning & normalization
        3. Calibrated probability calculation
        4. Uncertainty classification
        5. Clickbait & sensationalism audit
        6. News source analysis
        7. Claim verification with evidence
        8. Explainable AI word importance
        9. History persistence
        """
        raw_text = (text or "").strip()
        raw_headline = (headline or "").strip()

        if not raw_text and not raw_headline:
            return {
                "success": False,
                "error": "Please provide headline or article body text for analysis.",
            }

        # 1. Multilingual script detection
        combined_raw = f"{raw_headline}. {raw_text}".strip()
        lang_info = detect_language(combined_raw)
        lang_code = lang_info["code"]

        # 2. Text preprocessing
        cleaned_text = clean_text(combined_raw, lang=lang_code, remove_stopwords=False)

        # 3. Model selection & probability prediction
        active_model = self.model
        if model_name and model_name in self.all_models:
            active_model = self.all_models[model_name]
        selected_model_name = model_name or self.metadata.get("champion_model_name", "Primary Model")

        fake_prob = 0.5
        reliable_prob = 0.5

        if active_model is not None and self.vectorizer is not None:
            try:
                X_vec = self.vectorizer.transform([cleaned_text])
                if hasattr(active_model, "predict_proba"):
                    probs = active_model.predict_proba(X_vec)[0]
                    reliable_prob = float(probs[0])
                    fake_prob = float(probs[1])
                elif hasattr(active_model, "decision_function"):
                    # For non-calibrated models, pass decision score through sigmoid
                    score = float(active_model.decision_function(X_vec)[0])
                    fake_prob = 1.0 / (1.0 + np.exp(-score))
                    reliable_prob = 1.0 - fake_prob
            except Exception as e:
                print(f"[!] Prediction error: {e}")

        # 4. Uncertainty evaluation
        margin = uncertainty_threshold if uncertainty_threshold is not None else self.uncertainty_margin
        lower_bound = 0.5 - margin
        upper_bound = 0.5 + margin

        is_uncertain = bool(lower_bound <= fake_prob <= upper_bound)

        if is_uncertain:
            prediction_label = "Uncertain — Needs Human Review"
            prediction_badge = "⚠️ Uncertain — Needs Human Review"
            # Confidence is low / distance from boundary is small
            confidence_pct = round(max(fake_prob, reliable_prob) * 100, 1)
        elif fake_prob > upper_bound:
            prediction_label = "Fake-like News"
            prediction_badge = "🔴 Fake-like Content"
            confidence_pct = round(fake_prob * 100, 1)
        else:
            prediction_label = "Reliable-like News"
            prediction_badge = "🟢 Reliable-like Content"
            confidence_pct = round(reliable_prob * 100, 1)

        # 5. Clickbait and emotional analysis
        clickbait_result = analyze_clickbait_and_sentiment(
            headline=raw_headline,
            body=raw_text,
        )

        # 6. Source analysis
        source_result = self.source_analyzer.analyze_source(
            url_or_domain=url or "",
        )

        # 7. Claim extraction and evidence verification
        claim_verifications = self.claim_verifier.verify_article(raw_text, headline=raw_headline)

        # 8. Temporal Specificity and Geographic Verification
        temporal_geo_result = analyze_temporal_and_geographic(raw_text, headline=raw_headline)

        # 9. Explainable AI: Feature weights & Highlighted text
        # Predict_proba closure for LIME
        def predict_proba_fn(texts):
            clean_batch = [clean_text(t, lang=lang_code, remove_stopwords=False) for t in texts]
            vec_batch = self.vectorizer.transform(clean_batch)
            if hasattr(active_model, "predict_proba"):
                return active_model.predict_proba(vec_batch)
            elif hasattr(active_model, "decision_function"):
                scores = active_model.decision_function(vec_batch)
                f_probs = 1.0 / (1.0 + np.exp(-scores))
                return np.vstack([1.0 - f_probs, f_probs]).T
            return np.tile([0.5, 0.5], (len(texts), 1))

        # Direct linear weights
        linear_weights = self.explainer.explain_with_linear_weights(
            cleaned_text,
            self.vectorizer,
            active_model,
            top_n=12,
        )

        # LIME local explanation
        lime_explanation = self.explainer.explain_with_lime(
            cleaned_text[:1200],  # Bound LIME perturbation size for low latency
            predict_proba_fn,
            num_features=10,
        )

        # Select weights for token highlighting
        display_weights = lime_explanation.get("features") or linear_weights
        highlighted_html = self.explainer.generate_highlighted_html(
            raw_text[:2500] if raw_text else raw_headline,
            display_weights,
        )

        # 10. Save to SQLite History
        record_id = None
        if save_to_history:
            try:
                record_id = save_prediction_record(
                    headline=raw_headline,
                    text=raw_text or raw_headline,
                    url=url,
                    domain=source_result.get("domain"),
                    language=lang_info["name"],
                    prediction=prediction_label,
                    confidence=confidence_pct,
                    uncertainty_flag=is_uncertain,
                    clickbait_score=clickbait_result.get("clickbait_score", 0.0),
                    temporal_score=temporal_geo_result["temporal"]["score"],
                    geo_score=temporal_geo_result["geographic"]["score"],
                    model_name=selected_model_name,
                )
            except Exception as e:
                print(f"[!] Failed to log prediction to history DB: {e}")

        return {
            "success": True,
            "record_id": record_id,
            "prediction": prediction_label,
            "badge": prediction_badge,
            "confidence": confidence_pct,
            "fake_probability": round(fake_prob * 100, 2),
            "reliable_probability": round(reliable_prob * 100, 2),
            "is_uncertain": is_uncertain,
            "uncertainty_margin": margin,
            "model_used": selected_model_name,
            "language": lang_info,
            "clickbait": clickbait_result,
            "source": source_result,
            "claim_verifications": claim_verifications,
            "temporal": temporal_geo_result["temporal"],
            "geographic": temporal_geo_result["geographic"],
            "grounding_index": temporal_geo_result["grounding_index"],
            "grounding_badge": temporal_geo_result["composite_badge"],
            "xai": {
                "feature_weights": display_weights,
                "highlighted_html": highlighted_html,
                "lime_meta": lime_explanation,
            },
            "disclaimer": (
                "Important: Statistical machine learning models detect structural and lexical patterns "
                "from historical training data; they cannot independently establish absolute factual truth. "
                "Always consult verified sources and cross-reference multiple reputable news agencies."
            ),
        }

    # Ergonomic alias for scikit-learn convention
    predict = predict_article
