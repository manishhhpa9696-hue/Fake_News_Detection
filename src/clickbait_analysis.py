"""
Clickbait and Emotional Language Detection Module.
Detects sensational headlines, affective/polarizing language,
excessive capitalization/punctuation, and headline-body semantic mismatch.
"""

import re
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SENSATIONAL_KEYWORDS = {
    # English markers
    "shocking", "unbelievable", "mind blowing", "mind-blowing", "miracle", "secret revealed",
    "won't believe", "wont believe", "you won't believe", "will shock you", "exposed",
    "leaked", "conspiracy", "banned", "secret cure", "jaw dropping", "urgent warning",
    "destroyed", "slams", "blasts", "rips into", "viral video", "must watch", "hidden truth",
    "magic remedy", "pure poison", "miracle pill", "apocalypse", "panic", "nightmare",
    # Hindi/Marathi markers
    "धमाका", "चौंकाने वाला", "बड़ा खुलासा", "सावधान", "हाहाकार", "होश उड़ जाएंगे", "खुफिया",
    "वायरल सच", "चमत्कार", "गुप्त", "रहस्य", "हैरान", "सनसनीखेज", "धक्कादायक", "मोठा खुलासा"
}

EMOTIONAL_INTENSITY_TERMS = {
    "horrific", "catastrophic", "evil", "disaster", "furious", "terrified", "rage",
    "miraculous", "disgusting", "shameful", "hate", "scandalous", "traitor", "heroic",
    "खौफनाक", "भयानक", "गुस्सा", "गद्दार", "कहर", "त्रासदायक", "संतापजनक"
}


def compute_headline_body_similarity(headline: str, body: str) -> float:
    """
    Computes cosine similarity between headline and body text via TF-IDF.
    Returns score between 0.0 (no overlap / mismatch) and 1.0 (strong semantic coherence).
    """
    if not headline or not body or len(body.strip()) < 20:
        return 1.0  # Cannot evaluate mismatch without sufficient body text

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=r'(?u)[^\s,.:;!?\"\'()\[\]{}।॥]+',
            stop_words='english'
        )
        tfidf_matrix = vectorizer.fit_transform([headline, body])
        similarity = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        return max(0.0, min(1.0, similarity))
    except Exception:
        return 0.5


def analyze_clickbait_and_sentiment(headline: str, body: str = "") -> Dict[str, Any]:
    """
    Performs multidimensional clickbait analysis:
    1. Sensational lexical keywords
    2. Capitalization and exclamation punctuation patterns
    3. Emotional intensity terms
    4. Headline-body mismatch detection
    """
    headline_clean = headline.strip() if headline else ""
    body_clean = body.strip() if body else ""

    if not headline_clean and not body_clean:
        return {
            "clickbait_score": 0.0,
            "rating": "Insufficient Content",
            "sensational_words_found": [],
            "caps_ratio": 0.0,
            "punctuation_flags": False,
            "headline_body_similarity": 1.0,
            "mismatch_detected": False,
            "explanation": "No text provided for linguistic analysis."
        }

    target_headline = headline_clean if headline_clean else body_clean[:200]
    headline_lower = target_headline.lower()

    # 1. Check sensational phrases
    found_sensational: List[str] = []
    for term in SENSATIONAL_KEYWORDS:
        if term in headline_lower:
            found_sensational.append(term)

    # 2. Capitalization check (excluding short titles)
    letters = re.findall(r'[A-Za-z]', target_headline)
    caps = re.findall(r'[A-Z]', target_headline)
    caps_ratio = (len(caps) / len(letters)) if len(letters) >= 10 else 0.0

    # 3. Punctuation checks
    exclamations = target_headline.count("!")
    question_marks = target_headline.count("?")
    has_excessive_punctuation = exclamations >= 2 or (exclamations >= 1 and question_marks >= 1)

    # 4. Emotional intensity words
    found_emotional: List[str] = []
    for term in EMOTIONAL_INTENSITY_TERMS:
        if term in headline_lower or term in body_clean[:500].lower():
            found_emotional.append(term)

    # 5. Headline-body semantic consistency
    sim_score = compute_headline_body_similarity(target_headline, body_clean) if body_clean else 1.0
    mismatch_detected = bool(body_clean and len(body_clean) > 100 and sim_score < 0.08)

    # Calculate overall Clickbait Score (0 to 100%)
    base_score = 0.0
    base_score += min(len(found_sensational) * 25.0, 50.0)
    if caps_ratio > 0.45:
        base_score += 20.0
    elif caps_ratio > 0.30:
        base_score += 10.0

    if has_excessive_punctuation:
        base_score += 15.0

    if found_emotional:
        base_score += min(len(found_emotional) * 10.0, 20.0)

    if mismatch_detected:
        base_score += 25.0

    final_score = min(round(base_score, 1), 100.0)

    # Label classification
    if final_score >= 65:
        rating = "High Clickbait / Sensational Language"
        badge = "🔥 High Clickbait Signals"
    elif final_score >= 35:
        rating = "Moderate Clickbait / Emotional Tone"
        badge = "⚡ Moderate Sensationalism"
    else:
        rating = "Low Clickbait / Neutral Editorial Style"
        badge = "📰 Neutral / Factual Style"

    explanation_parts = []
    if found_sensational:
        explanation_parts.append(f"Sensational phrases detected: {', '.join(found_sensational[:4])}.")
    if caps_ratio > 0.35:
        explanation_parts.append(f"High uppercase letter ratio ({round(caps_ratio * 100, 1)}%).")
    if has_excessive_punctuation:
        explanation_parts.append("Excessive exclamation or question punctuation.")
    if mismatch_detected:
        explanation_parts.append("Significant semantic divergence between headline claims and article body.")
    if not explanation_parts:
        explanation_parts.append("Headline and text maintain sober, standard journalistic phrasing.")

    return {
        "clickbait_score": final_score,
        "rating": rating,
        "badge": badge,
        "sensational_words_found": found_sensational,
        "emotional_words_found": found_emotional,
        "caps_ratio": round(caps_ratio, 3),
        "punctuation_flags": has_excessive_punctuation,
        "headline_body_similarity": round(sim_score, 3),
        "mismatch_detected": mismatch_detected,
        "explanation": " ".join(explanation_parts),
        "disclaimer": "Linguistic sensation is a supplementary signal, not proof of factual falsity."
    }
