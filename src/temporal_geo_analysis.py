"""
Temporal Specificity and Geographic Verification Module.
Analyzes articles for:
1. Temporal Specificity: Verifiable timestamps, explicit dates, and calendar anchors
   versus vague floating markers typical of recycled hoaxes.
2. Geographic Verification: Verifiable municipal, state, national, and scientific venues
   versus vague/fictitious locations.
"""

import re
from typing import Dict, Any, List

# Regex patterns for calendar dates (English, Hindi, Marathi)
DATE_PATTERNS = [
    # Explicit dates: "August 23, 2023", "23 August 2023", "23-08-2023", "2023/08/23"
    re.compile(r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b(?:\s*,?\s*\d{4})?', re.IGNORECASE),
    re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?\b', re.IGNORECASE),
    re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
    re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),
    # Standalone 4-digit years (1900 - 2035)
    re.compile(r'\b(?:19|20)\d{2}\b'),
    # Hindi/Marathi months and calendar anchors
    re.compile(r'\b\d{1,2}\s+(?:जनवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर|जानेवारी|फेब्रुवारी|मार्च|एप्रिल|मे|जून|जुलै|ऑगस्ट|सप्टेंबर|ऑक्टोबर|नोव्हेंबर|डिसेंबर)\b(?:\s+\d{4})?'),
]

# Precise time markers
TIME_PATTERNS = [
    re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|ist|gmt|utc)\b', re.IGNORECASE),
    re.compile(r'\b(?:सकाळी|दुपारी|संध्याकाळी|रात्री)\s*\d{1,2}(?::\d{2})?\s*वाजता\b'),
    re.compile(r'\b(?:सुबह|दोपहर|शाम|रात)\s*\d{1,2}(?::\d{2})?\s*बजे\b'),
]

# Vague / Floating Temporal Markers (common in recycled misinformation)
VAGUE_TEMPORAL_MARKERS = {
    # English
    "yesterday", "recently", "just now", "breaking news", "few days ago", "few hours ago",
    "last night", "in a few days", "next month", "tomorrow", "viral message", "secret report",
    "days ago", "coming weeks", "shocking video today",
    # Hindi
    "हाल ही में", "कल", "अभी-अभी", "कुछ दिन पहले", "ब्रेकिंग", "वायरल संदेश", "कल रात",
    "आने वाले दिनों में", "जल्द ही", "ताजा खबर",
    # Marathi
    "नुकतीच", "काल", "सध्या", "काही दिवसांपूर्वी", "ताजी बातमी", "व्हायरल मेसेज",
    "लवकरच", "पुढील महिन्यात", "काल रात्री"
}

# Recognized real-world geographic entities (Cities, States, Nations, Facilities)
KNOWN_GEO_ENTITIES = {
    # International Nations & Capitals
    "india", "united states", "usa", "uk", "united kingdom", "russia", "china",
    "france", "germany", "japan", "australia", "geneva", "london", "washington",
    "paris", "new york", "tokyo", "beijing", "berlin", "arizona", "sriharikota",
    # Indian Cities & States
    "new delhi", "delhi", "mumbai", "pune", "bengaluru", "bangalore", "hyderabad",
    "chennai", "kolkata", "ahmedabad", "jaipur", "lucknow", "maharashtra",
    "gujarat", "karnataka", "tamil nadu", "kerala", "uttar pradesh", "rajasthan",
    "madhya pradesh", "bihar", "punjab", "haryana", "odisha", "andhra pradesh",
    "goa", "kashmir", "ladakh", "sriharikota", "satish dhawan",
    # Hindi Devanagari Locations
    "भारत", "नई दिल्ली", "दिल्ली", "मुंबई", "पुणे", "श्रीहरिकोटा", "बेंगलुरु", "चेन्नई",
    "कोलकाता", "महाराष्ट्र", "गुजरात", "उत्तर प्रदेश", "राजस्थान", "मध्य प्रदेश",
    "बिहार", "पंजाब", "हरियाणा", "सतीश धवन", "अंतरिक्ष केंद्र",
    # Marathi Devanagari Locations
    "महाराष्ट्र", "मुंबई", "पुणे", "नागपूर", "नाशिक", "औरंगाबाद", "छत्रपती संभाजीनगर",
    "ठाणे", "कोकण", "गोवा", "श्रीहरिकोटा", "भारत", "नवी दिल्ली"
}

# Suspicious / Vague / Fictitious Location Cues
SUSPICIOUS_GEO_MARKERS = {
    # English
    "secret laboratory", "unnamed underground bunker", "undisclosed location",
    "somewhere in the mountains", "secret base", "hidden military compound",
    "an unnamed village", "unspecified country",
    # Hindi
    "गुप्त प्रयोगशाला", "अज्ञात स्थान", "रहस्यमयी गुफा", "खुफिया बंकर", "अनाम गांव",
    # Marathi
    "गुप्त तळ", "अज्ञात ठिकाण", "रहस्यमय जागा", "भूमिगत बंकर", "अनामिक गाव"
}


def analyze_temporal_specificity(text: str) -> Dict[str, Any]:
    """
    Evaluates how precisely anchored the article is in calendar time.
    High specificity -> exact dates, timestamps, verifiable years.
    Low specificity -> floating relative phrases ('recently', 'yesterday', 'just now').
    """
    if not text or not text.strip():
        return {
            "score": 0.0,
            "rating": "No Temporal Data",
            "badge": "⚪ Undated Content",
            "explicit_dates_found": [],
            "vague_markers_found": [],
            "explanation": "No text provided to evaluate temporal anchors.",
        }

    text_lower = text.lower()

    # Find explicit calendar dates
    found_dates: List[str] = []
    for pat in DATE_PATTERNS:
        matches = pat.findall(text)
        for m in matches:
            if isinstance(m, str) and len(m.strip()) > 1:
                found_dates.append(m.strip())

    # Find precise timestamps
    found_times: List[str] = []
    for pat in TIME_PATTERNS:
        matches = pat.findall(text)
        for m in matches:
            if isinstance(m, str) and len(m.strip()) > 1:
                found_times.append(m.strip())

    # Find vague/floating markers
    found_vague: List[str] = []
    for marker in VAGUE_TEMPORAL_MARKERS:
        if marker in text_lower or marker in text:
            found_vague.append(marker)

    # Calculate Temporal Specificity Score (0 to 100%)
    base_score = 30.0  # neutral starting point
    base_score += min(len(found_dates) * 25.0, 50.0)
    base_score += min(len(found_times) * 15.0, 20.0)
    # Penalty for floating markers without explicit dates
    if found_vague and not found_dates:
        base_score -= min(len(found_vague) * 15.0, 30.0)
    elif found_vague and found_dates:
        base_score += 5.0  # Contextual narrative anchor

    final_score = max(5.0, min(100.0, round(base_score, 1)))

    if final_score >= 70:
        rating = "High Temporal Precision"
        badge = "⏱️ Verified Calendar Anchoring"
        expl = f"Explicit calendar anchors detected ({len(found_dates)} date/year references), enabling historical verification."
    elif final_score >= 40:
        rating = "Moderate Temporal Anchoring"
        badge = "📅 General Timeframe Mentioned"
        expl = "Contains broad temporal context, but lacks exact calendar dates or official timestamps."
    else:
        rating = "Low / Floating Temporal Anchoring"
        badge = "⚠️ Vague / Recycled Hoax Risk"
        expl = (
            f"Relies heavily on floating relative terms ({', '.join(found_vague[:3]) if found_vague else 'none'}) "
            "without fixed calendar dates, a common pattern in evergreen recycled rumors."
        )

    return {
        "score": final_score,
        "rating": rating,
        "badge": badge,
        "explicit_dates_found": list(set(found_dates))[:5],
        "precise_times_found": list(set(found_times))[:3],
        "vague_markers_found": list(set(found_vague))[:5],
        "explanation": expl,
    }


def analyze_geographic_verification(text: str) -> Dict[str, Any]:
    """
    Evaluates geographic grounding and venue plausibility.
    Verifiable entities -> real cities, states, nations, research facilities.
    Suspicious markers -> 'secret bunker', 'undisclosed lab', vague unnamed locales.
    """
    if not text or not text.strip():
        return {
            "score": 0.0,
            "rating": "No Geographic Data",
            "badge": "⚪ Unlocated Content",
            "detected_locations": [],
            "suspicious_markers": [],
            "explanation": "No text provided to evaluate geographic entities.",
        }

    text_lower = text.lower()

    # 1. Detect verifiable locations
    found_locations: List[str] = []
    for entity in KNOWN_GEO_ENTITIES:
        # Check boundary match for Latin, direct match for Devanagari
        if re.search(r'[\u0900-\u097F]', entity):
            if entity in text:
                found_locations.append(entity)
        else:
            if re.search(r'\b' + re.escape(entity) + r'\b', text_lower):
                found_locations.append(entity.title())

    # 2. Detect suspicious / fictitious location phrases
    found_suspicious: List[str] = []
    for susp in SUSPICIOUS_GEO_MARKERS:
        if susp in text_lower or susp in text:
            found_suspicious.append(susp)

    # Calculate Geographic Grounding Score (0 to 100%)
    base_score = 30.0
    base_score += min(len(found_locations) * 25.0, 60.0)

    if found_suspicious:
        base_score -= min(len(found_suspicious) * 25.0, 50.0)

    final_score = max(5.0, min(100.0, round(base_score, 1)))

    if final_score >= 70:
        rating = "Strong Geographic Grounding"
        badge = "📍 Verifiable Real-World Locations"
        expl = f"Anchored in recognized real-world venues/regions: {', '.join(found_locations[:4])}."
    elif final_score >= 40:
        rating = "Broad Regional Mention"
        badge = "🌍 Broad Geographic Scope"
        expl = "Mentions broad regional or national context, but lacks municipal or venue-specific details."
    else:
        rating = "Unverifiable / Vague Geographic Origin"
        badge = "⚠️ Unverifiable / Mythical Locales"
        expl = (
            f"Lacks verifiable geography or employs suspicious fictitious location cues "
            f"({', '.join(found_suspicious) if found_suspicious else 'undisclosed location'})."
        )

    return {
        "score": final_score,
        "rating": rating,
        "badge": badge,
        "detected_locations": list(set(found_locations))[:6],
        "suspicious_markers": list(set(found_suspicious)),
        "explanation": expl,
    }


def analyze_temporal_and_geographic(text: str, headline: str = "") -> Dict[str, Any]:
    """
    Combined utility for temporal and geographic credibility auditing.
    """
    full_text = f"{headline}. {text}".strip()
    temporal = analyze_temporal_specificity(full_text)
    geographic = analyze_geographic_verification(full_text)

    # Composite Grounding Index (harmonic / balanced blend)
    composite_index = round((temporal["score"] * 0.5) + (geographic["score"] * 0.5), 1)

    return {
        "temporal": temporal,
        "geographic": geographic,
        "grounding_index": composite_index,
        "composite_badge": (
            "🧭 Highly Grounded (Time & Place)" if composite_index >= 70
            else "⚖️ Moderately Grounded" if composite_index >= 40
            else "⚠️ Low Grounding (High Misinformation Risk)"
        ),
    }
