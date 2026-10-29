"""
Multilingual Processing Module
Supports English, Hindi, and Marathi text detection, script validation,
and language-specific text cleaning.
"""

import re
from typing import Dict, Tuple, List, Set

# Devanagari Unicode Block: \u0900 - \u097F
DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]')

# Common Hindi specific stop words and marker tokens
HINDI_MARKERS: Set[str] = {
    "है", "हैं", "नहीं", "कहा", "गया", "गई", "था", "थी", "थे", "ने", "को", "का",
    "की", "के", "में", "पर", "से", "और", "किया", "कि", "यह", "वह", "इस", "उस",
    "अपने", "अपनी", "होगा", "होगी", "रहा", "रही", "रहे", "सकते", "सकता", "लेकिन",
    "भारत", "समाचार", "दावा", "वायरल", "सोशल", "मीडिया", "सच", "झूठ"
}

# Common Marathi specific stop words and marker tokens
MARATHI_MARKERS: Set[str] = {
    "आहे", "आहेत", "नाही", "नाहीत", "म्हणाले", "झाले", "झाली", "केले", "केली",
    "यांनी", "त्यांनी", "होते", "होती", "त्यांच्या", "यांच्या", "करणार", "सांगितले",
    "घडली", "त्यांना", "यांना", "केंद्राने", "राज्यात", "मुंबई", "पुणे", "महाराष्ट्र",
    "माहिती", "दिला", "दिली", "केल्याचे", "असल्याचे", "समोर", "दावा", "खोट्या"
}

# English stop words (essential subset)
ENGLISH_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "of", "by", "from", "up", "about", "into", "through", "after", "before",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "this", "that", "these", "those", "it", "its", "they",
    "their", "them", "he", "him", "his", "she", "her", "we", "us", "our", "you",
    "your", "not", "so", "as", "if", "out", "all", "more"
}


def detect_language(text: str) -> Dict[str, str]:
    """
    Identifies if text is English, Hindi, Marathi, or an unsupported language/script.
    Returns dictionary with language code, display name, and confidence score.
    """
    if not text or not text.strip():
        return {
            "code": "en",
            "name": "English",
            "supported": True,
            "confidence": "Low (Empty text)",
            "script": "Latin",
        }

    sample = text[:1500]
    devanagari_chars = len(DEVANAGARI_PATTERN.findall(sample))
    latin_chars = len(re.findall(r'[a-zA-Z]', sample))
    total_chars = max(len(re.findall(r'\S', sample)), 1)

    devanagari_ratio = devanagari_chars / total_chars
    latin_ratio = latin_chars / total_chars

    # Devanagari script detected
    if devanagari_ratio > 0.15:
        # Check Marathi vs Hindi tokens
        words = set(re.findall(r'[\u0900-\u097F]+', sample))
        marathi_hits = len(words.intersection(MARATHI_MARKERS))
        hindi_hits = len(words.intersection(HINDI_MARKERS))

        if marathi_hits > hindi_hits:
            return {
                "code": "mr",
                "name": "Marathi (मराठी)",
                "supported": True,
                "confidence": f"{round((marathi_hits / max(len(words), 1)) * 100, 1)}% marker match",
                "script": "Devanagari",
            }
        else:
            return {
                "code": "hi",
                "name": "Hindi (हिंदी)",
                "supported": True,
                "confidence": f"{round((hindi_hits / max(len(words), 1)) * 100, 1)}% marker match",
                "script": "Devanagari",
            }

    # Latin script
    elif latin_ratio > 0.4:
        return {
            "code": "en",
            "name": "English",
            "supported": True,
            "confidence": f"{round(latin_ratio * 100, 1)}% Latin script",
            "script": "Latin",
        }

    # Other non-supported scripts (Arabic, Cyrillic, Chinese, etc.)
    else:
        return {
            "code": "unsupported",
            "name": "Unsupported Language / Mixed Script",
            "supported": False,
            "confidence": "Low script recognition",
            "script": "Other",
        }


def clean_multilingual_text(text: str, lang_code: str = "en") -> str:
    """
    Performs language-aware cleaning:
    - Normalizes spacing and removes URLs
    - Retains relevant script characters
    - Cleans punctuation appropriate for language (including danda । for Indic)
    """
    if not text:
        return ""

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Remove HTML markup tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)

    if lang_code in {"hi", "mr"}:
        # Retain Devanagari characters, numbers, and basic punctuation
        # Replace danda (।) and double danda (॥) with space or sentence boundaries
        text = text.replace("।", " ").replace("॥", " ")
        text = re.sub(r'[^\u0900-\u097F0-9\s]', ' ', text)
    else:
        # Standard Latin normalization
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_stopwords_for_lang(lang_code: str) -> Set[str]:
    """
    Returns appropriate stopword set for the given language.
    """
    if lang_code == "hi":
        return HINDI_MARKERS
    elif lang_code == "mr":
        return MARATHI_MARKERS.union({"आणि", "व", "किंवा", "पण", "तर", "हे", "ती", "तो"})
    else:
        return ENGLISH_STOPWORDS
