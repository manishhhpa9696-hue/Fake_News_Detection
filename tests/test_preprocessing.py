"""
Unit Tests for Data Preprocessing and Leakage Prevention.
"""

import pytest
import pandas as pd
from src.preprocessing import clean_text, normalize_labels, normalize_label, prepare_dataset, stratified_leakage_free_split
from src.multilingual import detect_language, clean_multilingual_text


def test_clean_text_basic():
    raw = "BREAKING: Check this out at https://fake-news.xyz/story! <b>Astonishing</b> results."
    cleaned = clean_text(raw, lang="en", remove_stopwords=False)
    assert "https" not in cleaned
    assert "<b>" not in cleaned
    assert "astonishing" in cleaned.lower()


def test_multilingual_detection():
    hindi_sample = "भारतीय अंतरिक्ष अनुसंधान संगठन ने चंद्रयान मिशन शुरू किया।"
    marathi_sample = "महाराष्ट्रात नवीन शैक्षणिक धोरण लागू करण्यात आले आहे."
    english_sample = "The Federal Reserve raised interest rates by twenty-five basis points today."

    det_hi = detect_language(hindi_sample)
    assert det_hi["code"] == "hi"
    assert det_hi["supported"] is True

    det_mr = detect_language(marathi_sample)
    assert det_mr["code"] == "mr"
    assert det_mr["supported"] is True

    det_en = detect_language(english_sample)
    assert det_en["code"] == "en"
    assert det_en["supported"] is True


def test_label_normalization():
    df = pd.DataFrame({
        "text": ["article 1", "article 2", "article 3", "article 4"],
        "label": ["fake", "Real", "1", "0"]
    })
    norm_df = normalize_labels(df, "label")
    assert norm_df["label"].tolist() == [1, 0, 1, 0]
    # Test default argument
    norm_df_default = normalize_labels(df)
    assert norm_df_default["label"].tolist() == [1, 0, 1, 0]
    # Test single label helper
    assert normalize_label("fake") == 1
    assert normalize_label("Real") == 0
    assert normalize_label("false") == 1
    assert normalize_label(0) == 0


def test_stratified_leakage_free_split():
    # Construct a dataset with diverse texts and labels
    texts = [f"Unique verified article content number {i} reporting on world facts" for i in range(40)]
    labels = [0 if i % 2 == 0 else 1 for i in range(40)]
    df = pd.DataFrame({"text": texts, "label": labels})
    df = prepare_dataset(df, text_col="text", title_col=None, label_col="label")

    train_df, val_df, test_df = stratified_leakage_free_split(
        df,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=42
    )

    # Check zero overlap between splits
    train_hashes = set(train_df["content_hash"])
    val_hashes = set(val_df["content_hash"])
    test_hashes = set(test_df["content_hash"])

    assert len(train_hashes.intersection(val_hashes)) == 0
    assert len(train_hashes.intersection(test_hashes)) == 0
    assert len(val_hashes.intersection(test_hashes)) == 0
