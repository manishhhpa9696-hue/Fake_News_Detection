"""
Data Preprocessing and Leakage Prevention Module.
Handles text cleaning, token normalization, duplicate purging,
and strict stratified data partitioning before feature extraction.
"""

import re
import hashlib
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from sklearn.model_selection import train_test_split
from .multilingual import clean_multilingual_text, detect_language, get_stopwords_for_lang


def hash_text(text: str) -> str:
    """Calculates SHA-256 hash of normalized text for exact duplicate detection."""
    normalized = re.sub(r'\s+', ' ', (text or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clean_text(text: str, lang: str = "en", remove_stopwords: bool = True) -> str:
    """
    Standardizes raw text:
    - Removes URLs, HTML tags, and non-alphanumeric characters
    - Normalizes case and spacing
    - Optionally removes language-specific stopwords
    """
    if not text or not isinstance(text, str):
        return ""

    cleaned = clean_multilingual_text(text, lang_code=lang)

    if remove_stopwords:
        stopwords = get_stopwords_for_lang(lang)
        tokens = cleaned.split()
        tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
        cleaned = " ".join(tokens)

    return cleaned


def normalize_label(label: Any) -> int:
    """
    Standardizes a single classification label to binary integer:
    1 -> 'Fake-like' (Misinformation / Unreliable)
    0 -> 'Reliable-like' (Verified / Reliable)
    """
    label_map = {
        "fake": 1, "1": 1, 1: 1, 1.0: 1, "false": 1, "unreliable": 1, "spam": 1,
        "real": 0, "0": 0, 0: 0, 0.0: 0, "true": 0, "reliable": 0, "factual": 0
    }
    s = str(label).strip().lower()
    return label_map.get(s, 1 if "fake" in s or "false" in s else 0)


def normalize_labels(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    Standardizes classification labels to binary integers:
    1 -> 'Fake-like' (Misinformation / Unreliable)
    0 -> 'Reliable-like' (Verified / Reliable)
    """
    df = df.copy()
    label_map = {
        # Fake mappings
        "fake": 1, "1": 1, 1: 1, 1.0: 1, "false": 1, "unreliable": 1, "spam": 1,
        # Reliable mappings
        "real": 0, "0": 0, 0: 0, 0.0: 0, "true": 0, "reliable": 0, "factual": 0
    }

    # Normalize column values to lower-case string for matching
    df[label_col] = df[label_col].astype(str).str.strip().str.lower().map(
        lambda x: label_map.get(x, label_map.get(str(int(float(x))) if x.replace(".", "", 1).isdigit() else x, None))
    )

    # Drop unmapped or null labels
    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)
    return df


def prepare_dataset(
    df: pd.DataFrame,
    text_col: str = "text",
    title_col: Optional[str] = "title",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Cleans, deduplicates, and structures input raw dataset:
    - Merges title and text if title is provided
    - Purges exact and near duplicates across the dataset to prevent split leakage
    - Normalizes target labels
    """
    df = df.copy()

    # Verify column existence
    if text_col not in df.columns:
        raise ValueError(f"Required text column '{text_col}' not found in dataframe.")
    if label_col not in df.columns:
        raise ValueError(f"Required label column '{label_col}' not found in dataframe.")

    # Drop nulls in core fields
    df = df.dropna(subset=[text_col, label_col])

    # Normalize labels
    df = normalize_labels(df, label_col)

    # Create combined full text
    if title_col and title_col in df.columns:
        df["title"] = df[title_col].fillna("").astype(str)
        df["combined_text"] = df["title"] + ". " + df[text_col].astype(str)
    else:
        df["combined_text"] = df[text_col].astype(str)

    # Filter out empty or whitespace-only articles
    df["combined_text"] = df["combined_text"].str.strip()
    df = df[df["combined_text"].str.len() > 20]

    # Deduplication using cryptographic hash of normalized content
    df["content_hash"] = df["combined_text"].apply(hash_text)
    initial_len = len(df)
    df = df.drop_duplicates(subset=["content_hash"]).reset_index(drop=True)
    dedup_removed = initial_len - len(df)

    # Add cleaned text field with language-aware cleaning
    def _clean_article(row):
        t = row["combined_text"]
        lang = row.get("language")
        if not lang or pd.isna(lang) or lang == "unsupported":
            lang = detect_language(t)["code"]
        return clean_text(t, lang=lang, remove_stopwords=False)

    df["clean_text"] = df.apply(_clean_article, axis=1)

    return df


def stratified_leakage_free_split(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "label",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataframe into Train, Validation, and Test partitions.
    Uses stratification on target labels and verifies zero overlap across hashes.
    """
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-5, "Split proportions must sum to 1.0"

    # Step 1: Separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state,
    )

    # Step 2: Separate train and validation sets
    relative_val_size = val_size / (train_size + val_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df[label_col],
        random_state=random_state,
    )

    # Leakage check: Verify no overlapping content hashes
    if "content_hash" in df.columns:
        train_hashes = set(train_df["content_hash"])
        val_hashes = set(val_df["content_hash"])
        test_hashes = set(test_df["content_hash"])

        leakage_val = train_hashes.intersection(val_hashes)
        leakage_test = train_hashes.intersection(test_hashes)

        if leakage_val or leakage_test:
            raise RuntimeError(
                f"Data leakage detected! {len(leakage_val)} hashes in Val and "
                f"{len(leakage_test)} hashes in Test exist in Train."
            )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
