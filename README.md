# Veritas AI: Advanced Fake News Detection System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-red.svg)](https://streamlit.io/)
[![XAI](https://img.shields.io/badge/XAI-LIME%20%26%20SHAP-orange.svg)](https://github.com/marcotcr/lime)

An enterprise-grade, professional Fake News Detection and Credibility Analysis platform engineered with **Machine Learning**, **Explainable AI (XAI)**, **Multilingual NLP (English, Hindi, Marathi)**, **Calibrated Uncertainty Detection**, **SSRF-Protected News Scraping**, and **Claim Evidence Verification**.

---

## 📑 Table of Contents
1. [Core Features](#-core-features)
2. [Architecture Overview](#-architecture-overview)
3. [Project Structure](#-project-structure)
4. [Installation & Setup](#-installation--setup)
5. [Training & Model Evaluation](#-training--model-evaluation)
6. [Launching the Web Application](#-launching-the-web-application)
7. [Running Unit Tests](#-running-unit-tests)
8. [Security & SSRF Mitigation](#-security--ssrf-mitigation)
9. [Ethical Considerations & Limitations](#-ethical-considerations--limitations)
10. [Future Roadmap](#-future-roadmap)

---

## 🌟 Core Features

### 1. 🧠 Explainable AI (XAI)
- **Token-Level Attribution**: Highlights words and phrases influencing the prediction directly in the article text (Red for fake-indicative terms, Green for reliable-indicative terms).
- **LIME Local Surrogates**: Fits local interpretable models (`LimeTextExplainer`) to quantify exact feature influence scores.
- **Interactive Visualizations**: Dynamic Plotly bar charts of positive and negative term contributions.
- **Critical XAI Disclaimer**: Explains that highlighted terms reflect mathematical correlations in training data, not autonomous factual truth.

### 2. 🌐 News Source Credibility Analysis
- **Domain Reputation Directory**: Curated database of verified mainstream news outlets, certified fact-checkers (IFCN signatories), and known satire/parody sources.
- **Unverified Source Policy**: Unindexed domains are flagged for human review and **never** automatically branded as fake without evidence.
- **Safe Metadata Extraction**: Pulls publisher names, author bylines, and publication dates via safe HTML parsing.

### 3. 🇮🇳 Multilingual Fake News Detection
- **Languages Supported**: **English**, **Hindi (हिंदी)**, and **Marathi (मराठी)**.
- **Script & Marker Detection**: Unicode-based Devanagari script detection (`\u0900-\u097F`) and lexical marker disambiguation between Hindi and Marathi.
- **Language-Aware Preprocessing**: Preserves language-specific punctuation (such as the danda `।` and double danda `॥`) and applies dedicated multilingual stopword lists.

### 4. 🛡️ Claim Extraction & Evidence Verification
- **Assertive Claim Extractor**: Segments article text and extracts declarative, statistical, and assertive sentences.
- **Semantic Evidence Matching**: Compares extracted claims against verified fact-check archives using TF-IDF cosine similarity.
- **Tri-State Verdicts**:
  - `🟢 Supported`: Corroborated by verified reporting.
  - `🔴 Contradicted`: Debunked by accredited fact-checkers (Alt News, Boom Live, Snopes, Reuters, WHO).
  - `⚪ Insufficient Evidence`: No matching fact-check in database. **Important:** Insufficient evidence does *not* imply the claim is false.

### 5. ⚠️ Calibrated Confidence & Uncertainty Gating
- **Probability Calibration**: Uses `CalibratedClassifierCV` (sigmoid scaling) to provide calibrated posterior probabilities.
- **Uncertainty Margin**: Low-confidence predictions near the decision boundary (e.g. $50\% \pm 12\%$) are flagged as `⚠️ Uncertain — Needs Human Review` rather than forced into binary categories.

### 6. ⚡ Clickbait & Emotional Language Detection
- **Sensationalism Index**: Identifies sensationalist trigger words (*miracle, shocking, secret revealed, exposed*).
- **Headline-Body Mismatch**: Computes semantic cosine similarity between the headline and body text to flag bait-and-switch tactics.
- **Affective Tone Scoring**: Detects extreme uppercase capitalization ratios and aggressive punctuation patterns (`!!`, `!?!`).

### 7. 📊 Multi-Model Benchmarking
- Compares **Logistic Regression**, **Calibrated Linear SVM**, **Random Forest**, and **Multinomial Naive Bayes**.
- Strict zero-leakage workflow: Data is stratified and deduplicated by SHA-256 hash **before** vectorization.
- Models are compared on the **Validation Set**, and the champion is evaluated **once** on the held-out **Test Set**.

### 8. 📜 Privacy-Preserving Audit History
- Stored locally in an embedded **SQLite** database (`data/news_history.db`).
- Preserves privacy: Articles are hashed with SHA-256 and only 200-character audit snippets are stored.
- Filter, search, delete individual records, or export logs to CSV.

---

## 🏛️ Architecture Overview

```
                      +-----------------------------+
                      |   User Input (Text / URL)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  SSRF Defense & URL Fetch   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Multilingual Classifier   |
                      |   (English / Hindi / MR)    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     TF-IDF Featurization    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Calibrated ML Engine     |
                      |  (Logistic / SVM / RF / NB) |
                      +--------------+--------------+
                                     |
                   +-----------------+-----------------+
                   |                                   |
                   v                                   v
        +---------------------+             +---------------------+
        |  Confidence >= Min  |             |  Within Uncertainty |
        |      Threshold      |             |        Margin       |
        +----------+----------+             +----------+----------+
                   |                                   |
        +----------v----------+             +----------v----------+
        |   Fake / Reliable   |             |     Uncertain —     |
        |     Prediction      |             | Needs Human Review  |
        +----------+----------+             +----------+----------+
                   |                                   |
                   +-----------------+-----------------+
                                     |
                                     v
                      +-----------------------------+
                      | Multi-Signal Verification   |
                      | - LIME & Token Highlights   |
                      | - Source Credibility Record |
                      | - Clickbait & Mismatch      |
                      | - Evidence Claim Checks     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      | Streamlit Web Dashboard UI  |
                      | & SQLite Audit Persistence  |
                      +-----------------------------+
```

---

## 📁 Project Structure

```
D:\Project ML\Mini Project\
├── app.py                      # Multi-page Streamlit Dashboard
├── requirements.txt            # Pinned dependency requirements
├── README.md                   # Comprehensive documentation
├── .env.example                # Configuration templates
├── .gitignore                  # Git exclusion rules
├── src/
│   ├── __init__.py             # Package initializer
│   ├── preprocessing.py        # Text cleaning, label normalization & leakage-free splits
│   ├── multilingual.py         # Hindi, Marathi, and English script processing
│   ├── security.py             # SSRF protection, private IP blocking & safe URL fetcher
│   ├── source_analysis.py      # Publisher domain reputation & IFCN verification
│   ├── clickbait_analysis.py   # Sensationalism & headline-body mismatch analysis
│   ├── verification.py         # Claim extraction & evidence matching engine
│   ├── explainability.py       # LIME, token highlighting & XAI visualizations
│   ├── evaluation.py           # Metrics calculation & Plotly chart generators
│   ├── database.py             # SQLite audit trail & history management
│   ├── train.py                # Zero-leakage model training & benchmark pipeline
│   └── predict.py              # Unified inference engine
├── data/
│   ├── news_dataset.csv        # Seed multilingual training benchmark dataset
│   ├── reputable_sources.json  # Publisher credibility database
│   └── verified_claims.json    # Fact-check evidence repository
├── models/
│   ├── best_model.joblib       # Serialized champion model
│   ├── vectorizer.joblib       # Fitted TF-IDF vectorizer
│   ├── all_models.joblib       # Dictionary of all benchmarked candidate models
│   ├── evaluation_metrics.json # Full benchmark scores & confusion matrix
│   └── metadata.json           # Model vocabulary and training metadata
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py   # Tests for cleaning, normalization, and splits
    ├── test_security.py        # Tests for SSRF defense and IP validation
    ├── test_source_analysis.py # Tests for domain lookups and unverified handling
    └── test_predict.py         # Tests for inference, uncertainty gating, and clickbait
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone or Open Project Directory
```powershell
cd "D:\Project ML\Mini Project"
```

### 2. Initialize Virtual Environment
On Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required NLTK Corpora
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## 🎯 Training & Model Evaluation

To train all benchmark models, execute the training pipeline:
```bash
python -m src.train
```

### Output:
- Splits data into **70% Train**, **15% Validation**, and **15% Test** partitions.
- Fits `TfidfVectorizer` **exclusively on the Train partition**.
- Trains and evaluates **Logistic Regression**, **Calibrated Linear SVM**, **Random Forest**, and **Multinomial Naive Bayes**.
- Selects the best performing model based on **Validation Macro F1-score**.
- Evaluates the champion model **once on the held-out Test set**.
- Serializes trained artifacts into the `models/` directory.

---

## 🚀 Launching the Web Application

To launch the interactive multi-page Streamlit web dashboard:

```bash
streamlit run app.py
```

The application will open in your browser at:
`http://localhost:8501`

### Pages Included:
1. **🏠 Home & Overview**: System metrics, quick-start sample loaders, and architecture.
2. **🔍 News Analyzer**: Input text, headline, or public URL for end-to-end evaluation.
3. **🧠 Explainable AI (XAI)**: LIME local surrogates, highlighted text, and feature importance charts.
4. **🛡️ Claim Verification**: Salient claim extraction and fact-checking evidence comparison.
5. **📊 Model Benchmarks**: Comparative metric tables, ROC curves, and confusion matrices.
6. **📜 Analysis History**: Searchable SQLite audit log with CSV export and record management.
7. **ℹ️ About & Limitations**: System limits, ethical considerations, and security details.

---

## 🧪 Running Unit Tests

Execute the complete test suite using `pytest`:

```bash
python -m pytest -v tests
```

### Test Coverage:
- **`test_preprocessing.py`**: Validates text cleaning, Devanagari script processing, label normalization, and zero-leakage split verification.
- **`test_security.py`**: Verifies that SSRF defense blocks `127.0.0.1`, `localhost`, `10.0.0.0/8`, `192.168.0.0/16`, AWS metadata `169.254.169.254`, and illegal URI schemes (`file://`, `ftp://`).
- **`test_source_analysis.py`**: Ensures reputable sources are recognized, satire is flagged, and unindexed domains are not falsely condemned.
- **`test_predict.py`**: Verifies end-to-end inference, clickbait scoring, and uncertainty threshold triggers.

---

## 🔒 Security & SSRF Mitigation

When scraping article text from user-provided URLs, the system applies layered defense mechanisms in `src/security.py`:
1. **Scheme Whitelisting**: Only `http` and `https` protocols are permitted.
2. **DNS Resolution Verification**: Resolves the hostname before issuing HTTP requests to verify it points to a publicly routable IP address.
3. **Private IP Blocking**: Blocks all loopback (`127.0.0.0/8`), private (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), link-local (`169.254.0.0/16`), multicast, and IPv6 equivalents (`::1`, `fc00::/7`).
4. **Redirect Validation**: Followed HTTP redirects are re-validated against the SSRF rule set.
5. **Payload Bounds**: Enforces strict download limits (maximum 2 MB) and text size caps (50,000 characters) to prevent memory exhaustion and DoS attacks.

---

## ⚖️ Ethical Considerations & Limitations

- **Statistical Correlation vs. Truth**: The machine learning model detects lexical and structural patterns resembling known misinformation. It **cannot verify physical or historical facts independently**.
- **Source Nuance**: An unverified publisher domain may publish accurate breaking news, and an established news organization can make factual errors. Source credibility is an auxiliary indicator, not a definitive verdict.
- **Tone vs. Accuracy**: Clickbait headlines are common across digital media and do not automatically indicate falsity. Clickbait detection is treated as a supporting signal.
- **Claim Verification Scope**: The claim verification engine is grounded in accredited fact-checking archives. The absence of a debunking record (*Insufficient Evidence*) does **not** prove a claim is false.

---

## 🔮 Future Roadmap

- **Multimodal Misinformation**: Reverse image search and deepfake audio/video detection.
- **Real-Time Fact-Check APIs**: Integration with Google Fact Check Tools API and live IFCN ClaimReview feeds.
- **Graph Neural Networks**: Analyzing misinformation propagation dynamics across social media networks.
- **Full Transformer Fine-Tuning**: Optional fine-tuning of `XLM-RoBERTa` or `IndicBERT` for deeper contextual semantics across additional Indian regional languages (Tamil, Telugu, Bengali).

---

## 📄 License
This project is licensed under the Apache 2.0 License.
