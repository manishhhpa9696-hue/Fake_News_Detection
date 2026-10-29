"""
Advanced Fake News Detection System - Streamlit Web Dashboard
Features: Explainable AI (LIME / Feature Attribution), Calibrated Uncertainty Gating,
Multilingual Support (English, Hindi, Marathi), Source Credibility Checking (SSRF-safe),
Clickbait & Affective Language Scoring, Claim Verification, and SQLite Audit History.
"""

import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.predict import NewsPredictor
from src.security import safe_fetch_url, validate_url_security
from src.source_analysis import SourceAnalyzer
from src.verification import ClaimVerifier
from src.explainability import ModelExplainer
from src.database import (
    get_history_records,
    delete_record_by_id,
    clear_all_history,
    get_history_summary_stats,
)
from src.evaluation import (
    create_confusion_matrix_figure,
    create_model_comparison_bar_chart,
    create_roc_curve_figure,
    create_circular_metric_chart,
    create_circular_matrix_radar,
)

# Set page config
st.set_page_config(
    page_title="Veritas AI - Advanced Fake News Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_predictor():
    """Caches predictor instance across user sessions."""
    return NewsPredictor()

predictor = load_predictor()


def render_stats_banner(stats: dict):
    """Renders the top summary metric cards across pages."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"<div class='card-metric' style='border-top: 3px solid #6366f1;'>"
            f"<div style='font-size:12px; font-weight:700; letter-spacing:1px; color:var(--text-secondary); text-transform:uppercase;'>TOTAL ANALYSES</div>"
            f"<div style='font-size:32px; font-weight:800; color:var(--text-primary); margin-top:6px;'>{stats['total_analyzed']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='card-metric' style='border-top: 3px solid #10b981;'>"
            f"<div style='font-size:12px; font-weight:700; letter-spacing:1px; color:#6ee7b7; text-transform:uppercase;'>RELIABLE CONTENT</div>"
            f"<div style='font-size:32px; font-weight:800; color:#34d399; margin-top:6px;'>{stats['reliable_count']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='card-metric' style='border-top: 3px solid #ef4444;'>"
            f"<div style='font-size:12px; font-weight:700; letter-spacing:1px; color:#fca5a5; text-transform:uppercase;'>FAKE-LIKE SIGNALS</div>"
            f"<div style='font-size:32px; font-weight:800; color:#f87171; margin-top:6px;'>{stats['fake_count']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"<div class='card-metric' style='border-top: 3px solid #f59e0b;'>"
            f"<div style='font-size:12px; font-weight:700; letter-spacing:1px; color:#fde68a; text-transform:uppercase;'>UNCERTAIN FLAGGED</div>"
            f"<div style='font-size:32px; font-weight:800; color:#fbbf24; margin-top:6px;'>{stats['uncertain_count']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# Sidebar Brand & Navigation
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 12px; padding: 4px 0 16px 0;">
        <div style="background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%); padding: 10px; border-radius: 12px; box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);">
            <span style="font-size: 26px;">🛡️</span>
        </div>
        <div>
            <div style="font-size: 19px; font-weight: 800; letter-spacing: -0.3px; color: #f8fafc;">Veritas AI</div>
            <div style="font-size: 12px; color: #38bdf8; font-weight: 500;">Multilingual Verification Engine</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home & Overview",
        "🔍 News Analyzer",
        "🧠 Explainable AI (XAI)",
        "🛡️ Claim Verification",
        "📊 Model Benchmarks",
        "📜 Analysis History",
        "ℹ️ About & Limitations",
    ],
)

st.sidebar.divider()

# Sidebar Theme Selector
st.sidebar.markdown("### 🎨 Visual Theme")
theme_choice = st.sidebar.selectbox(
    "Background & Aesthetic",
    [
        "🌌 Midnight Cyber (Cosmic Indigo & Neon)",
        "⚡ Electric Aurora (Deep Violet & Cyan)",
        "🌿 Obsidian Matrix (Deep Emerald & Jade)",
        "☀️ Pearl Minimalist (Clean Luminous Slate)",
    ],
    index=0,
)

st.sidebar.markdown("### ⚙️ Inference Settings")
selected_model = st.sidebar.selectbox(
    "Active Classifier",
    options=list(predictor.all_models.keys()) if predictor.all_models else ["Logistic Regression"],
    index=0,
)
uncertainty_slider = st.sidebar.slider(
    "Uncertainty Margin (±%)",
    min_value=0.05,
    max_value=0.25,
    value=0.12,
    step=0.01,
    help="Predictions with probability within 50% ± Margin will be categorized as 'Uncertain — Needs Human Review'.",
)

# Dynamic CSS Theme Variables based on user selection
if "Electric Aurora" in theme_choice:
    theme_vars = """
    :root {
        --bg-base: #090815;
        --card-bg: rgba(22, 18, 42, 0.75);
        --border-color: rgba(168, 85, 247, 0.22);
        --hover-border: rgba(168, 85, 247, 0.6);
        --glow-color: rgba(168, 85, 247, 0.28);
        --text-primary: #faf5ff;
        --text-secondary: #c084fc;
        --accent-gradient: linear-gradient(135deg, #7c3aed 0%, #ec4899 100%);
        --bg-mesh: 
            radial-gradient(circle at 8% 8%, rgba(168, 85, 247, 0.25) 0%, transparent 45%),
            radial-gradient(circle at 92% 12%, rgba(236, 72, 153, 0.22) 0%, transparent 45%),
            radial-gradient(circle at 50% 90%, rgba(6, 182, 212, 0.18) 0%, transparent 50%),
            radial-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px);
        --bg-grid-size: 100% 100%, 100% 100%, 100% 100%, 28px 28px;
        --sidebar-bg: linear-gradient(180deg, #070611 0%, #130f26 100%);
    }
    """
elif "Obsidian Matrix" in theme_choice:
    theme_vars = """
    :root {
        --bg-base: #040907;
        --card-bg: rgba(10, 26, 20, 0.75);
        --border-color: rgba(16, 185, 129, 0.22);
        --hover-border: rgba(16, 185, 129, 0.6);
        --glow-color: rgba(16, 185, 129, 0.25);
        --text-primary: #ecfdf5;
        --text-secondary: #6ee7b7;
        --accent-gradient: linear-gradient(135deg, #059669 0%, #0d9488 100%);
        --bg-mesh: 
            radial-gradient(circle at 10% 12%, rgba(16, 185, 129, 0.24) 0%, transparent 45%),
            radial-gradient(circle at 88% 15%, rgba(20, 184, 166, 0.20) 0%, transparent 45%),
            radial-gradient(circle at 50% 88%, rgba(5, 150, 105, 0.16) 0%, transparent 50%),
            radial-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px);
        --bg-grid-size: 100% 100%, 100% 100%, 100% 100%, 28px 28px;
        --sidebar-bg: linear-gradient(180deg, #030705 0%, #081611 100%);
    }
    """
elif "Pearl Minimalist" in theme_choice:
    theme_vars = """
    :root {
        --bg-base: #f8fafc;
        --card-bg: rgba(255, 255, 255, 0.95);
        --border-color: rgba(226, 232, 240, 0.9);
        --hover-border: rgba(99, 102, 241, 0.5);
        --glow-color: rgba(99, 102, 241, 0.12);
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --accent-gradient: linear-gradient(135deg, #4f46e5 0%, #0ea5e9 100%);
        --bg-mesh: 
            radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.08) 0%, transparent 45%),
            radial-gradient(circle at 90% 15%, rgba(14, 165, 233, 0.08) 0%, transparent 45%),
            radial-gradient(circle at 50% 90%, rgba(168, 85, 247, 0.05) 0%, transparent 50%),
            radial-gradient(rgba(0, 0, 0, 0.04) 1px, transparent 1px);
        --bg-grid-size: 100% 100%, 100% 100%, 100% 100%, 26px 26px;
        --sidebar-bg: #ffffff;
    }
    """
else:
    # Midnight Cyber (Default)
    theme_vars = """
    :root {
        --bg-base: #070a13;
        --card-bg: rgba(15, 23, 42, 0.72);
        --border-color: rgba(99, 102, 241, 0.22);
        --hover-border: rgba(99, 102, 241, 0.6);
        --glow-color: rgba(99, 102, 241, 0.25);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --accent-gradient: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        --bg-mesh: 
            radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.24) 0%, transparent 45%),
            radial-gradient(circle at 90% 12%, rgba(14, 165, 233, 0.22) 0%, transparent 45%),
            radial-gradient(circle at 50% 90%, rgba(168, 85, 247, 0.16) 0%, transparent 50%),
            radial-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px);
        --bg-grid-size: 100% 100%, 100% 100%, 100% 100%, 28px 28px;
        --sidebar-bg: linear-gradient(180deg, #05070f 0%, #0c1222 100%);
    }
    """

# Injected Modern CSS
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    {theme_vars}

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background-color: var(--bg-base) !important;
        background-image: var(--bg-mesh) !important;
        background-size: var(--bg-grid-size) !important;
        background-attachment: fixed !important;
        color: var(--text-primary) !important;
    }}

    [data-testid="stSidebar"] {{
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border-color) !important;
        box-shadow: 6px 0 30px rgba(0, 0, 0, 0.5) !important;
    }}

    [data-testid="stSidebar"] .stRadio label {{
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 4px 0 !important;
    }}

    .main-header {{
        background: var(--card-bg) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 20px !important;
        padding: 28px 36px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35), 0 0 30px var(--glow-color) !important;
        position: relative;
        overflow: hidden;
    }}

    .main-header::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #6366f1, #38bdf8, #a855f7, #6366f1);
        background-size: 200% auto;
        animation: gradientShift 6s linear infinite;
    }}

    @keyframes gradientShift {{
        0% {{ background-position: 0% center; }}
        100% {{ background-position: 200% center; }}
    }}

    .card-metric {{
        background: var(--card-bg) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        padding: 22px 18px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}

    .card-metric:hover {{
        transform: translateY(-4px) !important;
        border-color: var(--hover-border) !important;
        box-shadow: 0 16px 36px var(--glow-color), inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
    }}

    .badge-fake {{
        background: rgba(239, 68, 68, 0.16) !important;
        color: #fca5a5 !important;
        border: 1px solid rgba(239, 68, 68, 0.5) !important;
        box-shadow: 0 0 16px rgba(239, 68, 68, 0.25) !important;
        padding: 6px 18px !important;
        border-radius: 9999px !important;
        font-weight: 700 !important;
        display: inline-block !important;
        letter-spacing: 0.3px !important;
    }}
    
    .badge-reliable {{
        background: rgba(16, 185, 129, 0.16) !important;
        color: #6ee7b7 !important;
        border: 1px solid rgba(16, 185, 129, 0.5) !important;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.25) !important;
        padding: 6px 18px !important;
        border-radius: 9999px !important;
        font-weight: 700 !important;
        display: inline-block !important;
        letter-spacing: 0.3px !important;
    }}
    
    .badge-uncertain {{
        background: rgba(245, 158, 11, 0.16) !important;
        color: #fcd34d !important;
        border: 1px solid rgba(245, 158, 11, 0.5) !important;
        box-shadow: 0 0 16px rgba(245, 158, 11, 0.25) !important;
        padding: 6px 18px !important;
        border-radius: 9999px !important;
        font-weight: 700 !important;
        display: inline-block !important;
        letter-spacing: 0.3px !important;
    }}

    .badge-lang {{
        background: rgba(14, 165, 233, 0.16) !important;
        color: #7dd3fc !important;
        border: 1px solid rgba(14, 165, 233, 0.4) !important;
        box-shadow: 0 0 14px rgba(14, 165, 233, 0.2) !important;
        padding: 5px 14px !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        display: inline-block !important;
    }}

    .stButton > button {{
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid var(--border-color) !important;
    }}

    .stButton > button[kind="primary"] {{
        background: var(--accent-gradient) !important;
        box-shadow: 0 6px 20px var(--glow-color) !important;
        border: none !important;
        color: white !important;
    }}

    .stButton > button[kind="primary"]:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px var(--glow-color) !important;
    }}

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background-color: rgba(15, 23, 42, 0.65) !important;
        color: #f8fafc !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }}

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 2px var(--glow-color) !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: var(--accent-gradient) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px var(--glow-color) !important;
        font-weight: 700 !important;
    }}

    .warning-disclaimer {{
        background: rgba(245, 158, 11, 0.12) !important;
        border: 1px solid rgba(245, 158, 11, 0.35) !important;
        border-left: 4px solid #f59e0b !important;
        padding: 16px 20px !important;
        border-radius: 10px !important;
        margin-top: 20px !important;
        font-size: 14px !important;
        color: #fde68a !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# PAGE 1: HOME & OVERVIEW
# ==============================================================================
if page == "🏠 Home & Overview":
    st.markdown(
        """
        <div class="main-header">
            <div style="display:inline-flex; align-items:center; gap:8px; background:rgba(99,102,241,0.18); border:1px solid rgba(99,102,241,0.35); padding:4px 12px; border-radius:9999px; font-size:12px; font-weight:700; color:#818cf8; margin-bottom:12px; letter-spacing:0.5px;">
                <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#10b981; box-shadow:0 0 10px #10b981;"></span>
                LIVE PRODUCTION ENGINE
            </div>
            <h1 style="margin: 0; font-size: 34px; font-weight: 800; letter-spacing: -0.5px;">🛡️ Veritas AI: Advanced Fake News Detection</h1>
            <p style="margin: 10px 0 0 0; color: var(--text-secondary); font-size: 16px; line-height: 1.6;">
                Calibrated Machine Learning, Explainable AI (LIME), Multilingual Verification (English, Hindi, Marathi), and Publisher Credibility Intelligence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = get_history_summary_stats()
    render_stats_banner(stats)

    st.write("")

    active_in_mem = st.session_state.get("headline_input", "").strip()
    if active_in_mem:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-left: 5px solid #10b981; border-radius: 10px; padding: 12px 18px; margin-bottom: 20px;">
                <div style="font-size: 11px; font-weight: 800; color: #10b981; letter-spacing: 0.8px; text-transform: uppercase;">
                    🟢 ACTIVE WORKSPACE: READY TO ANALYZE IN NEWS ANALYZER
                </div>
                <div style="font-size: 16px; font-weight: 700; color: #f8fafc; margin-top: 3px;">
                    📰 {active_in_mem}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Model Matrix in Circles Section
    st.markdown("### 🎯 Model Verification Matrix in Circles")
    st.caption("Real-time signal analysis across the 4 core dimensions: Fake News Detection, Real News Authenticity, Temporal Specificity, and Geographic Verification.")


    # Interactive sample selector for the circular matrix
    sample_options = {
        "NASA James Webb Space Telescope (Real Science)": {
            "headline": "NASA confirms James Webb Telescope spots oldest galaxies ever observed",
            "text": "Astronomers using the James Webb Space Telescope have discovered galaxies that formed just 300 million years after the Big Bang. The spectroscopic analysis confirmed their immense redshift and stellar composition.",
            "url": "https://www.reuters.com/science/jwst",
            "fake_prob": 26.1,
            "real_prob": 73.9,
            "temporal": 85.0,
            "geo": 90.0,
            "temp_badge": "Calendar Anchored",
            "geo_badge": "Verified Facility",
            "summary": "Verified astrophysics reporting from accredited science correspondents with explicit temporal anchors and real-world observatory venues."
        },
        "Boiled Lemon Hot Water Miracle (Health Hoax)": {
            "headline": "SHOCKING MIRACLE: Boiled Lemon Hot Water Destroys 100% of Cancer Doctors Hiding Truth!!",
            "text": "A revolutionary secret leaked from unnamed sources reveals that boiling lemons and drinking the alkaline water every morning permanently cures all forms of cancer and eliminates 100 percent of viral infections.",
            "url": "http://natural-miracle-health-secret.xyz",
            "fake_prob": 73.3,
            "real_prob": 26.7,
            "temporal": 30.0,
            "geo": 30.0,
            "temp_badge": "Vague Floating Dates",
            "geo_badge": "Unnamed Sources",
            "summary": "Sensational medical clickbait exhibiting undated claims, missing institutional affiliations, and contradicted clinical assertions."
        },
        "Marathi 2000 Note Nano GPS Chip (Viral Currency Rumor)": {
            "headline": "दोन हजार रुपयांच्या नव्या नोटेत नॅनो जीपीएस चिप बसवण्यात आली आहे",
            "text": "सोशल मीडियावर व्हायरल झालेल्या बातमीनुसार नव्या दोन हजार रुपयांच्या नोटेत अत्याधुनिक नॅनो जीपीएस ट्रॅकिंग चिप बसवण्यात आली आहे. ही चिप जमिनीच्या खाली १२० मीटर खोलवर असली तरी उपग्रहाद्वारे पकडता येते असा दावा केला जात आहे.",
            "url": "",
            "fake_prob": 78.6,
            "real_prob": 21.4,
            "temporal": 30.0,
            "geo": 30.0,
            "temp_badge": "Recycled Evergreen Rumor",
            "geo_badge": "Unverifiable Venues",
            "summary": "Persistent Devanagari social media hoax making scientifically impossible claims regarding currency-embedded satellite transponders."
        },
        "ISRO INSAT-3DS Satellite Launch (Hindi Verified News)": {
            "headline": "इसरो ने मौसम उपग्रह इनसैट-3डीएस का सफल प्रक्षेपण किया",
            "text": "भारतीय अंतरिक्ष अनुसंधान संगठन (इसरो) ने श्रीहरिकोटा स्थित सतीश धवन अंतरिक्ष केंद्र से जीएसएलवी-एफ14 के जरिए मौसम उपग्रह इनसैट-3डीएस का सफल प्रक्षेपण किया।",
            "url": "https://www.ndtv.com/india-news",
            "fake_prob": 26.3,
            "real_prob": 73.7,
            "temporal": 85.0,
            "geo": 95.0,
            "temp_badge": "Explicit Launch Date",
            "geo_badge": "Sriharikota Space Center",
            "summary": "Authentic Hindi national news reporting confirmed space agency operations grounded in specific physical spaceport infrastructure."
        },
        "The Onion: 432 Hz Regrows Human Limbs (Satire & Parody)": {
            "headline": "Scientists discover magical sound frequency that regrows lost human limbs in 48 hours",
            "text": "Revolutionary suppressed laboratory trials prove that listening to 432 Hz scalar audio tones stimulates cellular stem cells to regenerate amputated arms and legs within two days without medical intervention.",
            "url": "https://theonion.com/sound-frequency-limbs",
            "fake_prob": 75.8,
            "real_prob": 24.2,
            "temporal": 30.0,
            "geo": 30.0,
            "temp_badge": "Floating Fiction",
            "geo_badge": "Fictitious Lab",
            "summary": "Satirical publication parodying pseudoscientific miracle discoveries with zero verifiable empirical evidence."
        }
    }

    selected_sample_key = st.selectbox(
        "🔬 Select Article Benchmark to View Circular Verification Matrix:",
        options=list(sample_options.keys()),
        index=0
    )
    active_sample = sample_options[selected_sample_key]

    # Prominent Ready to Analyze Banner with Proper Article Name Display
    is_active_in_analyzer = (st.session_state.get("headline_input", "").strip() == active_sample["headline"].strip())
    st.markdown(
        f"""
        <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.3); border-left: 5px solid {'#10b981' if is_active_in_analyzer else '#6366f1'}; border-radius: 12px; padding: 18px 22px; margin: 14px 0 18px 0;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div style="flex: 1; min-width: 280px;">
                    <div style="display: inline-flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 800; color: {'#10b981' if is_active_in_analyzer else '#818cf8'}; letter-spacing: 0.8px; text-transform: uppercase;">
                        <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:{'#10b981' if is_active_in_analyzer else '#818cf8'}; box-shadow:0 0 8px {'#10b981' if is_active_in_analyzer else '#818cf8'};"></span>
                        {'READY TO ANALYZE IN NEWS ANALYZER' if is_active_in_analyzer else 'ARTICLE READY TO LOAD INTO NEWS ANALYZER'}
                    </div>
                    <div style="font-size: 20px; font-weight: 800; color: #f8fafc; margin-top: 6px; line-height: 1.4;">
                        📰 {active_sample['headline']}
                    </div>
                    <div style="font-size: 13px; color: #94a3b8; margin-top: 6px;">
                        <span style="color:#e2e8f0; font-weight:600;">Benchmark:</span> {selected_sample_key} &nbsp;|&nbsp; 
                        <span style="color:#e2e8f0; font-weight:600;">Domain:</span> <code>{active_sample.get('url') or 'Direct Text Input'}</code>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_c1, btn_c2 = st.columns([1.6, 2])
    with btn_c1:
        if st.button("🚀 Load this Article into News Analyzer", key="load_current_benchmark", use_container_width=True, type="primary"):
            st.session_state["headline_input"] = active_sample["headline"]
            st.session_state["text_input"] = active_sample["text"]
            st.session_state["url_input"] = active_sample["url"]
            st.session_state["loaded_article_name"] = active_sample["headline"]
            st.rerun()
    with btn_c2:
        if is_active_in_analyzer:
            st.markdown(
                f"<div style='display:inline-flex; align-items:center; gap:8px; background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.4); padding:9px 18px; border-radius:8px; color:#34d399; font-size:13px; font-weight:600;'>"
                f"<span>✅ <strong>Active:</strong> Ready to Analyze in News Analyzer (switch to sidebar tab)</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.write("")

    # Render 4 Circular Gauges in 4 columns with Prominent Metric Header Cards
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)

    with c_m1:
        st.markdown(
            """
            <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.4); border-top: 3px solid #ef4444; border-radius: 10px; padding: 10px 8px; text-align: center; margin-bottom: 2px; box-shadow: 0 4px 10px rgba(0,0,0,0.25);">
                <div style="font-size: 14.5px; font-weight: 800; color: #fca5a5; letter-spacing: 0.3px; min-height: 40px; display: flex; align-items: center; justify-content: center; line-height: 1.3;">
                    🔴 Fake News Detection
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_c1 = create_circular_metric_chart(
            title="",
            value=active_sample["fake_prob"],
            color="#ef4444",
            subtitle="Deception Signal" if active_sample["fake_prob"] >= 50 else "Low Deception",
        )
        st.plotly_chart(fig_c1, use_container_width=True)
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: -10px; margin-bottom: 6px;">
                <span style="display: inline-block; background: {'rgba(239, 68, 68, 0.2)' if active_sample['fake_prob'] >= 50 else 'rgba(16, 185, 129, 0.2)'}; 
                             color: {'#f87171' if active_sample['fake_prob'] >= 50 else '#34d399'}; 
                             font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 6px; 
                             border: 1px solid {'rgba(239, 68, 68, 0.4)' if active_sample['fake_prob'] >= 50 else 'rgba(16, 185, 129, 0.4)'};">
                    {'⚠️ Deception Risk' if active_sample['fake_prob'] >= 50 else '🛡️ Low Deception'}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_m2:
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); border-top: 3px solid #10b981; border-radius: 10px; padding: 10px 8px; text-align: center; margin-bottom: 2px; box-shadow: 0 4px 10px rgba(0,0,0,0.25);">
                <div style="font-size: 14.5px; font-weight: 800; color: #6ee7b7; letter-spacing: 0.3px; min-height: 40px; display: flex; align-items: center; justify-content: center; line-height: 1.3;">
                    🟢 Real News Authenticity
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_c2 = create_circular_metric_chart(
            title="",
            value=active_sample["real_prob"],
            color="#10b981",
            subtitle="Authentic Pattern" if active_sample["real_prob"] >= 50 else "Unverified Pattern",
        )
        st.plotly_chart(fig_c2, use_container_width=True)
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: -10px; margin-bottom: 6px;">
                <span style="display: inline-block; background: {'rgba(16, 185, 129, 0.2)' if active_sample['real_prob'] >= 50 else 'rgba(239, 68, 68, 0.2)'}; 
                             color: {'#34d399' if active_sample['real_prob'] >= 50 else '#f87171'}; 
                             font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 6px; 
                             border: 1px solid {'rgba(16, 185, 129, 0.4)' if active_sample['real_prob'] >= 50 else 'rgba(239, 68, 68, 0.4)'};">
                    {'✅ Authentic Verified' if active_sample['real_prob'] >= 50 else '⚠️ Unverified Pattern'}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_m3:
        st.markdown(
            """
            <div style="background: rgba(6, 182, 212, 0.12); border: 1px solid rgba(6, 182, 212, 0.4); border-top: 3px solid #06b6d4; border-radius: 10px; padding: 10px 8px; text-align: center; margin-bottom: 2px; box-shadow: 0 4px 10px rgba(0,0,0,0.25);">
                <div style="font-size: 14.5px; font-weight: 800; color: #67e8f9; letter-spacing: 0.3px; min-height: 40px; display: flex; align-items: center; justify-content: center; line-height: 1.3;">
                    ⏱️ Temporal Specificity
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_c3 = create_circular_metric_chart(
            title="",
            value=active_sample["temporal"],
            color="#06b6d4",
            subtitle=active_sample["temp_badge"],
        )
        st.plotly_chart(fig_c3, use_container_width=True)
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: -10px; margin-bottom: 6px;">
                <span style="display: inline-block; background: rgba(6, 182, 212, 0.2); 
                             color: #22d3ee; 
                             font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 6px; 
                             border: 1px solid rgba(6, 182, 212, 0.4);">
                    ⏱️ {active_sample['temp_badge']}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_m4:
        st.markdown(
            """
            <div style="background: rgba(139, 92, 246, 0.12); border: 1px solid rgba(139, 92, 246, 0.4); border-top: 3px solid #8b5cf6; border-radius: 10px; padding: 10px 8px; text-align: center; margin-bottom: 2px; box-shadow: 0 4px 10px rgba(0,0,0,0.25);">
                <div style="font-size: 14.5px; font-weight: 800; color: #c4b5fd; letter-spacing: 0.3px; min-height: 40px; display: flex; align-items: center; justify-content: center; line-height: 1.3;">
                    📍 Geographic Verification
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_c4 = create_circular_metric_chart(
            title="",
            value=active_sample["geo"],
            color="#8b5cf6",
            subtitle=active_sample["geo_badge"],
        )
        st.plotly_chart(fig_c4, use_container_width=True)
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: -10px; margin-bottom: 6px;">
                <span style="display: inline-block; background: rgba(139, 92, 246, 0.2); 
                             color: #a78bfa; 
                             font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 6px; 
                             border: 1px solid rgba(139, 92, 246, 0.4);">
                    📍 {active_sample['geo_badge']}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("🕸️ View Multi-Dimensional Circular Verification Radar Matrix", expanded=False):
        col_rad1, col_rad2 = st.columns([1.2, 1])
        with col_rad1:
            radar_fig = create_circular_matrix_radar(
                fake_prob=active_sample["fake_prob"],
                real_prob=active_sample["real_prob"],
                temporal_score=active_sample["temporal"],
                geo_score=active_sample["geo"],
                title=f"Verification Geometry: {selected_sample_key[:35]}..."
            )
            st.plotly_chart(radar_fig, use_container_width=True)
        with col_rad2:
            st.markdown(f"#### 📑 Signal Summary: {selected_sample_key}")
            st.write(active_sample["summary"])


    st.write("")
    st.write("")


    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("💡 Core Architecture & Innovations")
        st.markdown(
            """
            - **🧠 Explainable AI (XAI)**: Understand the 'why' behind model outputs via LIME local surrogate explanations and term-level weight attribution.
            - **⏱️ Temporal Specificity Analysis**: Measures calendar date anchors and flags floating relative phrases (e.g. *yesterday, recently, just now*) that typify recycled evergreen hoaxes.
            - **📍 Geographic Verification**: Grounds articles in recognized real-world venues, municipalities, and scientific facilities versus unverifiable or mythical locations.
            - **⚠️ Calibrated Uncertainty Detection**: Low-confidence predictions near the decision boundary are flagged as `Uncertain — Needs Human Review` rather than forced into binary labels.
            - **🌐 Multilingual Support**: Detects and processes Indian languages (**Hindi** and **Marathi**) alongside standard **English**.
            - **🔒 SSRF-Protected Source Analysis**: Safely fetches and inspects news URLs with protections against Server-Side Request Forgery and evaluates publisher reputation.
            - **⚡ Clickbait & Emotional Language**: Analyzes sensational keywords, headline-body semantic mismatch, and punctuation cues as supplementary signals.
            - **✅ Fact-Checking Claim Verification**: Extracts factual assertions and maps them to accredited fact-check archives with *Supported*, *Contradicted*, or *Insufficient Evidence* verdicts.
            """
        )

    with col_right:
        st.subheader("📐 System Workflow")
        st.markdown(
            """
            ```mermaid
            flowchart TD
                A["News Input (Text / URL)"] --> B["SSRF-Safe Ingestion"]
                B --> C["Multilingual Detection (EN/HI/MR)"]
                C --> D["TF-IDF Featurization"]
                D --> E["Calibrated Probability Predictor"]
                E --> F{"Confidence > Threshold?"}
                F -- "No" --> G["Flag as Uncertain"]
                F -- "Yes" --> H["Classify: Fake or Reliable"]
                H --> I["LIME & Word Highlights"]
                H --> J["Source & Claim Verification"]
                J --> K["Streamlit Dashboard"]
                G --> K
            ```
            """
        )

    st.divider()
    st.subheader("🚀 Quick Sample Testing")
    st.write("Click any sample below to load it into the News Analyzer:")

    c1, c2, c3, c4 = st.columns(4)
    if c1.button("📄 Sample: Real Science (NASA)", use_container_width=True):
        st.session_state["headline_input"] = "NASA confirms James Webb Telescope spots oldest galaxies ever observed"
        st.session_state["text_input"] = "Astronomers using the James Webb Space Telescope have discovered galaxies that formed just 300 million years after the Big Bang. The spectroscopic analysis confirmed their immense redshift and stellar composition."
        st.session_state["url_input"] = "https://www.reuters.com/science/jwst"
        st.rerun()

    if c2.button("⚠️ Sample: Viral Medical Hoax", use_container_width=True):
        st.session_state["headline_input"] = "SHOCKING MIRACLE: Boiled Lemon Hot Water Destroys 100% of Cancer Doctors Hiding Truth!!"
        st.session_state["text_input"] = "A revolutionary secret leaked from unnamed sources reveals that boiling lemons and drinking the alkaline water every morning permanently cures all forms of cancer and eliminates 100 percent of viral infections."
        st.session_state["url_input"] = "http://natural-miracle-health-secret.xyz"
        st.rerun()

    if c3.button("🇮🇳 Sample: Marathi Currency Rumor", use_container_width=True):
        st.session_state["headline_input"] = "दोन हजार रुपयांच्या नव्या नोटेत नॅनो जीपीएस चिप बसवण्यात आली आहे"
        st.session_state["text_input"] = "सोशल मीडियावर व्हायरल झालेल्या बातमीनुसार नव्या दोन हजार रुपयांच्या नोटेत अत्याधुनिक नॅनो जीपीएस ट्रॅकिंग चिप बसवण्यात आली आहे. ही चिप जमिनीच्या खाली १२० मीटर खोलवर असली तरी उपग्रहाद्वारे पकडता येते असा दावा केला जात आहे."
        st.session_state["url_input"] = ""
        st.rerun()

    if c4.button("🎭 Sample: The Onion (Satire)", use_container_width=True):
        st.session_state["headline_input"] = "Scientists discover magical sound frequency that regrows lost human limbs in 48 hours"
        st.session_state["text_input"] = "Revolutionary suppressed laboratory trials prove that listening to 432 Hz scalar audio tones stimulates cellular stem cells to regenerate amputated arms and legs within two days without medical intervention."
        st.session_state["url_input"] = "https://theonion.com/sound-frequency-limbs"
        st.rerun()



# ==============================================================================
# PAGE 2: NEWS ANALYZER
# ==============================================================================
elif page == "🔍 News Analyzer":
    st.title("🔍 Comprehensive News Article Analyzer")
    st.write("Submit a headline, full body text, or fetch directly from a public news URL.")

    # Summary Statistics Banner on News Analyzer
    stats = get_history_summary_stats()
    render_stats_banner(stats)

    st.write("")


    # URL fetch section with SSRF defense
    with st.expander("🌐 Fetch Article Content from News URL (SSRF Protected)", expanded=False):
        fetch_col1, fetch_col2 = st.columns([3, 1])
        with fetch_col1:
            url_fetch_field = st.text_input(
                "Article Web Address",
                placeholder="https://www.reuters.com/world/india/...",
                key="url_fetch_field",
            )
        with fetch_col2:
            st.write("")
            st.write("")
            fetch_btn = st.button("Fetch & Extract Content", use_container_width=True)

        if fetch_btn and url_fetch_field:
            with st.spinner("Validating URL security and parsing HTML text..."):
                fetch_res = safe_fetch_url(url_fetch_field)
                if fetch_res["success"]:
                    st.success("Content fetched safely!")
                    st.session_state["url_input"] = url_fetch_field
                    st.session_state["headline_input"] = fetch_res["title"]
                    st.session_state["text_input"] = fetch_res["text"]
                else:
                    st.error(f"Security / Retrieval Error: {fetch_res['error']}")

    # Form inputs
    col_in1, col_in2 = st.columns([2, 1])
    with col_in1:
        headline = st.text_input(
            "News Headline / Title",
            value=st.session_state.get("headline_input", ""),
            placeholder="e.g. RBI leaves key interest rates unchanged...",
            key="headline_widget",
        )
        article_text = st.text_area(
            "Article Body / Text Excerpt",
            value=st.session_state.get("text_input", ""),
            height=180,
            placeholder="Paste the full news article body or main paragraphs here...",
            key="text_widget",
        )

    with col_in2:
        article_url = st.text_input(
            "Source URL / Publisher Domain (Optional)",
            value=st.session_state.get("url_input", ""),
            placeholder="e.g. https://www.bbc.com/news/...",
            key="url_widget",
        )
        st.info("💡 Entering the source domain enables automated publisher credibility checks and IFCN accreditation lookup.")

        clear_btn = st.button("🗑️ Clear Input Fields", use_container_width=True)
        if clear_btn:
            st.session_state["headline_input"] = ""
            st.session_state["text_input"] = ""
            st.session_state["url_input"] = ""
            st.rerun()

    analyze_btn = st.button("🚀 Analyze Article Reliability", type="primary", use_container_width=True)

    if analyze_btn:
        if not headline.strip() and not article_text.strip():
            st.warning("⚠️ Please provide at least a headline or article body text to analyze.")
        else:
            with st.spinner("Executing multi-signal analysis..."):
                res = predictor.predict_article(
                    text=article_text,
                    headline=headline,
                    url=article_url,
                    model_name=selected_model,
                    uncertainty_threshold=uncertainty_slider,
                    save_to_history=True,
                )
                st.session_state["last_analysis"] = res

    # Display Analysis Results
    if "last_analysis" in st.session_state and st.session_state["last_analysis"]["success"]:
        res = st.session_state["last_analysis"]

        st.divider()
        st.subheader("📊 Analysis Assessment & Predictions")

        # Top Banner
        b1, b2, b3 = st.columns([1.5, 1, 1])
        with b1:
            pred = res["prediction"]
            if "Fake" in pred:
                st.markdown(f"### Result: <span class='badge-fake'>{res['badge']}</span>", unsafe_allow_html=True)
            elif "Reliable" in pred:
                st.markdown(f"### Result: <span class='badge-reliable'>{res['badge']}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"### Result: <span class='badge-uncertain'>{res['badge']}</span>", unsafe_allow_html=True)

        with b2:
            st.metric(
                label="Model Confidence Score",
                value=f"{res['confidence']}%",
                help="Calibrated prediction probability score.",
            )

        with b3:
            lang_data = res["language"]
            st.markdown(
                f"**Language Detected:**<br><span class='badge-lang'>🌐 {lang_data['name']}</span>",
                unsafe_allow_html=True,
            )

        # Core Signal Metrics: RELIABLE CONTENT & FAKE-LIKE SIGNALS
        sig_c1, sig_c2 = st.columns(2)
        with sig_c1:
            st.markdown(
                f"""
                <div class="card-metric" style="border-top: 4px solid #10b981; background: rgba(16, 185, 129, 0.08); padding: 16px 20px; border-radius: 12px; margin: 10px 0 16px 0;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:13px; font-weight:800; letter-spacing:1px; color:#6ee7b7; text-transform:uppercase;">
                            🟢 RELIABLE CONTENT
                        </span>
                        <span style="font-size:12px; font-weight:700; color:#34d399; background:rgba(16,185,129,0.22); border:1px solid rgba(16,185,129,0.4); padding:3px 10px; border-radius:9999px;">
                            {'DOMINANT SIGNAL' if res['reliable_probability'] >= 50 else 'SECONDARY SIGNAL'}
                        </span>
                    </div>
                    <div style="font-size:38px; font-weight:800; color:#34d399; margin:8px 0 4px 0; letter-spacing:-0.5px;">
                        {res['reliable_probability']}%
                    </div>
                    <div style="font-size:13px; color:#94a3b8; line-height:1.4;">
                        Journalistic integrity markers, verifiable facts, and balanced reporting signals.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with sig_c2:
            st.markdown(
                f"""
                <div class="card-metric" style="border-top: 4px solid #ef4444; background: rgba(239, 68, 68, 0.08); padding: 16px 20px; border-radius: 12px; margin: 10px 0 16px 0;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:13px; font-weight:800; letter-spacing:1px; color:#fca5a5; text-transform:uppercase;">
                            🔴 FAKE-LIKE SIGNALS
                        </span>
                        <span style="font-size:12px; font-weight:700; color:#f87171; background:rgba(239,68,68,0.22); border:1px solid rgba(239,68,68,0.4); padding:3px 10px; border-radius:9999px;">
                            {'HIGH RISK SIGNAL' if res['fake_probability'] >= 50 else 'LOW RISK SIGNAL'}
                        </span>
                    </div>
                    <div style="font-size:38px; font-weight:800; color:#f87171; margin:8px 0 4px 0; letter-spacing:-0.5px;">
                        {res['fake_probability']}%
                    </div>
                    <div style="font-size:13px; color:#94a3b8; line-height:1.4;">
                        Sensationalized assertions, misleading linguistic markers, and affective trigger terms.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Tabbed deep dives
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔎 Key Signals Summary",
            "🧭 Temporal & Geographic",
            "🌐 Source Credibility",
            "⚡ Clickbait & Tone",
            "🛡️ Fact-Check Claims",
        ])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 📊 Probability Distribution")
                fig = go.Figure(
                    go.Bar(
                        x=[res["reliable_probability"], res["fake_probability"]],
                        y=["RELIABLE CONTENT", "FAKE-LIKE SIGNALS"],
                        orientation="h",
                        marker_color=["#10b981", "#ef4444"],
                        text=[f"{res['reliable_probability']}%", f"{res['fake_probability']}%"],
                        textposition="auto",
                        textfont=dict(color="#ffffff", size=13, family="Plus Jakarta Sans, sans-serif"),
                    )
                )
                fig.update_layout(
                    height=200,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cbd5e1", family="Plus Jakarta Sans, sans-serif"),
                    xaxis=dict(
                        range=[0, 100],
                        title="Probability (%)",
                        gridcolor="rgba(255, 255, 255, 0.08)",
                        tickfont=dict(color="#94a3b8"),
                    ),
                    yaxis=dict(tickfont=dict(color="#f8fafc", size=13)),
                    margin=dict(l=20, r=20, t=20, b=30),
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("#### ⚙️ Configuration & Grounding")
                st.write(f"- **Evaluated by Model:** `{res['model_used']}`")
                st.write(f"- **Uncertainty Margin:** `±{int(res['uncertainty_margin']*100)}%` (Band: {int(50-res['uncertainty_margin']*100)}% - {int(50+res['uncertainty_margin']*100)}%)")
                st.write(f"- **Grounding Index:** `{res.get('grounding_index', 0.0)}/100` ({res.get('grounding_badge', '')})")
                st.write(f"- **Temporal Anchoring:** `{res.get('temporal', {}).get('score', 0.0)}/100`")
                st.write(f"- **Geographic Grounding:** `{res.get('geographic', {}).get('score', 0.0)}/100`")
                st.write(f"- **Audit Record ID:** `#{res.get('record_id', 'N/A')}` saved to SQLite")

            # XAI Preview snippet
            st.markdown("#### 🧠 Explainable AI Quick Highlight")
            st.caption("Tokens in red increase fake-pattern likelihood; tokens in green indicate reliable journalistic patterns.")
            st.markdown(res["xai"]["highlighted_html"], unsafe_allow_html=True)
            st.info("👉 Switch to the **'Explainable AI (XAI)'** tab in the sidebar for complete LIME explanations and feature weight charts.")



        with tab2:
            st.markdown(f"### Spatio-Temporal Grounding: `{res.get('grounding_badge', 'Analyzed')}`")
            col_tg1, col_tg2 = st.columns(2)
            with col_tg1:
                t_data = res.get("temporal", {})
                st.markdown(f"#### ⏱️ Temporal Specificity: `{t_data.get('score', 0)} / 100`")
                st.markdown(f"**Rating:** {t_data.get('badge')}")
                st.write(f"**Temporal Analysis:** {t_data.get('explanation')}")
                if t_data.get("explicit_dates_found"):
                    st.write(f"**Calendar Dates Identified:** `{', '.join(t_data['explicit_dates_found'])}`")
                if t_data.get("vague_markers_found"):
                    st.warning(f"⚠️ **Vague / Floating Markers Detected:** `{', '.join(t_data['vague_markers_found'])}` (High risk of recycled rumor)")

            with col_tg2:
                g_data = res.get("geographic", {})
                st.markdown(f"#### 📍 Geographic Verification: `{g_data.get('score', 0)} / 100`")
                st.markdown(f"**Rating:** {g_data.get('badge')}")
                st.write(f"**Geographic Grounding:** {g_data.get('explanation')}")
                if g_data.get("detected_locations"):
                    st.write(f"**Recognized Real-World Locales:** `{', '.join(g_data['detected_locations'])}`")
                if g_data.get("suspicious_markers"):
                    st.error(f"⚠️ **Suspicious / Mythical Venues Flagged:** `{', '.join(g_data['suspicious_markers'])}`")

            st.caption("ℹ️ Credible journalism grounds reports in verifiable calendar dates and identifiable real-world venues. Disinformation frequently recycles undated stories in mysterious, unnamed locales.")

        with tab3:
            src = res["source"]
            st.markdown(f"### Publisher Profile: `{src['domain']}`")
            st.markdown(f"**Credibility Status:** {src['status_badge']}")
            st.write(f"- **Publisher / Outlet:** {src.get('publisher_name', 'Not Specified')}")
            st.write(f"- **Category:** {src.get('category', 'General')}")
            st.write(f"- **Country / Origin:** {src.get('country', 'Global')}")
            st.write(f"- **IFCN Accredited Fact-Checker:** {'✅ Yes' if src.get('fact_check_certified') else '❌ No'}")
            st.write(f"- **Editorial Details:** {src.get('description')}")
            if src.get("warning"):
                st.warning(src["warning"])

        with tab4:
            cb = res["clickbait"]
            st.markdown(f"### Clickbait Assessment: `{cb['badge']}`")
            st.metric("Sensationalism Score", f"{cb['clickbait_score']} / 100")
            st.write(f"**Linguistic Analysis:** {cb['explanation']}")
            if cb.get("sensational_words_found"):
                st.write(f"**Sensational Trigger Terms Found:** `{', '.join(cb['sensational_words_found'])}`")
            if cb.get("mismatch_detected"):
                st.error("⚠️ **Headline-Body Mismatch Flagged:** Significant divergence detected between headline assertions and the article content.")
            st.caption(f"ℹ️ {cb.get('disclaimer')}")

        with tab5:
            st.markdown("### Fact-Check Verification Engine")
            claims = res["claim_verifications"]
            if not claims:
                st.write("No distinct assertive claims identified in the submitted text.")
            else:
                for idx, c in enumerate(claims, 1):
                    with st.expander(f"Claim #{idx}: {c['claim'][:80]}...", expanded=True):
                        st.markdown(f"**Status:** {c['badge']}")
                        st.write(f"**Evidence Summary:** {c['evidence_summary']}")
                        st.write(f"**Accredited Source:** {c['source_title']}")
                        if c.get("source_url"):
                            st.markdown(f"[🔗 View Verified Source Report]({c['source_url']})")
                        st.caption(c["disclaimer"])

        st.markdown(
            f"""
            <div class="warning-disclaimer">
                <strong>⚠️ Ethical AI Disclaimer:</strong> {res['disclaimer']}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        loaded_h = st.session_state.get("headline_input", "").strip()
        if loaded_h:
            st.info(f"📰 **Loaded Article Ready to Analyze:** `{loaded_h}` — Click **'🚀 Analyze Article Reliability'** above to run prediction.")
        else:
            st.info("ℹ️ Enter an article headline or body above, fetch from a news URL, or select a preset sample from **Home & Overview** to begin analysis.")

        st.write("")
        st.markdown("#### 📡 Signal Breakdown Preview")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(
                """
                <div class="card-metric" style="border-top: 4px solid #10b981; background: rgba(16, 185, 129, 0.05); padding: 14px 18px; border-radius: 10px;">
                    <span style="font-size:12px; font-weight:800; letter-spacing:1px; color:#6ee7b7; text-transform:uppercase;">
                        🟢 RELIABLE CONTENT
                    </span>
                    <div style="font-size:32px; font-weight:800; color:#34d399; margin:4px 0;">--%</div>
                    <div style="font-size:12px; color:#94a3b8;">Evaluates journalistic neutrality, verifiable facts, and source grounding.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with sc2:
            st.markdown(
                """
                <div class="card-metric" style="border-top: 4px solid #ef4444; background: rgba(239, 68, 68, 0.05); padding: 14px 18px; border-radius: 10px;">
                    <span style="font-size:12px; font-weight:800; letter-spacing:1px; color:#fca5a5; text-transform:uppercase;">
                        🔴 FAKE-LIKE SIGNALS
                    </span>
                    <div style="font-size:32px; font-weight:800; color:#f87171; margin:4px 0;">--%</div>
                    <div style="font-size:12px; color:#94a3b8;">Evaluates sensationalism, affective exaggeration, and deception cues.</div>
                </div>
                """,
                unsafe_allow_html=True
            )




# ==============================================================================
# PAGE 3: EXPLAINABLE AI (XAI)
# ==============================================================================
elif page == "🧠 Explainable AI (XAI)":
    st.title("🧠 Explainable AI (XAI) Deep Dive")
    st.write("Inspect how the machine learning model interprets specific words and phrases to form its prediction.")

    if "last_analysis" not in st.session_state:
        st.info("ℹ️ No active article analyzed yet. Please run an analysis in the 'News Analyzer' tab first, or load a preset sample below:")
        if st.button("Load & Explain Sample Article"):
            sample_res = predictor.predict_article(
                headline="SHOCKING MIRACLE CURE EXPOSED: Doctors hide boiled lemon remedy!!",
                text="A secret whistleblower leaked proof that hot lemon water cures 100 percent of viral sickness and kills all cancer cells.",
                save_to_history=False,
            )
            st.session_state["last_analysis"] = sample_res
            st.rerun()
    else:
        res = st.session_state["last_analysis"]
        xai_data = res["xai"]
        feature_weights = xai_data["feature_weights"]

        col_x1, col_x2 = st.columns([1.2, 1])

        with col_x1:
            st.subheader("Highlighted Article Text")
            st.caption("Visual token heatmap: Red denotes fake-indicative signals; Green denotes reliable-indicative signals.")
            st.markdown(xai_data["highlighted_html"], unsafe_allow_html=True)

        with col_x2:
            st.subheader("Feature Importance Weights")
            st.caption("Relative weight contribution toward the predicted outcome.")
            fig = predictor.explainer.create_feature_importance_plot(feature_weights)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🔬 Local Surrogate Explanation Details (LIME)")
        lime_meta = xai_data.get("lime_meta", {})
        st.write(f"- **Local Model Fit Score ($R^2$ surrogate):** `{lime_meta.get('score', 1.0):.3f}`")
        st.write(f"- **Surrogate Intercept:** `{lime_meta.get('intercept', 0.0):.4f}`")
        st.markdown(
            """
            > **Understanding XAI Attribution:**
            > Machine learning models identify correlations across thousands of historical labeled training documents.
            > When a word like *'miracle'* or *'whistleblower'* receives a positive weight, it means articles in the training set containing that word were disproportionately labeled as fake news.
            > **Crucially, this is a statistical signal, not philosophical or factual proof.**
            """
        )


# ==============================================================================
# PAGE 4: CLAIM VERIFICATION
# ==============================================================================
elif page == "🛡️ Claim Verification":
    st.title("🛡️ Claim Extraction & Evidence Verification")
    st.write("Veritas AI extracts factual claims from news stories and checks them against peer-reviewed records and accredited fact-checking agencies.")

    if "last_analysis" not in st.session_state:
        st.info("Run an analysis in the 'News Analyzer' tab to inspect extracted claims.")
    else:
        res = st.session_state["last_analysis"]
        claims = res.get("claim_verifications", [])

        if not claims:
            st.warning("No assertive claims were detected in the recent article.")
        else:
            for idx, c in enumerate(claims, 1):
                st.markdown(f"### Claim #{idx}")
                st.info(f"**Assertion:** \"{c['claim']}\"")

                col_c1, col_c2 = st.columns([2, 1])
                with col_c1:
                    st.markdown(f"**Verdict Status:** {c['badge']}")
                    st.write(f"**Evidence Finding:** {c['evidence_summary']}")
                    st.write(f"**Investigating Source:** {c['source_title']}")
                    if c.get("source_url"):
                        st.markdown(f"[🔗 View Verified Source Report]({c['source_url']})")
                with col_c2:
                    st.metric("Evidence Similarity Match", f"{c.get('similarity_score', 0.0)*100:.1f}%")
                    st.caption(f"Verification Date: {c.get('date', 'Archived')}")

                st.divider()

    st.subheader("📚 Verified Evidence Database Inspection")
    st.write("The system includes curated evidence records from WHO, Reuters Fact Check, Alt News, Snopes, Boom Live, and PIB Fact Check.")
    if st.checkbox("Show all verified claims stored in knowledge base"):
        df_claims = pd.DataFrame(predictor.claim_verifier.evidence_db)
        st.dataframe(
            df_claims[["claim", "status", "source", "date", "language"]],
            use_container_width=True,
        )


# ==============================================================================
# PAGE 5: MODEL BENCHMARKS
# ==============================================================================
elif page == "📊 Model Benchmarks":
    st.title("📊 Model Benchmarks & Evaluation")
    st.write("Compare the performance of multiple architectures evaluated on strict, zero-leakage validation and test partitions.")

    metrics_file = os.path.join(predictor.models_dir, "evaluation_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        benchmarks = eval_data.get("validation_benchmarks", [])
        test_eval = eval_data.get("final_test_evaluation", {})

        st.subheader("🏆 Champion Architecture")
        st.success(f"**Selected Model:** {eval_data.get('champion_model')} (Selected via Validation Macro F1 Score)")

        # Benchmark table
        st.subheader("Validation Set Benchmarks")
        df_bench = pd.DataFrame(benchmarks)
        st.dataframe(
            df_bench[[
                "model_name", "accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"
            ]],
            use_container_width=True,
        )

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.subheader("Metric Comparison")
            fig_bar = create_model_comparison_bar_chart(benchmarks)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b2:
            st.subheader("ROC Curves")
            fig_roc = create_roc_curve_figure(benchmarks)
            st.plotly_chart(fig_roc, use_container_width=True)

        st.divider()
        st.subheader("Held-Out Test Set Performance (Evaluated Once)")
        col_t1, col_t2 = st.columns([1, 1.2])
        with col_t1:
            st.metric("Test Set Accuracy", f"{test_eval.get('accuracy', 0.0)*100:.1f}%")
            st.metric("Test Macro F1-Score", f"{test_eval.get('f1_macro', 0.0):.4f}")
            st.metric("Test ROC-AUC", f"{test_eval.get('roc_auc', 0.0):.4f}")

        with col_t2:
            cm = test_eval.get("confusion_matrix", [[0, 0], [0, 0]])
            fig_cm = create_confusion_matrix_figure(cm, title="Held-Out Test Set Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)

        # Cross-validation & Language Slices from benchmark_report.json
        bench_report_file = os.path.join(predictor.models_dir, "benchmark_report.json")
        if os.path.exists(bench_report_file):
            with open(bench_report_file, "r", encoding="utf-8") as f:
                b_report = json.load(f)

            st.divider()
            st.subheader("🔁 5-Fold Stratified Cross-Validation Results")
            kfold_data = []
            for m_name, m_metrics in b_report.get("k_fold_cross_validation", {}).items():
                kfold_data.append({
                    "Model Architecture": m_name,
                    "Mean Accuracy (± Std)": f"{m_metrics['mean_accuracy']:.3f} (±{m_metrics['std_accuracy']:.3f})",
                    "Mean Macro F1 (± Std)": f"{m_metrics['mean_macro_f1']:.3f} (±{m_metrics['std_macro_f1']:.3f})",
                    "Mean ROC-AUC": f"{m_metrics['mean_roc_auc']:.3f}",
                })
            st.dataframe(pd.DataFrame(kfold_data), use_container_width=True)

            st.subheader("🌐 Multilingual Slice Evaluation (Fairness Across Languages)")
            slice_data = []
            for s_lang, s_vals in b_report.get("multilingual_slice_evaluation", {}).items():
                cm_s = s_vals["confusion_matrix"]
                slice_data.append({
                    "Language Slice": s_lang,
                    "Articles Evaluated": s_vals["samples"],
                    "Slice Accuracy": f"{s_vals['accuracy']*100:.1f}%",
                    "Macro F1-Score": f"{s_vals['macro_f1']:.4f}",
                    "Confusion Breakdown": f"True Reliable={cm_s[0][0]}, False Fake={cm_s[0][1]}, False Reliable={cm_s[1][0]}, True Fake={cm_s[1][1]}",
                })
            st.dataframe(pd.DataFrame(slice_data), use_container_width=True)

    else:
        st.warning("Model evaluation metrics not found. Please train models using `python -m src.train`.")


# ==============================================================================
# PAGE 6: ANALYSIS HISTORY
# ==============================================================================
elif page == "📜 Analysis History":
    st.title("📜 Audit Trail & Analysis History")
    st.write("Privacy-preserving audit log stored in local SQLite database. Articles are truncated and hashed with SHA-256.")

    col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
    with col_h1:
        search_kw = st.text_input("Search History", placeholder="Search by headline or domain...")
    with col_h2:
        filter_pred = st.selectbox("Filter Prediction", ["All", "Reliable-like News", "Fake-like News", "Uncertain — Needs Human Review"])
    with col_h3:
        st.write("")
        st.write("")
        if st.button("🗑️ Clear Entire History"):
            count = clear_all_history()
            st.success(f"Cleared {count} audit records.")
            st.rerun()

    records = get_history_records(limit=100, search_query=search_kw, prediction_filter=filter_pred)

    if not records:
        st.info("No analysis records found matching criteria.")
    else:
        df_hist = pd.DataFrame(records)
        st.dataframe(
            df_hist[[
                "id", "timestamp", "headline", "prediction", "confidence",
                "uncertainty_flag", "language", "domain", "clickbait_score", "model_name"
            ]],
            use_container_width=True,
        )

        # Download CSV
        csv_data = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export History as CSV",
            data=csv_data,
            file_name="fake_news_audit_history.csv",
            mime="text/csv",
        )

        st.subheader("Manage Individual Records")
        del_col1, del_col2 = st.columns([2, 1])
        with del_col1:
            rec_id_to_del = st.selectbox("Select Record ID to delete", options=[r["id"] for r in records])
        with del_col2:
            st.write("")
            st.write("")
            if st.button("Delete Record"):
                if delete_record_by_id(rec_id_to_del):
                    st.success(f"Deleted record #{rec_id_to_del}")
                    st.rerun()


# ==============================================================================
# PAGE 7: ABOUT & LIMITATIONS
# ==============================================================================
elif page == "ℹ️ About & Limitations":
    st.title("ℹ️ Methodology, Limitations & Ethical Guidelines")

    st.markdown(
        """
        ### 🎯 System Objectives
        Veritas AI is designed to assist journalists, researchers, and citizens in identifying linguistic patterns,
        sensationalism, and unverified assertions commonly found in online misinformation.

        ### ⚠️ Critical Limitations
        1. **Pattern Matching vs. Reality**:
           A model trained on labeled news articles detects linguistic similarities and lexical markers learned from its training data.
           It **cannot autonomously verify objective reality** or real-time breaking events.
        2. **Domain Reputation vs. Story Truth**:
           A reputable publisher can occasionally publish erroneous reports, and an unindexed or new local blog can report genuine breaking news.
           The source analyzer flags unknown domains for human review without branding them fake.
        3. **Sensationalism as a Signal**:
           Clickbait language often accompanies sensational or exaggerated stories, but legitimate publications also use provocative headlines to capture attention.
           Sensationalism is treated as a supporting signal, never definitive proof.
        4. **Absence of Evidence**:
           When the claim verifier returns *'Insufficient Evidence'*, it means no accredited fact-check entry was found in the archive.
           It does **NOT** mean the claim is false.

        ### 🛡️ Security Protections
        - **SSRF Defense**: The URL extraction engine rejects requests targeting localhost (`127.0.0.1`, `::1`), private networks (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), cloud metadata services (`169.254.169.254`), and non-HTTP protocols.
        - **Resource Limits**: Restricts downloaded payload size to 2 MB and caps input text to 50,000 characters to prevent denial-of-service.

        ### 🔮 Future Roadmap
        - **Multimodal Misinformation**: Reverse image search and deepfake audio/video detection.
        - **Live IFCN ClaimReview API**: Real-time federation with Google Fact Check Tools API.
        - **Social Context & Bot Detection**: Graph-based analysis of social media propagation cascades.
        """
    )
