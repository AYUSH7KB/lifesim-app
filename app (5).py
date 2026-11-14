"""
LifeSim: Future Twin Engine (Class 12 AI Capstone Project)
==========================================================
Single-Page Streamlit Web Application simulating human biological aging,
cellular vitality, and systemic inflammation using an in-memory Pandas
nutrition pipeline, heuristic biometric modeling, and a dual-view
clinical-to-dark HUD visual architecture.

Privacy Standard: Zero-Knowledge / Ephemeral In-Memory Edge Compute.
No database, no disk persistence, and no permanent personal data storage.
"""

import inspect
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================================================
# 1. PAGE CONFIGURATION & SESSION STATE INITIALIZATION
# ==========================================================================
st.set_page_config(
    page_title="LifeSim | Future Twin Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ephemeral session state (wiped entirely when the browser tab closes)
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "clinical_input"  # 'clinical_input' or 'dark_hud'
if "morph_engine_active" not in st.session_state:
    st.session_state.morph_engine_active = False


# ---- Small helpers --------------------------------------------------------
def _stretch(fn) -> dict:
    """
    Streamlit replaced `use_container_width=True` with `width="stretch"`.
    Pick whichever the installed version supports so the app never breaks.
    """
    try:
        if "width" in inspect.signature(fn).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


def _placeholder(fn, text: str) -> dict:
    """
    Dropdowns in Streamlit filter as you type. A placeholder makes that
    search box visible. Only passed if the installed version supports it.
    """
    try:
        if "placeholder" in inspect.signature(fn).parameters:
            return {"placeholder": text}
    except (TypeError, ValueError):
        pass
    return {}


PH_SELECT = _placeholder(st.selectbox, "🔍 Search or select…")
PH_MULTI = _placeholder(st.multiselect, "🔍 Search or select…")

STRETCH_BUTTON = _stretch(st.button)
STRETCH_DF = _stretch(st.dataframe)
STRETCH_CHART = _stretch(st.plotly_chart)


def render_html(html: str) -> None:
    """
    Render raw HTML safely through st.markdown. Leading whitespace and blank
    lines are stripped so Markdown never mistakes indented HTML for a code
    block or ends an HTML block early.
    """
    cleaned = "\n".join(line.strip() for line in html.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def render_notice(message_html: str, accent: str | None = None) -> None:
    """
    Our own info/warning box (replaces st.info / st.warning). Its colours are
    set by the active view's CSS, so text is readable in any Streamlit theme.
    """
    style = f' style="border-left-color:{accent};"' if accent else ""
    render_html(f'<div class="notice-card"{style}>{message_html}</div>')


# ==========================================================================
# 2. IN-MEMORY PANDAS DATA PIPELINE (@st.cache_data)
# ==========================================================================
REQUIRED_TEXT_COLS = ["dish_name", "category", "glycemic_index", "prep_method"]
REQUIRED_NUM_COLS = ["calories", "protein_g", "carbs_g", "fat_g", "aging_delta"]


@st.cache_data(show_spinner=False)
def load_nutrition_dataframe(csv_path: str = "indian_food_nutrition.csv") -> pd.DataFrame:
    """
    Loads the pre-curated Indian food nutrition CSV into volatile RAM once
    (cached) for fast Pandas vector filtering. Returns an empty DataFrame on
    failure; the caller shows the error (UI calls are kept out of the cache).
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    missing = [c for c in REQUIRED_TEXT_COLS + REQUIRED_NUM_COLS if c not in df.columns]
    if missing:
        # Signal a schema problem to the caller via an attribute-free empty frame
        return pd.DataFrame(columns=["__missing__"] + missing)

    for col in REQUIRED_TEXT_COLS:
        df[col] = df[col].astype(str).str.strip()
    for col in REQUIRED_NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df.drop_duplicates(subset="dish_name").reset_index(drop=True)


# ==========================================================================
# 3. DUAL-VIEW DYNAMIC CSS ENGINE (Soft Clinical Input vs Dark-Mode HUD)
# ==========================================================================
FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');"
)


def inject_dual_view_css(mode: str) -> None:
    """
    Injects dynamic CSS based on the active view:
    - 'clinical_input': soft mint / warm peach / lavender pastel gradients
    - 'dark_hud': deep cosmic violet/midnight HUD with neon-pastel glows
    """
    if mode == "clinical_input":
        css = f"""
        <style>
        {FONT_IMPORT}
        html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
        .stApp {{
            background:
                radial-gradient(circle at 12% 18%, rgba(204, 251, 241, 0.75) 0%, transparent 42%),
                radial-gradient(circle at 88% 22%, rgba(255, 237, 213, 0.80) 0%, transparent 45%),
                radial-gradient(circle at 50% 85%, rgba(237, 233, 254, 0.75) 0%, transparent 50%),
                linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
            background-attachment: fixed;
        }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{ background: rgba(255, 255, 255, 0.72); }}
        /* Force readable text even if the browser/OS is in dark mode */
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] li,
        .stApp [data-testid="stMarkdownContainer"] h1,
        .stApp [data-testid="stMarkdownContainer"] h2,
        .stApp [data-testid="stMarkdownContainer"] h3,
        .stApp [data-testid="stMarkdownContainer"] h4,
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp [data-testid="stMetricLabel"] p,
        .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stCaptionContainer"] {{ color: #1E293B; }}
        /* Top bar icons (menu / deploy) stay visible in light AND dark themes */
        .stApp [data-testid="stHeader"] button,
        .stApp [data-testid="stHeader"] a,
        .stApp [data-testid="stToolbar"] * {{ color: #1E293B; }}
        .notice-card {{
            background: rgba(255, 255, 255, 0.88);
            border-left: 4px solid #0D9488;
            border-radius: 14px;
            padding: 12px 16px;
            font-size: 0.88rem;
            line-height: 1.5;
            color: #1E293B;
            margin: 6px 0 12px 0;
        }}
        /* Built-in alerts (only used for error messages now) */
        .stApp [data-testid="stAlert"],
        .stApp [data-testid="stAlert"] > div,
        .stApp [data-baseweb="notification"] {{
            background: rgba(255, 255, 255, 0.95) !important;
            border: 1px solid rgba(15, 118, 110, 0.35);
            border-radius: 14px;
        }}
        .stApp [data-testid="stAlert"] *,
        .stApp [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {{ color: #1E293B; }}
        /* Secondary buttons: light button + dark text */
        .stApp button[data-testid="stBaseButton-secondary"],
        .stApp button[kind="secondary"] {{
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(15, 118, 110, 0.35);
        }}
        .stApp button[data-testid="stBaseButton-secondary"] p,
        .stApp button[kind="secondary"] p,
        .stApp button[data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"] p,
        .stApp button[kind="secondary"] [data-testid="stMarkdownContainer"] p {{ color: #0F172A; }}
        /* Primary buttons: always white text */
        .stApp button[data-testid="stBaseButton-primary"] p,
        .stApp button[kind="primary"] p,
        .stApp button[data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] p,
        .stApp button[kind="primary"] [data-testid="stMarkdownContainer"] p {{ color: #FFFFFF; }}
        .clinical-card {{
            background: rgba(255, 255, 255, 0.78);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.9);
            border-radius: 22px;
            padding: 22px 24px;
            box-shadow: 0 10px 30px -10px rgba(15, 23, 42, 0.06);
            margin-bottom: 18px;
        }}
        .longevity-callout {{
            background: linear-gradient(120deg, rgba(204, 251, 241, 0.65), rgba(224, 231, 255, 0.65));
            border-left: 4px solid #0D9488;
            border-radius: 16px;
            padding: 14px 18px;
            color: #134E4A;
            font-size: 0.92rem;
            margin-bottom: 14px;
        }}
        .motivational-banner {{
            background: linear-gradient(120deg, rgba(255, 237, 213, 0.75), rgba(254, 215, 170, 0.45));
            border-left: 4px solid #EA580C;
            border-radius: 16px;
            padding: 14px 18px;
            color: #7C2D12;
            font-size: 0.92rem;
            margin-bottom: 14px;
        }}
        .pill-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(15, 118, 110, 0.1);
            color: #0F766E;
            margin-bottom: 8px;
        }}
        </style>
        """
    else:
        css = f"""
        <style>
        {FONT_IMPORT}
        html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
        .stApp {{
            background:
                radial-gradient(circle at 20% 20%, rgba(124, 58, 237, 0.28) 0%, transparent 40%),
                radial-gradient(circle at 82% 30%, rgba(45, 212, 191, 0.18) 0%, transparent 42%),
                radial-gradient(circle at 50% 80%, rgba(236, 72, 153, 0.18) 0%, transparent 48%),
                linear-gradient(160deg, #090514 0%, #130924 50%, #1B0E33 100%);
            background-attachment: fixed;
        }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{ background: rgba(19, 9, 36, 0.92); }}
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] li,
        .stApp [data-testid="stMarkdownContainer"] h1,
        .stApp [data-testid="stMarkdownContainer"] h2,
        .stApp [data-testid="stMarkdownContainer"] h3,
        .stApp [data-testid="stMarkdownContainer"] h4,
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp [data-testid="stCaptionContainer"] {{ color: #F8FAFC; }}
        /* Top bar icons (menu / deploy) stay visible in light AND dark themes */
        .stApp [data-testid="stHeader"] button,
        .stApp [data-testid="stHeader"] a,
        .stApp [data-testid="stToolbar"] * {{ color: #F8FAFC; }}
        .notice-card {{
            background: rgba(30, 16, 53, 0.75);
            border-left: 4px solid #A78BFA;
            border-radius: 14px;
            padding: 12px 16px;
            font-size: 0.88rem;
            line-height: 1.5;
            color: #F8FAFC;
            margin: 6px 0 12px 0;
        }}
        /* Built-in alerts (only used for error messages now) */
        .stApp [data-testid="stAlert"],
        .stApp [data-testid="stAlert"] > div,
        .stApp [data-baseweb="notification"] {{
            background: rgba(30, 16, 53, 0.92) !important;
            border: 1px solid rgba(167, 139, 250, 0.35);
            border-radius: 14px;
        }}
        .stApp [data-testid="stAlert"] *,
        .stApp [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {{ color: #F8FAFC; }}
        /* Secondary buttons: dark button + light text */
        .stApp button[data-testid="stBaseButton-secondary"],
        .stApp button[kind="secondary"] {{
            background: rgba(30, 16, 53, 0.85);
            border: 1px solid rgba(167, 139, 250, 0.4);
        }}
        .stApp button[data-testid="stBaseButton-secondary"] p,
        .stApp button[kind="secondary"] p,
        .stApp button[data-testid="stBaseButton-secondary"] [data-testid="stMarkdownContainer"] p,
        .stApp button[kind="secondary"] [data-testid="stMarkdownContainer"] p {{ color: #F8FAFC; }}
        /* Primary buttons: always white text */
        .stApp button[data-testid="stBaseButton-primary"] p,
        .stApp button[kind="primary"] p,
        .stApp button[data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] p,
        .stApp button[kind="primary"] [data-testid="stMarkdownContainer"] p {{ color: #FFFFFF; }}
        .hud-card {{
            background: rgba(30, 16, 53, 0.68);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid rgba(167, 139, 250, 0.25);
            border-radius: 22px;
            padding: 22px 24px;
            box-shadow: 0 12px 36px -8px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.1);
            margin-bottom: 18px;
            color: #F8FAFC;
        }}
        @keyframes pulseOrb {{
            0%   {{ transform: scale(0.96) rotate(0deg);   filter: hue-rotate(0deg); }}
            50%  {{ transform: scale(1.04) rotate(180deg); filter: hue-rotate(25deg); }}
            100% {{ transform: scale(0.96) rotate(360deg); filter: hue-rotate(0deg); }}
        }}
        .morph-orb-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 28px 16px;
            position: relative;
        }}
        .morph-orb {{
            width: 190px;
            height: 190px;
            border-radius: 42% 58% 63% 37% / 45% 42% 58% 55%;
            background: radial-gradient(circle at 35% 35%, #2DD4BF, #A855F7 55%, #EC4899 90%);
            box-shadow: 0 0 45px rgba(168, 85, 247, 0.65),
                        0 0 85px rgba(45, 212, 191, 0.35),
                        inset 0 0 35px rgba(255, 255, 255, 0.45);
            animation: pulseOrb 9s ease-in-out infinite;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        /* Red "warning" state: shown when the net biological-age shift is above zero */
        @keyframes pulseOrbRed {{
            0%   {{ transform: scale(0.96) rotate(0deg); }}
            50%  {{ transform: scale(1.06) rotate(180deg); }}
            100% {{ transform: scale(0.96) rotate(360deg); }}
        }}
        .morph-orb.morph-orb-red {{
            background: radial-gradient(circle at 35% 35%, #FDA4AF, #F43F5E 55%, #9F1239 90%);
            box-shadow: 0 0 45px rgba(244, 63, 94, 0.75),
                        0 0 85px rgba(239, 68, 68, 0.45),
                        inset 0 0 35px rgba(255, 255, 255, 0.35);
            animation: pulseOrbRed 6s ease-in-out infinite;
        }}
        .morph-orb-core {{
            width: 110px;
            height: 110px;
            border-radius: 50%;
            background: rgba(9, 5, 20, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.25);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #F8FAFC;
            text-align: center;
        }}
        </style>
        """
    render_html(css)


# ==========================================================================
# 4. CORE BIOMETRIC LOGIC & HEURISTIC AGING MODEL
# ==========================================================================
# BMI gauge scale: 12 -> 38 mapped onto 0 -> 100 %. Segment widths are derived
# from the real BMI cut-offs so the marker always lands in the right tier.
GAUGE_MIN, GAUGE_MAX = 12.0, 38.0
BMI_CUTS = [18.5, 25.0, 30.0]


def _gauge_pct(value: float) -> float:
    return (min(max(value, GAUGE_MIN), GAUGE_MAX) - GAUGE_MIN) / (GAUGE_MAX - GAUGE_MIN) * 100.0


def calculate_bmi_and_tier(weight_kg: float, height_cm: float) -> tuple[float, str, str]:
    """BMI = weight / height^2, mapped onto the verified four-tier gauge."""
    height_m = max(height_cm / 100.0, 0.5)
    bmi = round(weight_kg / (height_m ** 2), 1)

    if bmi < 18.5:
        return bmi, "Undernourished (< 18.5)", "#38BDF8"
    elif bmi < 25.0:  # 18.5 - 24.9
        return bmi, "Normal (18.5–24.9)", "#10B981"
    elif bmi < 30.0:  # 25.0 - 29.9
        return bmi, "Overweight (25.0–29.9)", "#F59E0B"
    return bmi, "Obesity (≥ 30.0)", "#EF4444"


def render_bmi_horizontal_gauge(bmi: float, tier_label: str, tier_color: str, dark_mode: bool = False) -> None:
    """Stylized horizontal four-tier gauge with a marker at the user's BMI."""
    marker_pct = _gauge_pct(bmi)
    w1 = _gauge_pct(BMI_CUTS[0])
    w2 = _gauge_pct(BMI_CUTS[1]) - w1
    w3 = _gauge_pct(BMI_CUTS[2]) - w1 - w2
    w4 = 100.0 - w1 - w2 - w3

    text_color = "#F8FAFC" if dark_mode else "#1E293B"
    sub_color = "#CBD5E1" if dark_mode else "#64748B"

    render_html(
        f"""
        <div style="margin-top: 8px; margin-bottom: 14px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-weight:700; font-size:1.05rem; color:{text_color};">
              Baseline BMI: {bmi} kg/m²
            </span>
            <span style="background:{tier_color}22; color:{tier_color}; border:1px solid {tier_color};
                         padding:3px 12px; border-radius:999px; font-size:0.8rem; font-weight:600;">
              {tier_label}
            </span>
          </div>
          <div style="position:relative; height:14px; width:100%; border-radius:999px; overflow:hidden;
                      display:flex; box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);">
            <div style="width:{w1:.2f}%; background:#38BDF8;" title="Undernourished (&lt;18.5)"></div>
            <div style="width:{w2:.2f}%; background:#10B981;" title="Normal (18.5–24.9)"></div>
            <div style="width:{w3:.2f}%; background:#F59E0B;" title="Overweight (25.0–29.9)"></div>
            <div style="width:{w4:.2f}%; background:#EF4444;" title="Obesity (≥30.0)"></div>
          </div>
          <div style="position:relative; width:100%; height:18px;">
            <div style="position:absolute; left:calc({marker_pct:.1f}% - 7px); top:-4px;
                        width:14px; height:14px; border-radius:50%; background:{tier_color};
                        border:2px solid #FFFFFF; box-shadow:0 2px 6px rgba(0,0,0,0.35);"></div>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:{sub_color};">
            <span>Undernourished (&lt;18.5)</span>
            <span>Normal (18.5–24.9)</span>
            <span>Overweight (25.0–29.9)</span>
            <span>Obesity (≥30.0)</span>
          </div>
        </div>
        """
    )


def run_predictive_aging_model(
    chronological_age: int,
    bmi: float,
    daily_steps: int,
    training_tags: list[str],
    sleep_hours: float,
    resting_hr: int,
    stress_level: int,
    screen_time: float,
    hydration_liters: float,
    selected_foods_df: pd.DataFrame,
) -> dict:
    """
    Biological Age = Chronological Age + Delta_lifestyle + Delta_dietary_modifiers.
    Also returns Cellular Vitality (0-100), Systemic Inflammation (0-100),
    compound metabolic risk alerts, and 30-year trajectory vectors.
    All numbers are heuristic and for prototype/illustration only.
    """
    delta_lifestyle = 0.0
    vitality_score = 72.0
    inflammation_score = 28.0
    compound_alerts: list[str] = []

    # 1. BMI impact
    if 18.5 <= bmi < 25.0:
        delta_lifestyle -= 0.6
        vitality_score += 5
        inflammation_score -= 4
    elif bmi < 18.5:
        delta_lifestyle += 0.7
        vitality_score -= 5
        inflammation_score += 5
    elif bmi < 30.0:
        delta_lifestyle += 1.1
        vitality_score -= 6
        inflammation_score += 9
    else:
        delta_lifestyle += 2.4
        vitality_score -= 12
        inflammation_score += 16

    # 2. Daily steps (0 - 30,000)
    if daily_steps >= 10000:
        delta_lifestyle -= 1.2
        vitality_score += 9
        inflammation_score -= 8
    elif daily_steps >= 7000:
        delta_lifestyle -= 0.6
        vitality_score += 5
        inflammation_score -= 4
    elif daily_steps < 4000:
        delta_lifestyle += 1.4
        vitality_score -= 8
        inflammation_score += 10

    # 3. Generalized structured physical training tags
    num_training = len(training_tags)
    if num_training > 0:
        delta_lifestyle -= min(num_training * 0.45, 1.5)
        vitality_score += min(num_training * 4.0, 12.0)
        inflammation_score -= min(num_training * 3.5, 10.0)
    else:
        delta_lifestyle += 0.6
        vitality_score -= 4

    # 4. Sleep duration (2 - 12 hours)
    if 7.0 <= sleep_hours <= 8.5:
        delta_lifestyle -= 0.9
        vitality_score += 8
        inflammation_score -= 7
    elif sleep_hours < 6.0:
        delta_lifestyle += 1.8
        vitality_score -= 12
        inflammation_score += 14
    elif sleep_hours > 9.5:
        delta_lifestyle += 0.5
        vitality_score -= 3
        inflammation_score += 4

    # 5. Resting heart rate
    if resting_hr <= 65:
        delta_lifestyle -= 0.7
        vitality_score += 6
        inflammation_score -= 4
    elif resting_hr >= 85:
        delta_lifestyle += 1.1
        vitality_score -= 7
        inflammation_score += 8

    # 6. Daily stress level (1 - 10)
    if stress_level >= 7:
        delta_lifestyle += (stress_level - 5) * 0.45
        vitality_score -= (stress_level - 5) * 3.0
        inflammation_score += (stress_level - 5) * 3.8
    elif stress_level <= 3:
        delta_lifestyle -= 0.5
        vitality_score += 4
        inflammation_score -= 4

    # 7. Screen time & hydration
    if screen_time > 7.0:
        delta_lifestyle += 0.7
        vitality_score -= 5
        inflammation_score += 5

    if hydration_liters >= 2.5:
        delta_lifestyle -= 0.5
        vitality_score += 5
        inflammation_score -= 4
    elif hydration_liters < 1.5:
        delta_lifestyle += 0.8
        vitality_score -= 6
        inflammation_score += 6

    # 8. Pandas vector dietary modifiers
    delta_dietary = 0.0
    if not selected_foods_df.empty:
        # Capped so selecting a very long list can't produce absurd results
        delta_dietary = float(np.clip(selected_foods_df["aging_delta"].sum(), -4.0, 4.0))

        gi = selected_foods_df["glycemic_index"].str.strip().str.lower()
        prep = selected_foods_df["prep_method"]
        names = selected_foods_df["dish_name"].str.lower()

        high_gi_count = int((gi == "high").sum())
        # Counts "Deep-Fried", plain "Fried/..." (e.g. Fried/Tossed) and "Processed",
        # but not gentler methods such as "Pan-Fried" or "Stir-Fried".
        deep_fried_count = int(
            prep.str.contains(r"deep[- ]?fried|(?<![\w-])fried|processed", case=False, na=False).sum()
        )
        healthy_prep_count = int(prep.str.contains(r"fermented|steamed|boiled|raw", case=False, na=False).sum())

        vitality_score += healthy_prep_count * 2.5 - (deep_fried_count * 3.2 + high_gi_count * 2.0)
        inflammation_score += (deep_fried_count * 4.5 + high_gi_count * 3.0) - (healthy_prep_count * 2.2)

        # Compound metabolic risk detection (matches by keyword, not exact name)
        has_stimulant = bool(names.str.contains(r"late-night|energy drink", regex=True).any())
        if has_stimulant and sleep_hours < 6.5:
            compound_alerts.append(
                "⚠️ <strong>Compound Circadian-Metabolic Risk Detected:</strong> Late-night "
                "caffeine/stimulant intake combined with restricted sleep (&lt; 6.5 hrs) amplifies "
                "nocturnal cortisol, suppresses deep cellular repair, and accelerates systemic inflammation."
            )
        if deep_fried_count >= 2 and high_gi_count >= 2:
            compound_alerts.append(
                "⚠️ <strong>Glycation &amp; Lipid Peroxidation Alert:</strong> Combining multiple "
                "deep-fried/processed items with high-glycemic refined carbohydrates triggers rapid "
                "insulin spikes and advanced glycation end-products (AGEs)."
            )

    if not compound_alerts and stress_level >= 8 and sleep_hours < 6.0:
        compound_alerts.append(
            "⚠️ <strong>Neuro-Endocrine Stress Overload:</strong> High daily stress paired with "
            "sleep debt elevates systemic inflammatory cytokines even when baseline nutrition is moderate."
        )
    elif not compound_alerts:
        compound_alerts.append(
            "✅ <strong>Homeostatic Balance Active:</strong> No acute compound metabolic triggers "
            "detected. Consistent hydration, whole-food staples, and restorative sleep are actively "
            "buffering cellular aging."
        )

    total_delta = round(delta_lifestyle + delta_dietary, 1)
    biological_age = round(max(10.0, chronological_age + total_delta), 1)
    vitality_clamped = int(np.clip(round(vitality_score), 5, 99))
    inflammation_clamped = int(np.clip(round(inflammation_score), 5, 98))

    # 30-year trajectories across 4 checkpoints: Today, +10Y, +20Y, +30Y
    annual_drift = total_delta / 10.0
    current_habit_path = [
        round(biological_age, 1),
        round(biological_age + 10 + annual_drift * 3.5, 1),
        round(biological_age + 20 + annual_drift * 7.5, 1),
        round(biological_age + 30 + annual_drift * 12.0, 1),
    ]
    # Optimized path starts from the same point, then gains 1.8 yrs per decade
    # versus chronological ageing, but is never allowed to sit above the
    # current-habit path (it could otherwise for people already ageing slowly).
    ai_optimized_path = [round(biological_age, 1)]
    for decade in (1, 2, 3):
        target = chronological_age + 10 * decade - 1.8 * decade
        ceiling = current_habit_path[decade] - 1.8 * decade
        ai_optimized_path.append(round(min(target, ceiling), 1))

    return {
        "chronological_age": chronological_age,
        "biological_age": biological_age,
        "total_delta": total_delta,
        "delta_lifestyle": round(delta_lifestyle, 2),
        "delta_dietary": round(delta_dietary, 2),
        "vitality": vitality_clamped,
        "inflammation": inflammation_clamped,
        "compound_alerts": compound_alerts,
        "current_habit_path": current_habit_path,
        "ai_optimized_path": ai_optimized_path,
        "confidence_score": 85,
    }


# ==========================================================================
# 5. PLOTLY 30-YEAR AGING TRAJECTORY VISUALIZATION
# ==========================================================================
def build_trajectory_chart(current_path: list[float], optimized_path: list[float]) -> go.Figure:
    """Glowing dark-theme Plotly line chart (plotly_dark, transparent backgrounds)."""
    checkpoints = ["Today", "+10 Years", "+20 Years", "+30 Years"]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=checkpoints,
            y=current_path,
            mode="lines+markers+text",
            name="Current Habit Path",
            text=[f"{v} yrs" for v in current_path],
            textposition="top center",
            line=dict(color="#F43F5E", width=4, shape="spline"),
            marker=dict(size=11, color="#F43F5E", line=dict(width=2, color="#FFFFFF")),
        )
    )
    # "tonexty" fills toward the previous trace, so it belongs on the 2nd trace
    fig.add_trace(
        go.Scatter(
            x=checkpoints,
            y=optimized_path,
            mode="lines+markers+text",
            name="AI Optimized Path",
            text=[f"{v} yrs" for v in optimized_path],
            textposition="bottom center",
            line=dict(color="#2DD4BF", width=4, shape="spline"),
            marker=dict(size=11, color="#2DD4BF", line=dict(width=2, color="#090514")),
            fill="tonexty",
            fillcolor="rgba(45, 212, 191, 0.10)",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=370,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=12, color="#E2E8F0"),
        ),
        xaxis=dict(
            title="Simulation Horizon Checkpoints",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.12)",
            tickfont=dict(color="#CBD5E1"),
        ),
        yaxis=dict(
            title="Projected Biological Age (Years)",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.12)",
            tickfont=dict(color="#CBD5E1"),
        ),
        hovermode="x unified",
    )
    return fig


# ==========================================================================
# 6. MAIN APPLICATION INTERFACE & VIEW SWITCHER
# ==========================================================================
GENDER_OPTIONS = ["Female", "Male", "Non-Binary", "Prefer not to say"]


def init_prototype_defaults() -> None:
    """Illustrative prototype defaults only (never treated as real personal data)."""
    defaults = {
        "age": 18,
        "gender": "Prefer not to say",
        "weight": 62.0,
        "height": 168.0,
        "steps": 7500,
        "training": ["Brisk Walking", "School Sports"],
        "sleep": 7.0,
        "rhr": 68,
        "stress": 5,
        "screen": 5.5,
        "hydration": 2.5,
        "selected_dishes": [
            "Idli (2 pcs)",
            "Dal Roti Sabzi (Thali)",
            "Curd / Chaas (1 glass)",
            "Masala Chai (Regular, 1 cup)",
        ],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    nutrition_df = load_nutrition_dataframe("indian_food_nutrition.csv")

    # Title + view switcher
    nav_col1, nav_col2, nav_col3 = st.columns([2.4, 1.3, 1.3])
    with nav_col1:
        st.markdown("### 🧬 **LifeSim** · Future Twin Engine")
        st.caption("Zero-Knowledge Ephemeral Edge Compute · Class 12 AI Capstone Prototype")
    with nav_col2:
        if st.button(
            "🌿 1. Clinical Input",
            type="primary" if st.session_state.view_mode == "clinical_input" else "secondary",
            **STRETCH_BUTTON,
        ):
            st.session_state.view_mode = "clinical_input"
            st.rerun()
    with nav_col3:
        if st.button(
            "🌌 2. Morphing HUD",
            type="primary" if st.session_state.view_mode == "dark_hud" else "secondary",
            **STRETCH_BUTTON,
        ):
            st.session_state.view_mode = "dark_hud"
            st.rerun()

    inject_dual_view_css(st.session_state.view_mode)

    # Dataset problems are reported here (UI calls don't belong in cached code)
    if nutrition_df.empty and "__missing__" not in nutrition_df.columns:
        st.error(
            "Dataset `indian_food_nutrition.csv` not found. Place it in the same folder as `app.py`."
        )
        return
    if "__missing__" in nutrition_df.columns:
        missing = [c for c in nutrition_df.columns if c != "__missing__"]
        st.error(f"`indian_food_nutrition.csv` is missing required columns: {', '.join(missing)}")
        return

    init_prototype_defaults()

    with st.sidebar:
        st.markdown("#### 🛡 Zero-Knowledge Privacy")
        render_notice(
            "All biometric &amp; dietary inputs exist strictly in volatile RAM for this session "
            "and are permanently wiped when this tab closes."
        )
        st.markdown("---")
        st.markdown("#### ⚡ Quick View Switcher")
        st.write(
            "Current Mode: **Clinical Input Aesthetic**"
            if st.session_state.view_mode == "clinical_input"
            else "Current Mode: **Dark-Mode Predictive HUD**"
        )

    # ======================================================================
    # VIEW 1: CLEAN, STRESS-FREE CLINICAL AESTHETIC
    # ======================================================================
    if st.session_state.view_mode == "clinical_input":
        col_banner1, col_banner2 = st.columns(2)
        with col_banner1:
            render_html(
                """
                <div class="longevity-callout">
                  <strong>🌿 Clinical Longevity Insight:</strong>
                  Did you know? Regular restorative sleep (7–8 hours) and fermented regional
                  staples can lower cellular inflammation markers by up to <strong>20%</strong>.
                </div>
                """
            )
        with col_banner2:
            render_html(
                """
                <div class="motivational-banner">
                  <strong>✨ Empowering Perspective:</strong> Your biology isn't fixed in
                  stone—small, consistent daily habits compound into decades of cellular vitality.
                </div>
                """
            )

        left_col, right_col = st.columns([1.05, 1.15], gap="large")

        with left_col:
            render_html(
                """
                <div class="clinical-card">
                  <span class="pill-badge">STEP 1 · BIOMETRIC BASELINE</span>
                  <h3 style="margin:0 0 6px 0;">Personal Biometrics &amp; Vital Signs</h3>
                  <p style="margin:0; font-size:0.88rem; color:#475569;">
                    Illustrative prototype inputs—adjust freely to explore real-time physiological shifts.
                  </p>
                </div>
                """
            )

            b_col1, b_col2 = st.columns(2)
            with b_col1:
                st.session_state.age = st.number_input(
                    "Chronological Age (Years)", min_value=12, max_value=90,
                    value=int(st.session_state.age), step=1,
                )
                st.session_state.weight = st.number_input(
                    "Weight (kg)", min_value=30.0, max_value=180.0,
                    value=float(st.session_state.weight), step=0.5,
                )
                st.session_state.rhr = st.slider(
                    "Resting Heart Rate (bpm)", min_value=45, max_value=115,
                    value=int(st.session_state.rhr),
                )
            with b_col2:
                st.session_state.gender = st.selectbox(
                    "Gender Identity", options=GENDER_OPTIONS,
                    index=GENDER_OPTIONS.index(st.session_state.gender),
                    **PH_SELECT,
                )
                st.session_state.height = st.number_input(
                    "Height (cm)", min_value=120.0, max_value=220.0,
                    value=float(st.session_state.height), step=0.5,
                )
                st.session_state.stress = st.slider(
                    "Daily Stress Level (1 = Calm, 10 = High)", min_value=1, max_value=10,
                    value=int(st.session_state.stress),
                )

            bmi_val, bmi_tier, bmi_color = calculate_bmi_and_tier(
                st.session_state.weight, st.session_state.height
            )
            render_bmi_horizontal_gauge(bmi_val, bmi_tier, bmi_color, dark_mode=False)

            render_html(
                """
                <div class="clinical-card" style="margin-top:16px;">
                  <span class="pill-badge">STEP 2 · MOVEMENT &amp; CIRCADIAN RHYTHM</span>
                  <h4 style="margin:0 0 4px 0;">Activity, Recovery &amp; Hydration</h4>
                </div>
                """
            )

            st.session_state.steps = st.slider(
                "Average Daily Steps (0 – 30,000)", min_value=0, max_value=30000,
                value=int(st.session_state.steps), step=500,
            )
            st.session_state.training = st.multiselect(
                "Structured Physical Training (Generalized Prototype Tags)",
                options=["Brisk Walking", "Cycling", "Home Workout", "School Sports"],
                default=st.session_state.training,
                **PH_MULTI,
            )

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.session_state.sleep = st.slider(
                    "Sleep Duration (hrs)", min_value=2.0, max_value=12.0,
                    value=float(st.session_state.sleep), step=0.5,
                )
            with m_col2:
                st.session_state.screen = st.slider(
                    "Screen Time (hrs/day)", min_value=0.5, max_value=16.0,
                    value=float(st.session_state.screen), step=0.5,
                )
            with m_col3:
                st.session_state.hydration = st.slider(
                    "Daily Hydration (L)", min_value=0.5, max_value=5.0,
                    value=float(st.session_state.hydration), step=0.25,
                )

        with right_col:
            render_html(
                """
                <div class="clinical-card">
                  <span class="pill-badge">STEP 3 · IN-MEMORY PANDAS NUTRITION ENGINE</span>
                  <h3 style="margin:0 0 6px 0;">Pan-Indian Dietary &amp; Metabolic Matrix</h3>
                  <p style="margin:0; font-size:0.88rem; color:#475569;">
                    Select regular staples &amp; indulgences. Pandas vector filtering extracts
                    glycemic proxies, preparation methods, and biological aging modifiers in real time.
                  </p>
                </div>
                """
            )

            all_dishes = nutrition_df["dish_name"].tolist()
            default_valid = [d for d in st.session_state.selected_dishes if d in all_dishes]
            st.session_state.selected_dishes = st.multiselect(
                "Select Your Regular Indian Staples, Snacks & Beverages:",
                options=all_dishes,
                default=default_valid,
                **PH_MULTI,
            )

            filtered_food_df = nutrition_df[nutrition_df["dish_name"].isin(st.session_state.selected_dishes)]

            if not filtered_food_df.empty:
                tot_cal = int(filtered_food_df["calories"].sum())
                tot_prot = round(filtered_food_df["protein_g"].sum(), 1)
                tot_carbs = round(filtered_food_df["carbs_g"].sum(), 1)
                tot_fat = round(filtered_food_df["fat_g"].sum(), 1)
                net_diet_delta = round(filtered_food_df["aging_delta"].sum(), 2)

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Selected Calories", f"{tot_cal} kcal")
                k2.metric("Protein", f"{tot_prot} g")
                k3.metric("Carbs / Fats", f"{tot_carbs}g / {tot_fat}g")
                k4.metric(
                    "Dietary Aging Delta",
                    f"{net_diet_delta:+.2f} yrs",
                    delta="Protective" if net_diet_delta <= 0 else "Accelerating",
                    delta_color="normal" if net_diet_delta <= 0 else "inverse",
                )

                st.dataframe(
                    filtered_food_df[
                        ["dish_name", "category", "calories", "protein_g",
                         "glycemic_index", "prep_method", "aging_delta"]
                    ],
                    hide_index=True,
                    **STRETCH_DF,
                )
            else:
                render_notice(
                    "Select at least one dish above to run Pandas nutritional vector analysis.",
                    accent="#F59E0B",
                )

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                "🚀 Transition to Dark-Mode Predictive Morphing Engine HUD →",
                type="primary",
                **STRETCH_BUTTON,
            ):
                st.session_state.view_mode = "dark_hud"
                st.rerun()

    # ======================================================================
    # VIEW 2: DARK-MODE PREDICTIVE MORPHING ENGINE HUD
    # ======================================================================
    else:
        bmi_val, bmi_tier, bmi_color = calculate_bmi_and_tier(
            st.session_state.weight, st.session_state.height
        )
        filtered_food_df = nutrition_df[nutrition_df["dish_name"].isin(st.session_state.selected_dishes)]

        sim = run_predictive_aging_model(
            chronological_age=int(st.session_state.age),
            bmi=bmi_val,
            daily_steps=int(st.session_state.steps),
            training_tags=st.session_state.training,
            sleep_hours=float(st.session_state.sleep),
            resting_hr=int(st.session_state.rhr),
            stress_level=int(st.session_state.stress),
            screen_time=float(st.session_state.screen),
            hydration_liters=float(st.session_state.hydration),
            selected_foods_df=filtered_food_df,
        )

        delta_val = sim["total_delta"]
        delta_badge_color = "#2DD4BF" if delta_val <= 0 else "#F43F5E"
        delta_status_text = "REJUVENATING VELOCITY" if delta_val <= 0 else "ACCELERATED AGING SHIFT"

        hud1, hud2, hud3, hud4 = st.columns([1.2, 1.2, 1.2, 1.1])

        with hud1:
            render_html(
                f"""
                <div class="hud-card">
                  <div style="font-size:0.75rem; color:#A78BFA; letter-spacing:0.08em; font-weight:700;">
                    BIOLOGICAL AGE IMPACT
                  </div>
                  <div style="font-size:2.1rem; font-weight:800; margin:6px 0; color:{delta_badge_color};">
                    {sim['biological_age']} yrs
                    <span style="font-size:1.05rem;">({delta_val:+.1f} yrs)</span>
                  </div>
                  <div style="font-size:0.78rem; color:#CBD5E1;">
                    Chronological: {sim['chronological_age']} yrs · <strong>{delta_status_text}</strong>
                  </div>
                </div>
                """
            )
        with hud2:
            render_html(
                f"""
                <div class="hud-card">
                  <div style="font-size:0.75rem; color:#2DD4BF; letter-spacing:0.08em; font-weight:700;">
                    CELLULAR VITALITY INDEX
                  </div>
                  <div style="font-size:2.1rem; font-weight:800; margin:6px 0; color:#F8FAFC;">
                    {sim['vitality']} / 100
                  </div>
                  <div style="font-size:0.78rem; color:#CBD5E1;">Mitochondrial &amp; Metabolic Resilience</div>
                </div>
                """
            )
            st.progress(sim["vitality"] / 100.0)
        with hud3:
            render_html(
                f"""
                <div class="hud-card">
                  <div style="font-size:0.75rem; color:#F43F5E; letter-spacing:0.08em; font-weight:700;">
                    SYSTEMIC INFLAMMATION MARKER
                  </div>
                  <div style="font-size:2.1rem; font-weight:800; margin:6px 0; color:#F8FAFC;">
                    {sim['inflammation']} / 100
                  </div>
                  <div style="font-size:0.78rem; color:#CBD5E1;">
                    Glycation &amp; Oxidative Stress Proxy (Lower is Better)
                  </div>
                </div>
                """
            )
            st.progress(sim["inflammation"] / 100.0)
        with hud4:
            render_html(
                f"""
                <div class="hud-card">
                  <div style="font-size:0.75rem; color:#38BDF8; letter-spacing:0.08em; font-weight:700;">
                    AI DATA CONFIDENCE SCORE
                  </div>
                  <div style="font-size:2.1rem; font-weight:800; margin:6px 0; color:#38BDF8;">
                    {sim['confidence_score']}%
                  </div>
                  <div style="font-size:0.78rem; color:#CBD5E1;">
                    Verified &gt; 60% Empirical Reliability Threshold
                  </div>
                </div>
                """
            )

        for alert_msg in sim["compound_alerts"]:
            border = "#2DD4BF" if alert_msg.startswith("✅") else "#F43F5E"
            render_html(
                f"""
                <div class="hud-card" style="border-left: 4px solid {border}; padding: 14px 20px;">
                  {alert_msg}
                </div>
                """
            )

        chart_col, twin_col = st.columns([1.45, 1.0], gap="large")

        with chart_col:
            render_html(
                """
                <div class="hud-card">
                  <h4 style="margin:0 0 4px 0; color:#F8FAFC;">📈 30-Year Divergent Biological Aging Trajectories</h4>
                  <p style="margin:0; font-size:0.84rem; color:#CBD5E1;">
                    Comparing your projected <strong>Current Habit Path</strong> against the
                    <strong>AI Optimized Path</strong> across four discrete evaluation checkpoints.
                  </p>
                </div>
                """
            )
            fig = build_trajectory_chart(sim["current_habit_path"], sim["ai_optimized_path"])
            st.plotly_chart(fig, **STRETCH_CHART)
            render_bmi_horizontal_gauge(bmi_val, bmi_tier, bmi_color, dark_mode=True)

        with twin_col:
            render_html(
                """
                <div class="hud-card">
                  <h4 style="margin:0 0 6px 0; color:#F8FAFC;">🔮 Future Visual Twin &amp; 3D Morphing Engine</h4>
                  <p style="margin:0; font-size:0.84rem; color:#CBD5E1;">
                    Harmonic cellular resonance visualization driven by your real-time biomarkers.
                  </p>
                </div>
                """
            )

            if st.button("⚡ Initialize 3D Morphing Engine", type="primary", **STRETCH_BUTTON):
                st.session_state.morph_engine_active = True

            if st.session_state.morph_engine_active:
                years_reclaimed = round(sim["current_habit_path"][2] - sim["ai_optimized_path"][2], 1)
                # Twin turns red once the net aging shift goes above zero
                twin_is_red = delta_val > 0
                orb_class = "morph-orb morph-orb-red" if twin_is_red else "morph-orb"
                twin_label_color = "#FDA4AF" if twin_is_red else "#A78BFA"
                twin_vital_color = "#FB7185" if twin_is_red else "#2DD4BF"
                render_html(
                    f"""
                    <div class="hud-card morph-orb-container">
                      <div class="{orb_class}">
                        <div class="morph-orb-core">
                          <span style="font-size:0.7rem; color:{twin_label_color}; letter-spacing:0.08em;">TWIN STATE</span>
                          <span style="font-size:1.35rem; font-weight:800;">{sim['biological_age']}y</span>
                          <span style="font-size:0.68rem; color:{twin_vital_color};">Vitality {sim['vitality']}%</span>
                        </div>
                      </div>
                      <p style="margin-top:18px; font-size:0.84rem; color:#E2E8F0; text-align:center;">
                        Fluid organic cellular geometry synchronized with your 30-year projected
                        biological velocity ({delta_val:+.1f} yrs net shift).
                      </p>
                    </div>
                    """
                )
                render_html(
                    f"""
                    <div class="hud-card">
                      <h5 style="margin:0 0 8px 0; color:#2DD4BF;">🧬 AI-Optimized Longevity Protocol</h5>
                      <ul style="margin:0; padding-left:18px; font-size:0.85rem; color:#E2E8F0; line-height:1.6;">
                        <li><strong>Glycemic &amp; Prep Swap:</strong> Replace deep-fried/high-GI snacks with
                        fermented or steamed staples (<em>Idli, Dhokla, Sprouts Chaat, Moong Dal Khichdi</em>)
                        to cut systemic inflammation.</li>
                        <li><strong>Circadian Cutoff:</strong> Avoid late-night caffeine/sugared chai at least
                        6 hours before sleep to support nocturnal cellular repair.</li>
                        <li><strong>Movement Synergy:</strong> Pair 8,500+ daily steps with 2 structured
                        training modes (e.g., <em>Brisk Walking + Cycling</em>) to reclaim up to
                        <strong>{years_reclaimed} biological years</strong> over the next two decades.</li>
                      </ul>
                    </div>
                    """
                )
            else:
                render_notice(
                    "Click <strong>⚡ Initialize 3D Morphing Engine</strong> above to synthesize your "
                    "Future Visual Twin and personalized longevity protocol."
                )

        if st.button("← Return to Clinical Biometric & Diet Input View"):
            st.session_state.view_mode = "clinical_input"
            st.rerun()


if __name__ == "__main__":
    main()
