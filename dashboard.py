"""
Cardiomegaly Decision Support Dashboard
----------------------------------------
Run with: streamlit run dashboard.py

Upload a chest X-ray (or pick a sample) + patient metadata, and get:
- Tool A / Tool B CTR estimates
- Tool B segmentation overlay
- Tool agreement check
- Final pipeline verdict (No Cardiomegaly / Cardiomegaly Present / Flag for Review)
- Full clinical rationale + evidence tier

UI: light clinical theme -- white cards on a pale field, sage accent.
Backend logic is unchanged: same imports, same calls, same field usage.
"""

import glob
import io
import math
import os
import tempfile
from datetime import date

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st

from validation_gate import VisionResult
from pipeline import process_patient
from extraction_tool_b import run_tool_b

try:
    from extraction_tool_a import run_tool_a
    TOOL_A_AVAILABLE = True
except Exception:
    TOOL_A_AVAILABLE = False


SAMPLES_DIR = "samples"

# ---------------------------------------------------------------- palette ---
BG = "#F0EBE3"        # warm paper field
CARD = "#FFFDF9"      # card surface, a shade warmer than white
LINE = "#E4DCCE"      # hairline border
TRACK = "#EAE3D7"     # empty progress track
INK = "#3A3025"       # headings, values
SLATE = "#5C5145"     # body copy
MUTED = "#7A6C5C"     # labels
CLAY = "#A8663C"      # primary accent -- chrome, buttons, bars
CLAY_LT = "#C08659"   # lighter accent
CLAY_DK = "#7E4527"   # accent text, button gradient end
STEEL = "#3D6E8C"     # secondary accent + "flag for review"
GREEN = "#3F6B45"     # no cardiomegaly
RED = "#9E2B25"       # cardiomegaly present
FLAG = "#3D6E8C"      # flag fill (steel, not amber -- amber vanishes on cream)
FLAG_TXT = "#2F566F"  # flag text
FLAG_BG = "#E9EEF2"   # flag pill
LIGHTBOX = "#241E17"   # warm near-black frame behind the X-ray
LUNG_C = "#6FA8C4"     # lung outline on the lightbox (lighter than STEEL,
                       # so it never reads as the "flag for review" colour)
HEART_C = "#E0705E"    # heart outline -- brighter than RED, which is too dark
                       # to read against the lightbox frame

# Segmentation label mapping for Tool B's mask.
# ianpan/chest-x-ray-basic returns right lung / left lung / heart.
# If the overlay colours come out swapped, change these two lines only.
LUNG_LABELS = (1, 2)
HEART_LABELS = (3,)

SCALE_MIN = 30.0
SCALE_MAX = 70.0

VERDICT_STYLE = {
    "No cardiomegaly": (GREEN, GREEN, "Cardiothoracic ratio within normal limits"),
    "Cardiomegaly present": (RED, RED, "Cardiothoracic ratio above threshold"),
    "Flag for review": (FLAG_TXT, STEEL, "Manual radiologist review required"),
}

# ------------------------------------------------------------- small icons ---
def _icon(path_d, color=CLAY_DK, size=15):
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round">{path_d}</svg>'
    )

IC_SCAN = _icon('<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
                '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
                '<circle cx="12" cy="12" r="3"/>')
IC_USER = _icon('<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>')
IC_RULER = _icon('<path d="M3 12h18"/><path d="M7 9v6"/><path d="M12 8v8"/><path d="M17 9v6"/>')


# ------------------------------------------------------------------- CSS ---
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {{
  --bg: {BG}; --card: {CARD}; --line: {LINE}; --ink: {INK};
  --shadow-1: 0 1px 2px rgba(31,45,74,.05), 0 10px 28px -12px rgba(31,45,74,.16);
  --shadow-2: 0 1px 2px rgba(31,45,74,.06), 0 18px 42px -16px rgba(31,45,74,.22);
}}

/* font has to be forced broadly -- Streamlit sets font-family per element */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stSidebar"], button, input, select, textarea, optgroup,
p, h1, h2, h3, h4, h5, h6, span, div, label, li, td, th {{
  font-family: 'Plus Jakarta Sans', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
}}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
  background: {BG} !important; color: {SLATE} !important; color-scheme: light !important;
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
.block-container, [data-testid="stMainBlockContainer"] {{
  max-width: 1320px; padding-top: 1.2rem; padding-bottom: 3rem;
}}

/* ================= force light on every BaseWeb surface ================= */
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
  background: {CARD} !important; border-right: 1px solid {LINE} !important;
}}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {{ color: {SLATE}; }}
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div,
[data-baseweb="textarea"], [data-testid="stNumberInput"] > div,
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {{
  background: {BG} !important; color: {INK} !important;
  border-color: {LINE} !important; border-radius: 12px !important;
  -webkit-text-fill-color: {INK} !important;
}}
[data-baseweb="select"] svg, [data-testid="stNumberInput"] svg {{ fill: {MUTED} !important; }}
[data-baseweb="select"] > div:focus-within, [data-baseweb="base-input"]:focus-within {{
  border-color: {CLAY} !important; box-shadow: 0 0 0 3px {CLAY}2E !important;
}}
[data-testid="stNumberInput"] button {{
  background: {CARD} !important; color: {INK} !important; border-color: {LINE} !important;
}}
[data-baseweb="popover"], [data-baseweb="menu"], [data-baseweb="popover"] > div {{
  background: {CARD} !important; border-radius: 12px !important;
  box-shadow: var(--shadow-2) !important;
}}
[role="option"], [data-baseweb="menu"] li {{
  background: {CARD} !important; color: {INK} !important; font-size: 13px !important;
  border-radius: 8px !important; margin: 2px 4px !important;
}}
[role="option"]:hover, [data-baseweb="menu"] li:hover {{ background: {BG} !important; }}
[role="option"][aria-selected="true"] {{ background: {CLAY}22 !important; color: {CLAY_DK} !important; }}
[data-testid="stFileUploaderDropzone"] {{
  background: {BG} !important; border: 1.5px dashed {CLAY}70 !important; border-radius: 14px !important;
}}
[data-testid="stFileUploaderDropzone"] * {{ color: {SLATE} !important; }}
[data-testid="stFileUploaderDropzone"] button {{
  background: {CARD} !important; color: {INK} !important;
  border: 1px solid {LINE} !important; box-shadow: none !important; font-weight: 500 !important;
}}
[data-baseweb="radio"] div[aria-checked="true"] > div,
[data-baseweb="radio"] div[data-checked="true"],
[data-testid="stCheckbox"] span[data-checked="true"],
[data-testid="stCheckbox"] [aria-checked="true"] > div {{
  background-color: {CLAY} !important; border-color: {CLAY} !important;
}}
[data-baseweb="radio"] div[aria-checked="false"] > div,
[data-testid="stCheckbox"] [aria-checked="false"] > div {{
  background-color: {CARD} !important; border-color: #C9D2E4 !important;
}}
[data-testid="stRadio"] label p, [data-testid="stCheckbox"] label p {{
  font-size: 13px !important; color: {SLATE} !important;
}}

/* ---- sidebar text scale: widget labels, options, dropzone ---- */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label p {{
  font-size: 11.5px !important; font-weight: 600 !important;
  color: {MUTED} !important; margin-bottom: 3px !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] [data-testid="stCheckbox"] label p {{
  font-size: 12.5px !important; font-weight: 500 !important; color: {SLATE} !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] input {{ font-size: 12.5px !important; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small {{
  font-size: 11.5px !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{
  font-size: 12px !important; padding: 7px 14px !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{ padding: 16px 14px !important; }}
[data-testid="stAlert"] {{
  background: {BG} !important; color: {SLATE} !important;
  border-radius: 12px !important; border: 1px solid {LINE} !important;
}}
[data-testid="stSpinner"] * {{ color: {SLATE} !important; }}

.stButton > button, [data-testid="stDownloadButton"] > button {{
  background: linear-gradient(135deg, {CLAY} 0%, {CLAY_DK} 100%) !important;
  color: #FFFFFF !important; border: none !important; border-radius: 12px !important;
  font-weight: 700 !important; font-size: 13.5px !important; padding: 11px 18px !important;
  box-shadow: 0 4px 14px {CLAY}4D !important; transition: all .18s ease !important;
}}
.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {{
  transform: translateY(-1px); box-shadow: 0 8px 22px {CLAY}5C !important;
}}
.stButton > button:focus-visible, [data-testid="stDownloadButton"] > button:focus-visible {{
  outline: 2px solid {INK} !important; outline-offset: 2px;
}}

/* Streamlit puts the label in a child <p>/<div>, which ignores the button's
   own colour -- so paint every descendant explicitly. */
.stButton > button *, [data-testid="stDownloadButton"] > button *,
.stButton > button p, [data-testid="stDownloadButton"] > button p,
.stButton > button div, .stButton > button span,
.stButton > button[kind="primary"] *, .stButton > button[kind="primaryFormSubmit"] * {{
  color: #FFFFFF !important;
  -webkit-text-fill-color: #FFFFFF !important;
  font-weight: 700 !important;
}}
.stButton > button svg, [data-testid="stDownloadButton"] > button svg {{
  fill: #FFFFFF !important; stroke: #FFFFFF !important;
}}

/* ---- real containers: st.container(border=True) styled as a card ---- */
[data-testid="stVerticalBlockBorderWrapper"] {{
  background: {CARD}; border: 1px solid {LINE} !important; border-radius: 18px !important;
  box-shadow: var(--shadow-1); padding: 6px 4px;
}}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
  box-shadow: none; border: none !important; padding: 0;
}}

/* ============================ components ============================ */
.card {{
  background: {CARD}; border-radius: 18px; padding: 22px 24px;
  border: 1px solid {LINE}; box-shadow: var(--shadow-1); margin-bottom: 16px;
}}
.eyebrow {{
  display: flex; align-items: center; gap: 9px;
  font-size: 10.5px; font-weight: 800; letter-spacing: .1em;
  color: {MUTED}; margin: 0 0 16px 0;
}}
.eyebrow::before {{ content: ""; width: 3px; height: 13px; border-radius: 2px; background: {CLAY}; }}

/* hero strip */
.hero {{
  background: linear-gradient(115deg, {CARD} 0%, {CARD} 55%, {CLAY}1A 100%);
  border: 1px solid {LINE}; border-radius: 18px; padding: 22px 26px;
  display: flex; align-items: center; justify-content: space-between; gap: 22px;
  margin-bottom: 16px; box-shadow: var(--shadow-1);
}}
.hero h1 {{ font-size: 24px; font-weight: 800; color: {INK}; margin: 0; letter-spacing: -.025em; }}
.hero .sub {{ font-size: 12.5px; color: {MUTED}; margin: 7px 0 0 0; }}
.chips {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
.chip {{ font-size: 11px; font-weight: 700; padding: 7px 13px; border-radius: 20px; white-space: nowrap; }}
.chip.ok {{ background: {CLAY}22; color: {CLAY_DK}; }}
.chip.warn {{ background: {FLAG_BG}; color: {FLAG_TXT}; }}
.chip.neutral {{ background: {BG}; color: {SLATE}; }}

/* ---- HERO RESULT: the one dominant element ---- */
.result {{
  border-radius: 20px; padding: 26px 30px; margin-bottom: 16px;
  border: 1px solid {LINE}; box-shadow: var(--shadow-2);
  display: flex; align-items: center; justify-content: space-between; gap: 30px;
}}
.result .left {{ min-width: 0; }}
.result .word {{ font-size: 30px; font-weight: 800; letter-spacing: -.03em; line-height: 1.1; }}
.result .why {{ font-size: 12.5px; color: {MUTED}; margin-top: 9px; line-height: 1.55; }}
.result .tags {{ display: flex; gap: 8px; margin-top: 15px; flex-wrap: wrap; }}
.result .tag {{ font-size: 11px; font-weight: 700; padding: 6px 12px; border-radius: 9px; }}
.result .right {{ text-align: right; flex: none; }}
.result .big {{
  font-size: 66px; font-weight: 800; line-height: .95; letter-spacing: -.045em;
  font-variant-numeric: tabular-nums;
}}
.result .big .u {{ font-size: 26px; font-weight: 700; margin-left: 2px; }}
.result .cap {{
  font-size: 10.5px; font-weight: 800; letter-spacing: .12em;
  color: {MUTED}; margin-top: 8px;
}}
.result .delta {{ font-size: 12.5px; font-weight: 600; margin-top: 6px; }}

/* sidebar */
.sb-brand {{ display: flex; align-items: center; gap: 11px; margin-bottom: 4px; }}
.sb-mark {{
  width: 38px; height: 38px; border-radius: 12px; flex: none;
  background: linear-gradient(135deg, {CLAY} 0%, {STEEL} 100%);
  display: flex; align-items: center; justify-content: center; font-size: 17px;
}}
.sb-brand .t {{ font-size: 14.5px; font-weight: 800; color: {INK}; line-height: 1.2; }}
.sb-brand .s {{ font-size: 10.5px; color: {MUTED}; margin-top: 2px; }}
.sb-sec {{
  display: flex; align-items: center; gap: 8px;
  font-size: 10px; font-weight: 800; letter-spacing: .11em;
  color: {MUTED}; margin: 16px 0 4px 0;
}}
.sb-sec svg {{ flex: none; }}
.sb-rule {{ border: none; border-top: 1px solid {LINE}; margin: 14px 0 12px 0; }}
.sb-badge {{ font-size: 11px; padding: 10px 12px; border-radius: 11px; margin-top: 10px; line-height: 1.5; }}
.sb-badge.warn {{ background: {FLAG_BG}; color: {FLAG_TXT}; }}
.sb-badge.ok {{ background: {CLAY}1F; color: {CLAY_DK}; }}
[data-testid="stSidebar"] [data-testid="stElementContainer"] {{ margin-bottom: -6px; }}

/* patient */
.pt-head {{ display: flex; align-items: center; gap: 13px; margin-bottom: 14px; }}
.pt-avatar {{
  width: 52px; height: 52px; border-radius: 17px; flex: none;
  background: linear-gradient(135deg, {CLAY}30 0%, {STEEL}30 100%);
  display: flex; align-items: center; justify-content: center;
  font-size: 19px; font-weight: 800; color: {CLAY_DK};
}}
.pt-name {{ font-size: 15.5px; font-weight: 800; color: {INK}; }}
.pt-when {{ font-size: 11px; color: {MUTED}; margin-top: 2px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 170px; }}
.kv {{ display: flex; justify-content: space-between; font-size: 12.5px; padding: 8px 0; }}
.kv + .kv {{ border-top: 1px solid {LINE}; }}
.kv .k {{ color: {MUTED}; }}
.kv .v {{ color: {INK}; font-weight: 700; }}

/* bars */
.bar-row {{ display: flex; align-items: center; gap: 13px; margin-bottom: 15px; }}
.bar-body {{ flex: 1; min-width: 0; }}
.bar-top {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }}
.bar-val {{ font-size: 14px; font-weight: 800; color: {INK}; font-variant-numeric: tabular-nums; }}
.bar-val.na {{ color: {MUTED}; font-weight: 400; }}
.bar-lab {{ font-size: 11.5px; color: {MUTED}; }}
.bar-track {{ height: 8px; border-radius: 5px; background: {TRACK}; overflow: hidden; }}
.bar-fill {{ height: 100%; border-radius: 5px; animation: grow .7s cubic-bezier(.22,.61,.36,1) both; }}
.bar-sub {{ font-size: 10.5px; color: {MUTED}; margin-top: 5px; }}
.bar-badge {{
  width: 38px; height: 38px; border-radius: 12px; flex: none;
  display: flex; align-items: center; justify-content: center;
  font-size: 12.5px; font-weight: 800; color: #FFFFFF;
}}
.bar-note {{ font-size: 10.5px; color: {MUTED}; margin-top: 2px; }}

/* gauge */
.gauge-wrap {{ display: flex; justify-content: center; padding: 2px 0; }}
.gauge-cap {{ text-align: center; font-size: 11.5px; color: {MUTED}; margin-top: 8px; }}
.gauge-wrap line {{ animation: tick .5s ease-out both; }}

.rationale {{ font-size: 13px; line-height: 1.7; color: {SLATE}; }}
.legend {{ display: flex; gap: 18px; margin-top: 4px; padding: 0 4px; }}
.legend .item {{ display: flex; align-items: center; gap: 7px; font-size: 11.5px; color: {MUTED}; }}
.legend .sw {{ width: 11px; height: 11px; border-radius: 4px; }}

/* gate table */
.gate {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
.gate td {{ padding: 12px; border-bottom: 1px solid {LINE}; vertical-align: middle; }}
.gate tr:last-child td {{ border-bottom: none; }}
.gate td.st {{ width: 64px; }}
.gate td.nm {{ color: {INK}; font-weight: 700; white-space: nowrap; }}
.gate td.dt {{ color: {MUTED}; }}
.gate .tag {{ display: inline-block; font-size: 10px; font-weight: 800; padding: 4px 9px;
  border-radius: 7px; letter-spacing: .04em; }}

/* pre-run */
.step {{ display: flex; gap: 13px; padding: 13px 0; }}
.step + .step {{ border-top: 1px solid {LINE}; }}
.step .n {{
  width: 25px; height: 25px; border-radius: 9px; flex: none;
  background: {CLAY}22; color: {CLAY_DK}; font-size: 11.5px; font-weight: 800;
  display: flex; align-items: center; justify-content: center;
}}
.step .t {{ font-size: 13px; font-weight: 700; color: {INK}; }}
.step .d {{ font-size: 11.5px; color: {MUTED}; margin-top: 3px; line-height: 1.5; }}
.ghost {{
  border: 1.5px dashed {LINE}; border-radius: 16px; padding: 34px 20px 30px;
  text-align: center; color: {MUTED}; font-size: 12.5px; background: {CARD};
}}
.ghost .cap {{ margin-top: 14px; line-height: 1.6; }}
.preview-cap {{ font-size: 11.5px; color: {MUTED}; text-align: center; margin-top: 2px; }}

.disclaimer {{
  font-size: 11px; color: {MUTED}; line-height: 1.6;
  border-top: 1px solid {LINE}; padding-top: 14px; margin-top: 24px;
}}
::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: #D5DCEC; border-radius: 5px; }}
::-webkit-scrollbar-thumb:hover {{ background: #BFC9E0; }}

/* ---- one orchestrated entrance, first render after a run only ---- */
@keyframes riseIn {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes grow {{ from {{ width: 0 !important; }} }}
@keyframes tick {{ from {{ opacity: 0; }} }}
.enter [data-testid="stVerticalBlockBorderWrapper"],
.enter .card, .enter .result {{ animation: riseIn .5s cubic-bezier(.22,.61,.36,1) both; }}
@media (prefers-reduced-motion: reduce) {{
  .enter *, .bar-fill, .gauge-wrap line {{ animation: none !important; }}
}}
/* Preview image fills its card whatever st.image API the version honours. */
[data-testid="stMainBlockContainer"] [data-testid="stImage"],
[data-testid="stMainBlockContainer"] [data-testid="stImageContainer"] {{
  width: 100% !important;
}}
[data-testid="stMainBlockContainer"] [data-testid="stImage"] img,
[data-testid="stMainBlockContainer"] [data-testid="stImageContainer"] img {{
  width: 100% !important; height: auto !important;
  border-radius: 12px !important; display: block;
}}

/* ============ LAST WORD: sidebar text scale ============ */
/* Broad on purpose -- Streamlit's label markup differs between versions.
   Brand, section headers and the badge use <div>, so they are unaffected. */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {{
  font-size: 11.5px !important;
  line-height: 1.5 !important;
}}
/* field labels: quiet, so the section headers lead */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] > div label > div p {{
  font-weight: 600 !important; color: {MUTED} !important; margin-bottom: 2px !important;
}}
/* selectable options and typed values stay a touch larger */
[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] [data-testid="stCheckbox"] label p,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] input {{
  font-size: 12.5px !important; font-weight: 500 !important; color: {INK} !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] [data-testid="stCheckbox"] label p {{ color: {SLATE} !important; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{ padding: 15px 14px !important; }}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {{
  font-size: 12px !important; padding: 7px 14px !important;
}}
[data-testid="stSidebar"] .stButton button p,
[data-testid="stSidebar"] .stButton button {{ font-size: 13px !important; }}
</style>
"""


# --------------------------------------------------------------- helpers ---
def show_image(src, caption=None):
    """st.image full-width across Streamlit versions.

    Newer releases replaced use_container_width with width="stretch"; older
    ones don't know "stretch" at all. Try the new API, fall back to the old.
    """
    for attempt in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            st.image(src, caption=caption, **attempt)
            return
        except TypeError:
            continue
        except Exception:
            continue
    st.image(src, caption=caption)


def _load_sample_paths():
    if not os.path.isdir(SAMPLES_DIR):
        return []
    paths = sorted(
        glob.glob(os.path.join(SAMPLES_DIR, "*.png"))
        + glob.glob(os.path.join(SAMPLES_DIR, "*.jpg"))
        + glob.glob(os.path.join(SAMPLES_DIR, "*.jpeg"))
    )
    return paths[:10]


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_pct(value):
    v = _num(value)
    return f"{v:.1f}%" if v is not None else None


def _frac(value):
    v = _num(value)
    if v is None:
        return None
    return max(0.0, min(1.0, (v - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)))


def _as_label_map(mask):
    m = np.asarray(mask)
    if m.ndim == 3:
        ch = int(np.argmin(m.shape))
        m = np.moveaxis(m, ch, -1)
        if m.shape[-1] == 1:
            m = m[..., 0]
        elif float(np.nanmax(m)) <= 1.0 + 1e-6:
            lab = np.zeros(m.shape[:2], dtype=int)
            for i in range(m.shape[-1]):
                lab[m[..., i] > 0.5] = i + 1
            m = lab
        else:
            m = np.argmax(m, axis=-1)
    return np.rint(np.nan_to_num(m)).astype(int)


def _split_masks(mask):
    lab = _as_label_map(mask)
    labels = [int(v) for v in np.unique(lab) if int(v) != 0]
    if not labels:
        return None, None
    lungs = np.isin(lab, [l for l in LUNG_LABELS if l in labels])
    heart = np.isin(lab, [h for h in HEART_LABELS if h in labels])
    if not lungs.any() and not heart.any():
        heart = lab == labels[-1]
        lungs = np.isin(lab, labels[:-1])
    return lungs, heart


def build_overlay_fig(image, mask):
    """X-ray on an inset dark frame, so it reads as a lightbox panel."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(LIGHTBOX)
    ax.set_facecolor(LIGHTBOX)
    ax.imshow(image, cmap="gray")

    lungs, heart = _split_masks(mask)
    for m, color in ((lungs, LUNG_C), (heart, HEART_C)):
        if m is None or not m.any():
            continue
        m = m.astype(float)
        ax.imshow(np.ma.masked_where(m < 0.5, m), cmap=ListedColormap([color]),
                  alpha=0.24, vmin=0, vmax=1)
        ax.contour(m, levels=[0.5], colors=[color], linewidths=1.5)

    ax.axis("off")
    fig.subplots_adjust(left=0.035, right=0.965, top=0.965, bottom=0.035)
    return fig


def ribcage_svg():
    """Line-art ribcage for the empty state."""
    ribs = ""
    for i in range(5):
        y = 30 + i * 15
        w = 30 + i * 5
        ribs += (
            f'<path d="M60 {y} C {60 - w} {y + 4}, {60 - w} {y + 20}, 60 {y + 22}" />'
            f'<path d="M60 {y} C {60 + w} {y + 4}, {60 + w} {y + 20}, 60 {y + 22}" />'
        )
    return (
        f'<svg width="120" height="128" viewBox="0 0 120 128" fill="none" '
        f'stroke="{LINE}" stroke-width="2.4" stroke-linecap="round">'
        f'<path d="M60 16 V 118" stroke="{CLAY}66"/>'
        f'<path d="M38 14 H 82" />{ribs}</svg>'
    )


def radial_gauge_svg(value, threshold, color, dim=False):
    size, cx, cy = 200, 100, 100
    r_in, r_out = 64, 86
    start, sweep, n = 135.0, 270.0, 54

    frac = _frac(value)
    t_frac = _frac(threshold)
    idle = "#EEF1F8" if dim else "#E4E9F4"
    ticks = []
    for i in range(n):
        t = i / (n - 1)
        ang = math.radians(start + sweep * t)
        ca, sa = math.cos(ang), math.sin(ang)
        active = frac is not None and t <= frac
        col = color if active else idle
        w = 3.5 if active else 2.5
        delay = 0.25 + t * 0.5 if active else 0.0
        ticks.append(
            f'<line x1="{cx + r_in * ca:.1f}" y1="{cy + r_in * sa:.1f}" '
            f'x2="{cx + r_out * ca:.1f}" y2="{cy + r_out * sa:.1f}" '
            f'stroke="{col}" stroke-width="{w}" stroke-linecap="round" '
            f'style="animation-delay:{delay:.2f}s"/>'
        )
    if t_frac is not None:
        ang = math.radians(start + sweep * t_frac)
        ca, sa = math.cos(ang), math.sin(ang)
        ticks.append(
            f'<line x1="{cx + (r_in - 8) * ca:.1f}" y1="{cy + (r_in - 8) * sa:.1f}" '
            f'x2="{cx + (r_out + 7) * ca:.1f}" y2="{cy + (r_out + 7) * sa:.1f}" '
            f'stroke="{INK}" stroke-width="2.6" stroke-linecap="round" '
            f'style="animation-delay:.8s"/>'
        )

    centre = _fmt_pct(threshold) or "—"
    return (
        f'<div class="gauge-wrap"><svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'{"".join(ticks)}'
        f'<text x="{cx}" y="{cy - 2}" text-anchor="middle" font-size="10" font-weight="800" '
        f'letter-spacing="1.2" fill="{MUTED}">THRESHOLD</text>'
        f'<text x="{cx}" y="{cy + 22}" text-anchor="middle" font-size="21" font-weight="800" '
        f'fill="{INK if not dim else MUTED}">{centre}</text></svg></div>'
    )


def bar_row(label, value, badge, color, threshold=None):
    f = _frac(value)
    shown = _fmt_pct(value)
    v, t = _num(value), _num(threshold)
    fill = "" if f is None else f"width:{f * 100:.1f}%; background:{color};"
    val_cls = "bar-val" if shown else "bar-val na"
    badge_bg = color if shown else "#C9D2E4"

    if v is not None and t is not None:
        d = v - t
        sub = f"{abs(d):.1f} pts {'above' if d >= 0 else 'below'} threshold"
    elif shown:
        sub = "decision threshold" if threshold is None and badge == "T" else "&nbsp;"
    else:
        sub = "not available"

    return (
        f'<div class="bar-row"><div class="bar-body">'
        f'<div class="bar-top"><span class="{val_cls}">{shown or "N/A"}</span>'
        f'<span class="bar-lab">{label}</span></div>'
        f'<div class="bar-track"><div class="bar-fill" style="{fill}"></div></div>'
        f'<div class="bar-sub">{sub}</div>'
        f'</div><div class="bar-badge" style="background:{badge_bg}">{badge}</div></div>'
    )


def kv_html(pairs):
    return "".join(
        f'<div class="kv"><span class="k">{k}</span><span class="v">{v}</span></div>'
        for k, v in pairs
    )


def gate_table_html(checks):
    rows = ""
    for c in checks:
        ok = c["passed"]
        color, bg, word = (GREEN, f"{GREEN}1C", "PASS") if ok else (RED, f"{RED}16", "FAIL")
        rows += (
            f'<tr><td class="st"><span class="tag" style="background:{bg};color:{color}">{word}</span></td>'
            f'<td class="nm">{c["name"]}</td><td class="dt">{c["detail"]}</td></tr>'
        )
    return f'<div class="card" style="padding:10px 16px;"><table class="gate">{rows}</table></div>'


def result_hero_html(report, txt_color, fill_color, sub):
    resolved = _num(getattr(report, "resolved_ctr", None))
    thresh = _num(getattr(report, "threshold_center", None))
    tier = getattr(report, "evidence_tier", None)
    rule = getattr(report, "rule_id", None)
    diff = _num(getattr(report, "tool_agreement_pct_diff", None))

    tags = ""
    for text in filter(None, [
        f"Evidence tier {tier}" if tier else None,
        f"Rule {rule}" if rule else None,
        f"Tool agreement Δ {diff:.1f} pts" if diff is not None else "Tool agreement unavailable",
    ]):
        tags += f'<span class="tag" style="background:{fill_color}1A;color:{txt_color}">{text}</span>'

    if resolved is not None:
        whole, frac_part = f"{resolved:.1f}".split(".")
        big = f'{whole}.{frac_part}<span class="u">%</span>'
    else:
        big = '<span style="font-size:34px">N/A</span>'

    delta = ""
    if resolved is not None and thresh is not None:
        d = resolved - thresh
        delta = (f'<div class="delta" style="color:{txt_color}">'
                 f"{abs(d):.1f} pts {'above' if d >= 0 else 'below'} threshold</div>")

    return (
        f'<div class="result" style="background:linear-gradient(115deg,{CARD} 0%,{CARD} 58%,{fill_color}14 100%)">'
        f'<div class="left"><div class="word" style="color:{txt_color}">{report.final_verdict}</div>'
        f'<div class="why">{sub}</div><div class="tags">{tags}</div></div>'
        f'<div class="right"><div class="big" style="color:{txt_color}">{big}</div>'
        f'<div class="cap">RESOLVED CTR</div>{delta}</div></div>'
    )


# ------------------------------------------------------------------ page ---
st.set_page_config(page_title="Cardiomegaly Decision Support", page_icon="🫀", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------- sidebar ---
image_path = None
preview_name = None
is_dicom = False

with st.sidebar:
    st.markdown(
        '<div class="sb-brand"><div class="sb-mark">🫀</div>'
        '<div><div class="t">Cardiomegaly CDS</div>'
        '<div class="s">Clinical decision support</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(f'<div class="sb-sec">{IC_SCAN}IMAGE SOURCE</div>', unsafe_allow_html=True)
    sample_paths = _load_sample_paths()
    source_mode = st.radio(
        "Image source",
        ["Upload my own", "Use a sample image"] if sample_paths else ["Upload my own"],
        label_visibility="collapsed",
    )

    if source_mode == "Use a sample image":
        sample_labels = [os.path.basename(p) for p in sample_paths]
        chosen_label = st.selectbox("Sample chest X-ray", sample_labels)
        image_path = sample_paths[sample_labels.index(chosen_label)]
        preview_name = chosen_label
    else:
        uploaded_file = st.file_uploader(
            "Chest X-ray (jpg / png / dcm)",
            type=["jpg", "jpeg", "png", "dcm"],
            key="xray_uploader",
        )
        if uploaded_file is not None:
            suffix = "." + uploaded_file.name.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                image_path = tmp.name
            preview_name = uploaded_file.name
            is_dicom = suffix.lower() == ".dcm"

    st.markdown(f'<div class="sb-sec">{IC_USER}PATIENT DETAILS</div>', unsafe_allow_html=True)
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    ancestry = st.selectbox("Ancestry", ["Non-Caucasian", "Caucasian", "Unspecified"])

    st.markdown(f'<div class="sb-sec">{IC_RULER}IMAGE QUALITY</div>', unsafe_allow_html=True)
    view = st.selectbox("View (Auto trusts Tool B)", ["Auto", "PA", "AP"])
    inspiration_adequate = st.checkbox("Inspiration adequate", value=True)
    rotation_acceptable = st.checkbox("Rotation acceptable", value=True)

    st.markdown('<hr class="sb-rule">', unsafe_allow_html=True)
    run_button = st.button("Run clinical analysis", type="primary", use_container_width=True)

    st.markdown(
        '<div class="sb-badge ok">Tool A (HybridGNet) ready</div>'
        if TOOL_A_AVAILABLE
        else '<div class="sb-badge warn">Tool A (HybridGNet) is not wired up yet — '
        'see extraction_tool_a.py. Running on Tool B only, so cases route to '
        '“Flag for review” for a missing second estimate.</div>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ hero ---
saved = st.session_state.get("results")
chips = [
    f'<span class="chip {"ok" if TOOL_A_AVAILABLE else "warn"}">Tool A '
    f'{"ready" if TOOL_A_AVAILABLE else "not wired"}</span>',
    '<span class="chip ok">Tool B ready</span>',
    f'<span class="chip neutral">View {(saved or {}).get("resolved_view") or "—"}</span>',
    f'<span class="chip neutral">{date.today().strftime("%d %b %Y")}</span>',
]
st.markdown(
    f'<div class="hero"><div><h1>Cardiomegaly Decision Support</h1>'
    f'<p class="sub">Cardiothoracic ratio · dual-tool segmentation · rule-based safety gate</p></div>'
    f'<div class="chips">{"".join(chips)}</div></div>',
    unsafe_allow_html=True,
)


# ------------------------------------------------------------ run/analyse ---
if run_button:
    if image_path is None:
        st.error("Select a sample or upload a chest X-ray before running the analysis.")
        st.stop()

    with st.spinner("Running Tool B (ianpan/chest-x-ray-basic)..."):
        try:
            result_b = run_tool_b(image_path)
        except Exception as e:
            st.error(f"Tool B failed: {e}")
            st.stop()

    result_a = None
    if TOOL_A_AVAILABLE:
        with st.spinner("Running Tool A (HybridGNet)..."):
            try:
                result_a = run_tool_a(image_path)
            except NotImplementedError as e:
                st.warning(str(e))
            except Exception as e:
                st.error(f"Tool A failed: {e}")

    vision_a = (
        result_a["vision_result"] if result_a
        else VisionResult(ctr=None, confidence=None, tool_name="ToolA_HybridGNet")
    )
    vision_b = result_b["vision_result"]

    resolved_view = view if view != "Auto" else result_b.get("view")
    dicom_meta = {
        "age": int(age),
        "gender": gender,
        "view": resolved_view if resolved_view != "Lateral" else None,
        "ancestry": None if ancestry == "Unspecified" else ancestry,
    }

    report = process_patient(
        dicom_meta, vision_a, vision_b,
        inspiration_adequate=inspiration_adequate,
        rotation_acceptable=rotation_acceptable,
    )

    st.session_state["results"] = {
        "report": report, "vision_a": vision_a, "vision_b": vision_b,
        "result_b": result_b, "resolved_view": resolved_view,
        "age": int(age), "gender": gender, "ancestry": ancestry,
        "source_name": preview_name or os.path.basename(image_path),
    }
    st.session_state["animate"] = True   # entrance plays once, not on every rerun
    st.rerun()


# ---------------------------------------------------------------- results ---
results = st.session_state.get("results")
# The entrance plays on the first render after a run, then the flag is consumed
# so toggling the safety-gate checkbox doesn't replay it.
if st.session_state.pop("animate", False):
    st.markdown(
        "<style>"
        ".result { animation: riseIn .5s cubic-bezier(.22,.61,.36,1) both; animation-delay: .02s; }"
        ".card { animation: riseIn .5s cubic-bezier(.22,.61,.36,1) both; animation-delay: .12s; }"
        '[data-testid="stVerticalBlockBorderWrapper"] '
        "{ animation: riseIn .5s cubic-bezier(.22,.61,.36,1) both; animation-delay: .20s; }"
        "@media (prefers-reduced-motion: reduce) {"
        '.result, .card, [data-testid="stVerticalBlockBorderWrapper"] { animation: none !important; } }'
        "</style>",
        unsafe_allow_html=True,
    )

if not results:
    left, right = st.columns([1.2, 1], gap="medium")

    with left:
        with st.container(border=True):
            st.markdown('<p class="eyebrow">SELECTED IMAGE</p>', unsafe_allow_html=True)
            if image_path and not is_dicom:
                show_image(image_path)
                st.markdown(f'<div class="preview-cap">{preview_name}</div>', unsafe_allow_html=True)
            elif image_path and is_dicom:
                st.markdown(
                    f'<div class="ghost">{ribcage_svg()}<div class="cap">'
                    f"DICOM selected — {preview_name}<br>Preview renders after analysis.</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="ghost">{ribcage_svg()}<div class="cap">'
                    "No image selected yet.<br>Pick a sample or upload a file in the sidebar."
                    "</div></div>",
                    unsafe_allow_html=True,
                )

    with right:
        st.markdown(
            '<div class="card"><p class="eyebrow">WHAT THE ANALYSIS RETURNS</p>'
            '<div class="step"><div class="n">1</div><div><div class="t">Dual CTR estimate</div>'
            '<div class="d">Tool A (HybridGNet) and Tool B (ianpan) measure the cardiothoracic '
            "ratio independently.</div></div></div>"
            '<div class="step"><div class="n">2</div><div><div class="t">Agreement check</div>'
            '<div class="d">The two estimates are compared; a wide gap routes the case to review.</div></div></div>'
            '<div class="step"><div class="n">3</div><div><div class="t">Threshold rule</div>'
            '<div class="d">The resolved ratio is scored against an age, sex, ancestry and '
            "view-adjusted threshold.</div></div></div>"
            '<div class="step"><div class="n">4</div><div><div class="t">Verdict and rationale</div>'
            '<div class="d">A verdict, evidence tier, rule ID and full written rationale.</div></div></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card"><p class="eyebrow">CARDIOTHORACIC RATIO</p>'
            + radial_gauge_svg(None, None, CLAY, dim=True)
            + '<div class="gauge-cap">Awaiting analysis</div></div>',
            unsafe_allow_html=True,
        )

else:
    report = results["report"]
    vision_a = results["vision_a"]
    vision_b = results["vision_b"]
    result_b = results["result_b"]
    txt_color, fill_color, default_sub = VERDICT_STYLE.get(report.final_verdict, (INK, CLAY, ""))
    thresh = getattr(report, "threshold_center", None)

    # ---- the one dominant element ----
    st.markdown(result_hero_html(report, txt_color, fill_color, default_sub), unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 1.5, 1.05], gap="medium")

    with col_a:
        initials = (results["gender"][:1] or "?").upper()
        st.markdown(
            '<div class="card"><div class="pt-head">'
            f'<div class="pt-avatar">{initials}{results["age"]}</div>'
            f'<div><div class="pt-name">{results["gender"]}, {results["age"]}</div>'
            f'<div class="pt-when">{results["source_name"]}</div></div></div>'
            + kv_html([
                ("Age", results["age"]),
                ("Sex", results["gender"]),
                ("Ancestry", results["ancestry"]),
                ("View", results["resolved_view"] or "—"),
            ])
            + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card"><p class="eyebrow">CARDIOTHORACIC RATIO</p>'
            + radial_gauge_svg(getattr(report, "resolved_ctr", None), thresh, fill_color)
            + '<div class="gauge-cap">Filled arc is the resolved ratio · '
            "dark tick is the threshold</div></div>",
            unsafe_allow_html=True,
        )

    with col_b:
        with st.container(border=True):
            st.markdown('<p class="eyebrow">CHEST X-RAY · TOOL B SEGMENTATION</p>',
                        unsafe_allow_html=True)
            fig = build_overlay_fig(result_b["image"], result_b["mask"])
            st.pyplot(fig, use_container_width=True)
            st.markdown(
                f'<div class="legend">'
                f'<div class="item"><span class="sw" style="background:{LUNG_C}"></span>Lung fields</div>'
                f'<div class="item"><span class="sw" style="background:{HEART_C}"></span>Cardiac silhouette</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, facecolor=LIGHTBOX, bbox_inches="tight")
            plt.close(fig)
            st.download_button(
                "Download overlay",
                data=buf.getvalue(),
                file_name=f"overlay_{results['source_name']}.png",
                mime="image/png",
            )

    with col_c:
        bars = (
            bar_row("Tool A CTR", getattr(vision_a, "ctr", None), "A", CLAY_LT, thresh)
            + bar_row("Tool B CTR", getattr(vision_b, "ctr", None), "B", CLAY, thresh)
            + bar_row("Resolved CTR", getattr(report, "resolved_ctr", None), "R", fill_color, thresh)
            + bar_row("Threshold", thresh, "T", INK)
        )
        st.markdown(
            '<div class="card"><p class="eyebrow">MEASUREMENTS</p>' + bars
            + f'<div class="bar-note">Bars scaled {SCALE_MIN:.0f}–{SCALE_MAX:.0f}%</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="card"><p class="eyebrow">CLINICAL RATIONALE</p>'
        f'<div class="rationale">{report.clinical_rationale}</div></div>',
        unsafe_allow_html=True,
    )

    checks = report.gate_checks
    passed = sum(1 for c in checks if c["passed"])
    if st.checkbox(f"Safety gate checks — {passed} of {len(checks)} passed", value=False):
        st.markdown(gate_table_html(checks), unsafe_allow_html=True)

    disclaimer = getattr(report, "disclaimer", None)
    if disclaimer:
        st.markdown(f'<div class="disclaimer">{disclaimer}</div>', unsafe_allow_html=True)
