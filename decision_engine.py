"""
CTR Rule-Based Decision Engine (v2.0)

"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import math

CLINICAL_DISCLAIMER = (
    "This CTR-based verdict is a decision-support signal only. It is intended "
    "to support, not replace, interpretation by a qualified radiologist or physician. "
    "It does not account for rotation, inspiration depth, or magnification artifacts "
    "beyond the fixed AP offset applied here, unless those are supplied as separate inputs."
)

class Verdict(Enum):
    NO_CARDIOMEGALY = "No cardiomegaly"
    CARDIOMEGALY_PRESENT = "Cardiomegaly present"
    FLAG_FOR_REVIEW = "Flag for review"

class EvidenceTier(Enum):
    A = "high"          # Brakohiapa et al., n=1047 (age & sex-specific)
    B = "weak"          # Caucasian fixed baseline (no primary study traced)
    C = "missing_data"  # Incomplete, out-of-range, or implausible parameters

RULE_TABLE_VERSION = "2.0"
BAND_WIDTH_PCT = 0.5     # +/- 0.5% uncertainty band around cutoff
AP_OFFSET_PCT = 5.0      # Kabala & Wilde AP magnification offset
MIN_PLAUSIBLE_AGE = 21
MAX_PLAUSIBLE_AGE = 110
VALID_VIEWS = {"PA", "AP"}

@dataclass
class PatientInput:
    ancestry: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    ctr: Optional[float] = None
    view: Optional[str] = "PA"

@dataclass
class VerdictResult:
    verdict: Verdict
    rule_id: str
    threshold_center: Optional[float]
    base_pa_threshold: Optional[float]
    band_lower: Optional[float]
    band_upper: Optional[float]
    evidence_tier: EvidenceTier
    source: str
    message: str
    view_applied: Optional[str] = None
    ancestry_was_defaulted: bool = False
    disclaimer: str = CLINICAL_DISCLAIMER

_THRESHOLDS = {
    ("Non-Caucasian", "Male", "any"):     (50.0, ("A1", "A2", "A3"), "Brakohiapa et al. 2017, n=1047 (Male, all adult ages)"),
    ("Non-Caucasian", "Female", "21-40"): (50.0, ("A4", "A5", "A6"), "Brakohiapa et al. 2017, n=1047 (Female, 21-40y)"),
    ("Non-Caucasian", "Female", "41-60"): (52.0, ("A7", "A8", "A9"), "Brakohiapa et al. 2017, n=1047 (Female, 41-60y)"),
    ("Non-Caucasian", "Female", ">60"):   (53.0, ("A10", "A11", "A12"), "Brakohiapa et al. 2017, n=1047 (Female, >60y)"),
    ("Caucasian", "Male", "any"):        (50.0, ("B1", "B2", "B3"), "Regional/secondary literature baseline"),
    ("Caucasian", "Female", "any"):      (50.0, ("B1", "B2", "B3"), "Regional/secondary literature baseline"),
}

def _age_bracket(gender: str, age: int) -> str:
    if gender == "Male":
        return "any"
    if age <= 40:
        return "21-40"
    elif age <= 60:
        return "41-60"
    else:
        return ">60"

def _flag(rule_id: str, source: str, message: str, view_applied: Optional[str] = None,
          ancestry_was_defaulted: bool = False) -> VerdictResult:
    return VerdictResult(
        verdict=Verdict.FLAG_FOR_REVIEW,
        rule_id=rule_id,
        threshold_center=None,
        base_pa_threshold=None,
        band_lower=None,
        band_upper=None,
        evidence_tier=EvidenceTier.C,
        source=source,
        message=message,
        view_applied=view_applied,
        ancestry_was_defaulted=ancestry_was_defaulted,
    )

def evaluate(patient: PatientInput) -> VerdictResult:
    ancestry_was_defaulted = False
    ancestry = patient.ancestry

    if ancestry is None:
        ancestry = "Non-Caucasian"
        ancestry_was_defaulted = True
    elif ancestry not in ("Caucasian", "Non-Caucasian"):
        return _flag("C1", "N/A -- unrecognized ancestry", f"Ancestry '{patient.ancestry}' not recognized.")

    if patient.gender is None or patient.age is None:
        return _flag("C1", "N/A -- missing demographics", "Gender or age missing. Flagged for review.")

    if patient.ctr is None:
        return _flag("C1", "N/A -- missing CTR", "CTR measurement unavailable (tool disagreement or extraction failure). Flagged for review.")

    if isinstance(patient.ctr, float) and (math.isnan(patient.ctr) or math.isinf(patient.ctr)):
        return _flag("C1", "N/A -- invalid numeric CTR", "CTR value is NaN or infinite.")

    if patient.age < MIN_PLAUSIBLE_AGE or patient.age > MAX_PLAUSIBLE_AGE:
        return _flag("C1", "N/A -- out-of-range age", f"Age {patient.age} outside supported adult range ({MIN_PLAUSIBLE_AGE}-{MAX_PLAUSIBLE_AGE}).")

    if not (20.0 <= patient.ctr <= 90.0):
        return _flag("C1", "N/A -- unphysiological CTR", f"CTR {patient.ctr}% is outside physiological bounds (20-90%).")

    view = patient.view
    if view is None or view.upper() not in VALID_VIEWS:
        return _flag("C1", "N/A -- projection unverified", f"View '{patient.view}' invalid or missing.")
    view = view.upper()

    bracket = _age_bracket(patient.gender, patient.age) if ancestry == "Non-Caucasian" else "any"
    key = (ancestry, patient.gender, bracket)

    if key not in _THRESHOLDS:
        return _flag("C1", "N/A -- unmapped demographic rule", f"No rule mapping found for key {key}.")

    base_pa_center, (rule_normal, rule_band, rule_abnormal), source = _THRESHOLDS[key]
    evidence_tier = EvidenceTier.A if ancestry == "Non-Caucasian" else EvidenceTier.B

    center = base_pa_center + AP_OFFSET_PCT if view == "AP" else base_pa_center
    if view == "AP":
        source += " + AP Magnification Offset (+5.0%)"

    lower = center - BAND_WIDTH_PCT
    upper = center + BAND_WIDTH_PCT
    ap_note = f" [AP-adjusted from {base_pa_center:.1f}%]" if view == "AP" else ""

    if patient.ctr < lower:
        verdict = Verdict.NO_CARDIOMEGALY
        rule_id = rule_normal
        msg = f"CTR {patient.ctr:.1f}% is below cutoff ({center:.1f}% threshold{ap_note}, safe floor <{lower:.1f}%)."
    elif patient.ctr > upper:
        verdict = Verdict.CARDIOMEGALY_PRESENT
        rule_id = rule_abnormal
        msg = f"CTR {patient.ctr:.1f}% exceeds abnormal cutoff ({upper:.1f}% upper band edge, threshold {center:.1f}%{ap_note})."
    else:
        verdict = Verdict.FLAG_FOR_REVIEW
        rule_id = rule_band
        msg = f"CTR {patient.ctr:.1f}% falls within uncertainty band ({lower:.1f}%\u2013{upper:.1f}%) around {center:.1f}% threshold{ap_note}."

    if evidence_tier == EvidenceTier.B:
        msg += " [Tier-B Evidence Notice: Caucasian fixed threshold applied]"
    if ancestry_was_defaulted:
        msg += " [Policy Notice: Ancestry defaulted to Non-Caucasian]"

    return VerdictResult(
        verdict=verdict, rule_id=rule_id, threshold_center=center,
        base_pa_threshold=base_pa_center, band_lower=lower, band_upper=upper,
        evidence_tier=evidence_tier, source=source, message=msg,
        view_applied=view, ancestry_was_defaulted=ancestry_was_defaulted,
    )
