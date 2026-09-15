"""
Clinical Pipeline Orchestration Module (v2.2)
---------------------------------------------
Wires DICOM metadata parsing, Tool A/B cross-agreement check, Layer-1 rule
evaluation, and Layer-2 validation vetoes.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

from decision_engine import PatientInput, evaluate, Verdict, EvidenceTier
from validation_gate import run_gate, VisionResult

# Maximum allowed disagreement between Tool A and Tool B CTR estimates
# before the case is escalated instead of resolved automatically.
TOOL_AGREEMENT_THRESHOLD_PCT = 3.0


@dataclass
class PatientMetadata:
    age: Optional[int]
    gender: Optional[str]
    projection: Optional[str]
    ancestry: Optional[str]


def parse_metadata(meta_dict: Dict[str, Any]) -> PatientMetadata:

    # 1. Age Extraction (Handles DICOM '054Y' string formats & raw integers)
    raw_age = meta_dict.get("PatientAge", meta_dict.get("age"))
    age: Optional[int] = None

    if isinstance(raw_age, str):
        cleaned_age = raw_age.strip().rstrip("Yy")
        if cleaned_age.isdigit():
            age = int(cleaned_age)
    elif isinstance(raw_age, int):
        age = raw_age

    # 2. Gender Mapping (Fixes DICOM 'M'/'F' tags with fallback key checks)
    sex_map = {"M": "Male", "F": "Female", "MALE": "Male", "FEMALE": "Female"}
    raw_sex = meta_dict.get("PatientSex") or meta_dict.get("gender")

    if isinstance(raw_sex, str):
        raw_sex_clean = raw_sex.strip().upper()
        gender = sex_map.get(raw_sex_clean, raw_sex.strip().title())
    else:
        gender = None

    # 3. Projection / View Position Extraction
    raw_projection = meta_dict.get("ViewPosition") or meta_dict.get("view")
    projection: Optional[str] = None
    if raw_projection is not None:
        projection = str(raw_projection).strip().upper()

    # 4. Ancestry Mapping
    ancestry_raw = meta_dict.get("ancestry")
    ancestry: Optional[str] = None

    if ancestry_raw is not None:
        normalized_ancestry = str(ancestry_raw).strip().lower()
        if normalized_ancestry in ("caucasian", "c"):
            ancestry = "Caucasian"
        elif normalized_ancestry in ("non-caucasian", "non caucasian", "noncaucasian", "nc"):
            ancestry = "Non-Caucasian"
        else:
            ancestry = str(ancestry_raw).strip()

    return PatientMetadata(
        age=age,
        gender=gender,
        projection=projection,
        ancestry=ancestry
    )


@dataclass
class PipelineReport:
    final_verdict: str
    demographic_verdict: str
    overridden_by_gate: bool
    rule_id: str
    evidence_tier: str
    raw_ctr_tool_a: Optional[float]
    raw_ctr_tool_b: Optional[float]
    resolved_ctr: Optional[float]
    tool_agreement_pct_diff: Optional[float]
    tool_disagreement_flag: bool
    threshold_center: Optional[float]
    gate_checks: list
    clinical_rationale: str
    disclaimer: str


def _check_tool_agreement(result_a: VisionResult, result_b: VisionResult):
    """
    Compares Tool A and Tool B CTR estimates.

    Returns (resolved_ctr, pct_diff, disagreement_flag).
    If either estimate is missing, disagreement_flag is True and
    resolved_ctr falls back to whichever estimate is available.
    """
    ctr_a, ctr_b = result_a.ctr, result_b.ctr

    if ctr_a is None or ctr_b is None:
        resolved = ctr_a if ctr_a is not None else ctr_b
        return resolved, None, True

    pct_diff = abs(ctr_a - ctr_b)
    if pct_diff > TOOL_AGREEMENT_THRESHOLD_PCT:
        return None, pct_diff, True

    resolved = (ctr_a + ctr_b) / 2.0
    return resolved, pct_diff, False


def process_patient(
    metadata_source: Dict[str, Any],
    result_a: VisionResult,
    result_b: VisionResult,
    inspiration_adequate: Optional[bool] = None,
    rotation_acceptable: Optional[bool] = None,
) -> PipelineReport:

    # Parse & normalize DICOM metadata
    meta: PatientMetadata = parse_metadata(metadata_source)

    # Tool A / Tool B cross-agreement check (>3% diff -> escalate)
    resolved_ctr, pct_diff, tool_disagreement = _check_tool_agreement(result_a, result_b)

    if tool_disagreement:
        # Skip demographic evaluation entirely -- there is no trustworthy
        # CTR to evaluate against a threshold yet.
        demo_verdict_value = Verdict.FLAG_FOR_REVIEW.value
        rule_id = "T1"
        evidence_tier_value = EvidenceTier.C.value
        threshold_center = None
        diff_str = f"{pct_diff:.1f}%" if pct_diff is not None else "N/A (missing estimate)"
        rationale = (
            f"Tool cross-agreement check failed: Tool A CTR={result_a.ctr}, "
            f"Tool B CTR={result_b.ctr}, difference={diff_str} exceeds "
            f"{TOOL_AGREEMENT_THRESHOLD_PCT:.1f}% agreement threshold. "
            f"Flagged for review before demographic thresholding."
        )
        checks = run_gate(
            result_a=result_a,
            result_b=result_b,
            view=meta.projection,
            inspiration_adequate=inspiration_adequate,
            rotation_acceptable=rotation_acceptable,
        )
        return PipelineReport(
            final_verdict=Verdict.FLAG_FOR_REVIEW.value,
            demographic_verdict=demo_verdict_value,
            overridden_by_gate=False,
            rule_id=rule_id,
            evidence_tier=evidence_tier_value,
            raw_ctr_tool_a=result_a.ctr,
            raw_ctr_tool_b=result_b.ctr,
            resolved_ctr=None,
            tool_agreement_pct_diff=pct_diff,
            tool_disagreement_flag=True,
            threshold_center=threshold_center,
            gate_checks=[asdict(c) for c in checks],
            clinical_rationale=rationale,
            disclaimer=(
                "This CTR-based verdict is a decision-support signal only. "
                "It is intended to support, not replace, interpretation by a "
                "qualified radiologist or physician."
            ),
        )

    # Layer 1: Rule Engine Evaluation (using the resolved/averaged CTR)
    patient = PatientInput(
        ancestry=meta.ancestry,
        gender=meta.gender,
        age=meta.age,
        ctr=resolved_ctr,
        view=meta.projection,
    )
    demo_result = evaluate(patient)

    # Layer 2: Safety Gate Execution
    checks = run_gate(
        result_a=result_a,
        result_b=result_b,
        view=meta.projection,
        inspiration_adequate=inspiration_adequate,
        rotation_acceptable=rotation_acceptable,
    )
    failed_checks = [c for c in checks if not c.passed]

    # Evaluate Safety Gate Vetoes
    if failed_checks:
        final_verdict = Verdict.FLAG_FOR_REVIEW
        overridden = demo_result.verdict != Verdict.FLAG_FOR_REVIEW
        reasons = "; ".join(f"[{c.name.upper()} FAULT]: {c.detail}" for c in failed_checks)
        rationale = (
            f"Layer-1 Rule {demo_result.rule_id} computed '{demo_result.verdict.value}', "
            f"but overridden by Layer-2 Safety Veto: {reasons}"
        )
    else:
        final_verdict = demo_result.verdict
        overridden = False
        rationale = demo_result.message

    return PipelineReport(
        final_verdict=final_verdict.value,
        demographic_verdict=demo_result.verdict.value,
        overridden_by_gate=overridden,
        rule_id=demo_result.rule_id,
        evidence_tier=demo_result.evidence_tier.value,
        raw_ctr_tool_a=result_a.ctr,
        raw_ctr_tool_b=result_b.ctr,
        resolved_ctr=resolved_ctr,
        tool_agreement_pct_diff=pct_diff,
        tool_disagreement_flag=False,
        threshold_center=demo_result.threshold_center,
        gate_checks=[asdict(c) for c in checks],
        clinical_rationale=rationale,
        disclaimer=demo_result.disclaimer,
    )
