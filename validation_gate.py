"""
Layer-2 Safety Veto Gate -- SCAFFOLD / PLACEHOLDER
----------------------------------------------------
NOTE: The real validation_gate.py was never provided in this conversation.
This is a reasonable reconstruction based on how pipeline.py and main.py
call it (VisionResult, run_gate -> list of checks with .passed/.name/.detail),
and the Safety Veto Protocol described in the Cardiomegaly_agent_report.pdf
(model confidence verification, view/projection check, tool cross-agreement).

Replace this file with your actual validation_gate.py before running the
dashboard or notebook against real data. The tool cross-agreement check is
already handled directly in pipeline.py, so it is NOT duplicated here to
avoid a redundant/conflicting veto -- adjust if your real gate also checks it.
"""

from dataclasses import dataclass
from typing import Optional, List

MIN_MODEL_CONFIDENCE = 0.50


@dataclass
class VisionResult:
    ctr: Optional[float]
    confidence: Optional[float]
    tool_name: str


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str


def run_gate(
    result_a: VisionResult,
    result_b: VisionResult,
    view: Optional[str],
    inspiration_adequate: Optional[bool] = None,
    rotation_acceptable: Optional[bool] = None,
) -> List[GateCheck]:
    checks: List[GateCheck] = []

    # Metadata integrity: view must be PA or AP
    view_ok = view in ("PA", "AP")
    checks.append(GateCheck(
        name="view_metadata",
        passed=view_ok,
        detail="View recorded as PA/AP." if view_ok else f"View '{view}' missing or unrecognized.",
    ))

    # Model confidence verification
    for result in (result_a, result_b):
        conf_ok = result.confidence is not None and result.confidence >= MIN_MODEL_CONFIDENCE
        checks.append(GateCheck(
            name=f"{result.tool_name.lower()}_confidence",
            passed=conf_ok,
            detail=(
                f"{result.tool_name} confidence {result.confidence} OK."
                if conf_ok
                else f"{result.tool_name} confidence {result.confidence} below minimum {MIN_MODEL_CONFIDENCE}."
            ),
        ))

    # Inspiration depth (if supplied)
    if inspiration_adequate is not None:
        checks.append(GateCheck(
            name="inspiration_depth",
            passed=bool(inspiration_adequate),
            detail="Inspiration adequate." if inspiration_adequate else "Inadequate inspiration depth reported.",
        ))

    # Rotation (if supplied)
    if rotation_acceptable is not None:
        checks.append(GateCheck(
            name="rotation",
            passed=bool(rotation_acceptable),
            detail="Rotation within acceptable range." if rotation_acceptable else "Patient rotation exceeds acceptable range.",
        ))

    return checks
