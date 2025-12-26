from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from alice.mediator import mediate
from alice.types import (
    AliceResponse,
    AliceUtterance,
    AliceMove,
    BoundaryNotice,
    BoundaryType,
    Confidence,
    ConfidenceLevel,
    IsacSignal,
    IsacSignalKind,
)

router = APIRouter(prefix="/alice/preview", tags=["alice-preview"])


# ---------------------------------------------------------------------
# Existing status endpoint (UNCHANGED BEHAVIOR)
# ---------------------------------------------------------------------

@router.get("/status")
def alice_preview_status():
    return {
        "status": "ok",
        "mode": "preview",
        "authority": "read-only",
        "notes": [
            "Alice is not yet active",
            "No execution paths enabled",
            "Phase 8 preview surface only",
        ],
    }


# ---------------------------------------------------------------------
# Pydantic models (preview-only I/O surface)
# ---------------------------------------------------------------------

class ConfidenceModel(BaseModel):
    level: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class BoundaryNoticeModel(BaseModel):
    boundary_type: BoundaryType
    source: str
    message: str
    allowed_alternatives: List[str] = Field(default_factory=list)


class IsacSignalModel(BaseModel):
    kind: IsacSignalKind
    title: str
    summary: str
    confidence: ConfidenceModel = Field(default_factory=ConfidenceModel)
    boundaries: List[BoundaryNoticeModel] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[str] = None

    def to_domain(self) -> IsacSignal:
        return IsacSignal(
            kind=self.kind,
            title=self.title,
            summary=self.summary,
            confidence=Confidence(
                level=self.confidence.level,
                score=self.confidence.score,
            ),
            boundaries=[
                BoundaryNotice(
                    boundary_type=b.boundary_type,
                    source=b.source,
                    message=b.message,
                    allowed_alternatives=b.allowed_alternatives,
                )
                for b in self.boundaries
            ],
            data=self.data,
            trace_id=self.trace_id,
        )


class AliceUtteranceModel(BaseModel):
    text: str


class AliceResponseModel(BaseModel):
    move: AliceMove
    utterances: List[AliceUtteranceModel] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    options: List[str] = Field(default_factory=list)
    boundary_echo: List[BoundaryNoticeModel] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, resp: AliceResponse) -> "AliceResponseModel":
        return cls(
            move=resp.move,
            utterances=[AliceUtteranceModel(text=u.text) for u in resp.utterances],
            questions=resp.questions,
            options=resp.options,
            boundary_echo=[
                BoundaryNoticeModel(
                    boundary_type=b.boundary_type,
                    source=b.source,
                    message=b.message,
                    allowed_alternatives=b.allowed_alternatives,
                )
                for b in resp.boundary_echo
            ],
            meta=resp.meta,
        )


# ---------------------------------------------------------------------
# Preview mediation endpoint (READ-ONLY)
# ---------------------------------------------------------------------

@router.post("/mediate", response_model=AliceResponseModel)
def alice_preview_mediate(signal: IsacSignalModel):
    """
    Preview-only mediation endpoint.

    HARD GUARANTEES:
    - No persistence
    - No learning
    - No execution
    - No /ask wiring
    - Finance excluded by design
    """
    domain_signal = signal.to_domain()
    response = mediate(domain_signal)
    return AliceResponseModel.from_domain(response)
