from dataclasses import dataclass
from typing import Any


STATUS_FIELD_NAME = "AI-Processing Status"
STATUS_VALUES = ("Queued", "Done", "Failed")

PDF_ANALYSIS_MAX_PAGES = 1
PDF_SUBSTANTIAL_TEXT_CHARS = 100
PDF_IMAGE_DOMINANT_RATIO = 0.8
PDF_DOCUMENT_CLASSIFY_RATIO = 0.7
TITLE_SOURCE_MAX_CHARS = 2000
TITLE_MAX_LENGTH = 120


@dataclass
class BatchSummary:
    seen: int = 0
    eligible: int = 0
    processed_ok: int = 0
    failed: int = 0
    skipped: int = 0


@dataclass
class DocumentPreview:
    document_id: int
    current_title: str
    proposed_title: str
    language: str | None
    confidence: float | None
    content_preview: str
    overwrite_content: bool
    processing_mode: str
    pdf_mode: str | None
    detection_reason: str


@dataclass
class ProposedTitle:
    title: str
    language: str | None
    confidence: float | None
    content_preview: str
    content_to_save: str
    overwrite_content: bool
    processing_mode: str
    pdf_mode: str | None
    detection_reason: str
