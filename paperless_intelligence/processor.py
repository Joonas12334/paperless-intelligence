import base64
import io
import re
import time
from typing import Any, Callable

from .api import OllamaApi, PaperlessApi
from .config import PaperlessServerConfig, Settings
from .types import (
    STATUS_FIELD_NAME,
    STATUS_VALUES,
    BatchSummary,
    DocumentPreview,
    ProposedTitle,
    TITLE_MAX_LENGTH,
)

PDF_ANALYSIS_MAX_PAGES = 1
PDF_SUBSTANTIAL_TEXT_CHARS = 100
PDF_IMAGE_DOMINANT_RATIO = 0.8
PDF_DOCUMENT_CLASSIFY_RATIO = 0.7

try:
    import fitz
except ImportError:
    fitz = None

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None


class Processor:
    def __init__(
        self,
        settings: Settings,
        server: PaperlessServerConfig,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.server = server
        self.paperless = PaperlessApi(server, settings.request_timeout_seconds)
        self.ollama = OllamaApi(settings.ollama_base_url, settings.request_timeout_seconds)
        self.progress = progress

    def _progress(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def bootstrap_validate(
        self,
    ) -> tuple[bool, list[str], list[str], dict[str, int], dict[str, dict[str, str]]]:
        fields_by_name: dict[str, dict[str, Any]] = {}

        for field in self.paperless.iter_custom_fields(page_size=200):
            name = str(field.get("name", "")).strip()
            if name:
                fields_by_name[name] = field

        missing_fields: list[str] = []
        invalid_fields: list[str] = []
        custom_field_ids: dict[str, int] = {}
        select_option_ids: dict[str, dict[str, str]] = {}

        status_field = fields_by_name.get(STATUS_FIELD_NAME)
        if not status_field:
            missing_fields.append(STATUS_FIELD_NAME)
        else:
            field_id = status_field.get("id")
            if isinstance(field_id, int):
                custom_field_ids["status"] = field_id
            else:
                invalid_fields.append(f"{STATUS_FIELD_NAME}: missing field id")

            field_type = str(status_field.get("data_type", "")).strip().lower()
            if field_type != "select":
                invalid_fields.append(f"{STATUS_FIELD_NAME}: must be select field")

            actual_options = set(self._extract_select_options(status_field))
            expected_options = set(STATUS_VALUES)
            if actual_options != expected_options:
                invalid_fields.append(
                    f"{STATUS_FIELD_NAME}: expected options {sorted(expected_options)}, got {sorted(actual_options)}"
                )
            else:
                select_option_ids["status"] = self._extract_select_option_ids(status_field)

        ready = (not missing_fields) and (not invalid_fields)
        return (
            ready,
            missing_fields,
            invalid_fields,
            custom_field_ids,
            select_option_ids,
        )

    def run_batch_once(
        self,
        *,
        custom_field_id: int,
        status_option_ids: dict[str, str],
    ) -> BatchSummary:
        summary = BatchSummary()
        self._progress(f"[{self.server.name}] Starting batch scan...")

        for doc in self.paperless.iter_documents(page_size=100, ordering="-added"):
            summary.seen += 1
            doc_id = int(doc.get("id", 0) or 0)
            label = self._document_label(doc)
            self._progress(f"[{self.server.name}] [{summary.seen}] Checking {label}")
            if doc_id < 1:
                summary.skipped += 1
                self._progress(f"[{self.server.name}] [{summary.seen}] Skipped invalid document entry")
                continue

            if not self._is_eligible(
                doc,
                custom_field_id=custom_field_id,
                status_option_ids=status_option_ids,
            ):
                summary.skipped += 1
                self._progress(f"[{self.server.name}] [{summary.seen}] Skipped {label} (already Done)")
                continue

            summary.eligible += 1
            self._progress(
                f"[{self.server.name}] [{summary.seen}] Processing {label} ({summary.eligible} eligible so far)"
            )
            ok = self._process_document(
                doc_id,
                custom_field_id=custom_field_id,
                status_option_ids=status_option_ids,
            )
            if ok:
                summary.processed_ok += 1
                self._progress(f"[{self.server.name}] [{summary.seen}] Saved {label}")
            else:
                summary.failed += 1
                self._progress(f"[{self.server.name}] [{summary.seen}] Failed {label}")

        return summary

    def preview_document(self, document_id: int) -> DocumentPreview:
        self._progress(f"[{self.server.name}] Loading document #{document_id}...")
        doc = self.paperless.get_document(document_id)

        current_title = self._clean_title(str(doc.get("title", "")))
        proposed = self._propose_title(doc)
        self._progress(f"[{self.server.name}] Preview ready for document #{document_id}.")

        return DocumentPreview(
            document_id=document_id,
            current_title=current_title,
            proposed_title=proposed.title,
            language=proposed.language,
            confidence=proposed.confidence,
            content_preview=proposed.content_preview,
            overwrite_content=proposed.overwrite_content,
            processing_mode=proposed.processing_mode,
            pdf_mode=proposed.pdf_mode,
            detection_reason=proposed.detection_reason,
        )

    def save_preview(
        self,
        document_id: int,
        proposed_title: str,
        proposed_content: str,
        *,
        custom_field_id: int,
        status_option_ids: dict[str, str],
        overwrite_content: bool,
    ) -> str:
        self._progress(f"[{self.server.name}] Preparing to save document #{document_id}...")

        clean_title = self._sanitize_title(proposed_title)
        if not clean_title:
            raise RuntimeError("Proposed title is empty")
        clean_content = self._sanitize_ocr_text(proposed_content)
        if not clean_content:
            raise RuntimeError("Proposed content is empty")

        doc = self.paperless.get_document(document_id)
        current_tags = sorted(self._extract_tag_ids(doc))

        try:
            self._progress(f"[{self.server.name}] Updating title for document #{document_id}...")
            self.paperless.update_title(document_id, clean_title, tags=current_tags)
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to update the document title in Paperless: {exc}") from exc
        if overwrite_content:
            try:
                self._progress(f"[{self.server.name}] Overwriting OCR content for document #{document_id}...")
                self.paperless.update_content(document_id, clean_content, tags=current_tags)
            except RuntimeError as exc:
                raise RuntimeError(f"Failed to overwrite OCR content in Paperless: {exc}") from exc
        else:
            self._progress(f"[{self.server.name}] Skipping OCR overwrite for document #{document_id}...")

        self._progress(f"[{self.server.name}] Marking document #{document_id} as Done...")
        self._update_status(
            document_id,
            custom_field_id=custom_field_id,
            status_name="Done",
            status_option_ids=status_option_ids,
            tags=current_tags,
        )
        return clean_title

    def _process_document(
        self,
        document_id: int,
        *,
        custom_field_id: int,
        status_option_ids: dict[str, str],
    ) -> bool:
        initial_doc = self.paperless.get_document(document_id)
        current_tags = sorted(self._extract_tag_ids(initial_doc))

        self._progress(f"[{self.server.name}] Document #{document_id}: Setting status to Queued...")
        self._update_status(
            document_id,
            custom_field_id=custom_field_id,
            status_name="Queued",
            status_option_ids=status_option_ids,
            tags=current_tags,
        )

        retries = max(1, int(self.settings.max_retries))
        for attempt in range(1, retries + 1):
            try:
                self._progress(f"[{self.server.name}] Document #{document_id}: attempt {attempt}/{retries}")
                fresh_doc = self.paperless.get_document(document_id)
                current_tags = sorted(self._extract_tag_ids(fresh_doc))
                proposed = self._propose_title(fresh_doc)
                self._progress(f"[{self.server.name}] Document #{document_id}: saving title")
                self.paperless.update_title(document_id, proposed.title, tags=current_tags)
                if proposed.overwrite_content:
                    self._progress(f"[{self.server.name}] Document #{document_id}: overwriting OCR content")
                    self.paperless.update_content(document_id, proposed.content_to_save, tags=current_tags)
                else:
                    self._progress(f"[{self.server.name}] Document #{document_id}: skipping OCR overwrite")
                self._progress(f"[{self.server.name}] Document #{document_id}: marking status Done")
                self._update_status(
                    document_id,
                    custom_field_id=custom_field_id,
                    status_name="Done",
                    status_option_ids=status_option_ids,
                    tags=current_tags,
                )
                return True
            except Exception as exc:
                self._progress(f"[{self.server.name}] Document #{document_id}: attempt {attempt} failed: {exc}")
                if self._is_ollama_timeout_error(exc):
                    break
                if attempt < retries:
                    time.sleep(min(30, 2**attempt))

        self._progress(f"[{self.server.name}] Document #{document_id}: marking status Failed")
        self._update_status(
            document_id,
            custom_field_id=custom_field_id,
            status_name="Failed",
            status_option_ids=status_option_ids,
            tags=current_tags,
        )
        return False

    def _propose_title(self, doc: dict[str, Any]) -> ProposedTitle:
        document_id = int(doc.get("id", 0) or 0)
        if document_id < 1:
            raise RuntimeError("Invalid document id")

        context = self._build_context(doc)
        pdf_mode: str | None = None
        detection_reason = "Original file is not a PDF. Using Ollama OCR."
        if self._document_may_be_pdf(doc):
            self._progress(f"[{self.server.name}] Document #{document_id}: inspecting original PDF")
            pdf_mode = self._classify_original_pdf(document_id)
            if pdf_mode is not None:
                self._progress(f"[{self.server.name}] Document #{document_id}: detected PDF mode {pdf_mode}")
                detection_reason = self._describe_pdf_mode(pdf_mode)
            else:
                detection_reason = "Original file is not a PDF. Using Ollama OCR."

        if pdf_mode == "original_digital_pdf":
            source_text = self._paperless_content_for_title(doc)
            candidate = self._run_ollama_with_timeout_fallback(
                document_id=document_id,
                mode_label="text title generation",
                runner=lambda model: self.ollama.generate_title_from_text(
                    model=model,
                    text=source_text,
                    context=context,
                    title_hint=self.settings.resolved_ollama_title_prompt_hint,
                ),
            )
            title = self._sanitize_title(str(candidate.get("title", "")))
            if not title:
                raise RuntimeError("Model returned empty title")
            confidence = self._normalize_confidence(candidate.get("confidence"))
            return ProposedTitle(
                title=title,
                language=str(candidate.get("language", "")).strip() or None,
                confidence=confidence,
                content_preview=source_text,
                content_to_save=source_text,
                overwrite_content=False,
                processing_mode="paperless_text_title_only",
                pdf_mode=pdf_mode,
                detection_reason=detection_reason,
            )

        self._progress(f"[{self.server.name}] Document #{document_id}: loading source image")
        image_b64 = self._load_vision_image_base64(document_id)
        if not image_b64:
            raise RuntimeError(
                "Could not prepare a document image for Ollama OCR. "
                "For image documents, make sure the original file can be downloaded. "
                "For PDFs, Paperless thumbnail generation must be available."
            )

        candidate = self._run_ollama_with_timeout_fallback(
            document_id=document_id,
            mode_label="vision OCR/title generation",
            runner=lambda model: self.ollama.generate_title(
                model=model,
                image_base64=image_b64,
                context=context,
                system_prompt=self.settings.ollama_system_prompt,
                title_hint=self.settings.resolved_ollama_title_prompt_hint,
            ),
        )
        title = self._sanitize_title(str(candidate.get("title", "")))
        if not title:
            raise RuntimeError("Model returned empty title")
        ocr_text = self._sanitize_ocr_text(str(candidate.get("ocr_text", "")))
        if not ocr_text:
            raise RuntimeError("Model returned empty OCR text")

        confidence = self._normalize_confidence(candidate.get("confidence"))

        return ProposedTitle(
            title=title,
            language=str(candidate.get("language", "")).strip() or None,
            confidence=confidence,
            content_preview=ocr_text,
            content_to_save=ocr_text,
            overwrite_content=True,
            processing_mode="ollama_ocr",
            pdf_mode=pdf_mode,
            detection_reason=detection_reason,
        )

    def _build_context(self, doc: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in ("original_file_name", "archive_serial_number", "correspondent", "document_type"):
            value = doc.get(key)
            if value:
                parts.append(f"{key}={value}")
        return "; ".join(parts)

    def _run_ollama_with_timeout_fallback(
        self,
        *,
        document_id: int,
        mode_label: str,
        runner: Callable[[str], dict[str, Any]],
    ) -> dict[str, Any]:
        primary_model = self.settings.ollama_model.strip()
        fallback_model = self.settings.ollama_fallback_model.strip()

        self._progress(f"[{self.server.name}] Document #{document_id}: using Ollama model {primary_model} for {mode_label}")
        try:
            return runner(primary_model)
        except Exception as exc:
            if not self._is_ollama_timeout_error(exc):
                raise

            self._progress(
                f"[{self.server.name}] Document #{document_id}: primary model {primary_model} timed out, retrying with {fallback_model}"
            )
            try:
                self._progress(f"[{self.server.name}] Document #{document_id}: using Ollama fallback model {fallback_model} for {mode_label}")
                return runner(fallback_model)
            except Exception as fallback_exc:
                if self._is_ollama_timeout_error(fallback_exc):
                    raise RuntimeError(
                        f"Ollama timed out for document {document_id} on both models ({primary_model} and {fallback_model})."
                    ) from fallback_exc
                raise

    def _is_ollama_timeout_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return "ollama did not respond within" in text or "ollama did not respond in time" in text

    def _normalize_confidence(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _is_eligible(
        self,
        doc: dict[str, Any],
        *,
        custom_field_id: int,
        status_option_ids: dict[str, str],
    ) -> bool:
        custom_fields = self._extract_custom_field_map(doc)
        status = self._normalize_status_value(custom_fields.get(custom_field_id), status_option_ids)
        return status != "Done"

    def _document_may_be_pdf(self, doc: dict[str, Any]) -> bool:
        mime_type = str(doc.get("mime_type", "")).strip().lower()
        if "pdf" in mime_type:
            return True

        for key in ("original_file_name", "archived_file_name", "title"):
            value = str(doc.get(key, "")).strip().lower()
            if value.endswith(".pdf"):
                return True

        return False

    def _extract_select_options(self, field: dict[str, Any]) -> list[str]:
        options: list[str] = []
        direct = field.get("select_options")
        if isinstance(direct, list):
            options.extend(self._normalize_options(direct))

        extra = field.get("extra_data")
        if isinstance(extra, dict):
            nested = extra.get("select_options")
            if isinstance(nested, list):
                options.extend(self._normalize_options(nested))

        seen: set[str] = set()
        result: list[str] = []
        for option in options:
            if option not in seen:
                seen.add(option)
                result.append(option)
        return result

    def _extract_select_option_ids(self, field: dict[str, Any]) -> dict[str, str]:
        options: dict[str, str] = {}
        raw_groups: list[list[Any]] = []

        direct = field.get("select_options")
        if isinstance(direct, list):
            raw_groups.append(direct)

        extra = field.get("extra_data")
        if isinstance(extra, dict):
            nested = extra.get("select_options")
            if isinstance(nested, list):
                raw_groups.append(nested)

        for group in raw_groups:
            for option in group:
                if not isinstance(option, dict):
                    continue

                label = str(option.get("label", option.get("value", ""))).strip()
                option_id = str(option.get("id", "")).strip()
                if label and option_id:
                    options[label] = option_id

        return options

    def _normalize_options(self, options: list[Any]) -> list[str]:
        result: list[str] = []
        for option in options:
            if isinstance(option, str):
                value = option.strip()
                if value:
                    result.append(value)
            elif isinstance(option, dict):
                value = option.get("value", option.get("label"))
                if value is not None:
                    text = str(value).strip()
                    if text:
                        result.append(text)
        return result

    def _extract_tag_ids(self, doc: dict[str, Any]) -> set[int]:
        result: set[int] = set()
        raw_tags = doc.get("tags", [])
        if isinstance(raw_tags, list):
            for item in raw_tags:
                if isinstance(item, int):
                    result.add(item)
                elif isinstance(item, dict) and isinstance(item.get("id"), int):
                    result.add(int(item["id"]))
        return result

    def _extract_custom_field_map(self, doc: dict[str, Any]) -> dict[int, Any]:
        raw = doc.get("custom_fields")
        out: dict[int, Any] = {}

        if isinstance(raw, dict):
            for key, value in raw.items():
                try:
                    out[int(key)] = value
                except Exception:
                    continue
            return out

        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue

                field: Any = item.get("field")
                if isinstance(field, dict):
                    field = field.get("id")

                if field is None:
                    field = item.get("id") or item.get("field_id") or item.get("custom_field")

                try:
                    field_id = int(field)
                except Exception:
                    continue

                out[field_id] = item.get("value")

        return out

    def _load_vision_image_base64(self, document_id: int) -> str | None:
        try:
            source_bytes, content_type = self.paperless.download_original(document_id)
            if self._is_pdf_file(source_bytes, content_type):
                rendered_pdf = self._render_pdf_page_to_png(source_bytes, document_id)
                if rendered_pdf:
                    return base64.b64encode(rendered_pdf).decode("ascii")
            if self._is_image_content_type(content_type):
                normalized_image = self._normalize_image_bytes_for_ollama(source_bytes, content_type)
                if normalized_image:
                    return base64.b64encode(normalized_image).decode("ascii")
        except Exception:
            pass

        try:
            image_bytes, content_type = self.paperless.download_thumb(document_id)
        except Exception:
            return None

        if not image_bytes:
            return None

        if not self._is_image_content_type(content_type):
            return None

        normalized_image = self._normalize_image_bytes_for_ollama(image_bytes, content_type)
        if not normalized_image:
            return None

        return base64.b64encode(normalized_image).decode("ascii")

    def _normalize_image_bytes_for_ollama(self, image_bytes: bytes, content_type: str) -> bytes | None:
        if not image_bytes:
            return None

        lowered = content_type.lower()
        if "png" in lowered:
            return image_bytes
        if "jpeg" in lowered or "jpg" in lowered:
            return image_bytes

        if Image is None or ImageOps is None:
            return None

        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                normalized: Any = ImageOps.exif_transpose(image)
                if normalized.mode not in ("RGB", "RGBA", "L"):
                    normalized = normalized.convert("RGBA") if "A" in normalized.mode else normalized.convert("RGB")

                output = io.BytesIO()
                normalized.save(output, format="PNG")
                return output.getvalue()
        except Exception:
            return None

    def _render_pdf_page_to_png(self, file_bytes: bytes, document_id: int) -> bytes | None:
        if not file_bytes or fitz is None:
            return None

        try:
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            self._progress(f"[{self.server.name}] Document #{document_id}: failed to render PDF preview ({exc})")
            return None

        try:
            if getattr(pdf, "page_count", 0) < 1:
                return None

            page = pdf.load_page(0)
            matrix = fitz.Matrix(2, 2)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            return pixmap.tobytes("png")
        except Exception as exc:
            self._progress(f"[{self.server.name}] Document #{document_id}: failed to rasterize PDF page ({exc})")
            return None
        finally:
            pdf.close()

    def _is_image_content_type(self, content_type: str) -> bool:
        lowered = content_type.lower()
        return "image" in lowered or lowered.startswith("application/octet-stream")

    def _is_pdf_file(self, file_bytes: bytes, content_type: str) -> bool:
        lowered = content_type.lower()
        return "pdf" in lowered or file_bytes.startswith(b"%PDF")

    def _classify_original_pdf(self, document_id: int) -> str | None:
        try:
            file_bytes, content_type = self.paperless.download_original(document_id)
        except RuntimeError as exc:
            self._progress(f"[{self.server.name}] Document #{document_id}: original download failed, skipping PDF detection ({exc})")
            return None

        if not self._is_pdf_file(file_bytes, content_type):
            return None

        if fitz is None:
            raise RuntimeError(
                "PyMuPDF is required for strict PDF classification. Install `PyMuPDF` to analyze original PDFs."
            )

        try:
            pdf = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            raise RuntimeError(f"Failed to read original PDF for document {document_id}: {exc}") from exc

        try:
            return self._classify_pdf_document(document_id, pdf)
        finally:
            pdf.close()

    def _classify_pdf_document(self, document_id: int, pdf: Any) -> str:
        analyzed_pages = min(getattr(pdf, "page_count", 0), PDF_ANALYSIS_MAX_PAGES)
        if analyzed_pages < 1:
            return "uncertain_pdf"

        counts = {
            "original_digital_page": 0,
            "searchable_scanned_page": 0,
            "image_only_scanned_page": 0,
            "uncertain_page": 0,
        }

        for index in range(analyzed_pages):
            page = pdf.load_page(index)
            page_mode = self._classify_pdf_page(page)
            counts[page_mode] += 1
            self._progress(f"[{self.server.name}] Document #{document_id}: page {index + 1}/{analyzed_pages} -> {page_mode}")

        for page_mode, pdf_mode in (
            ("original_digital_page", "original_digital_pdf"),
            ("searchable_scanned_page", "searchable_scanned_pdf"),
            ("image_only_scanned_page", "image_only_scanned_pdf"),
        ):
            if counts[page_mode] / analyzed_pages >= PDF_DOCUMENT_CLASSIFY_RATIO:
                return pdf_mode

        return "uncertain_pdf"

    def _classify_pdf_page(self, page: Any) -> str:
        text = page.get_text("text") or ""
        text_chars = len(re.sub(r"\s+", "", text))
        substantial_text = text_chars >= PDF_SUBSTANTIAL_TEXT_CHARS
        page_area = max(float(page.rect.width) * float(page.rect.height), 1.0)
        largest_image_area = 0.0

        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", []) if isinstance(page_dict, dict) else []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if int(block.get("type", -1)) != 1:
                continue
            bbox = block.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                x0, y0, x1, y1 = (float(value) for value in bbox)
            except Exception:
                continue
            area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            if area > largest_image_area:
                largest_image_area = area

        image_dominant = (largest_image_area / page_area) >= PDF_IMAGE_DOMINANT_RATIO
        if substantial_text and not image_dominant:
            return "original_digital_page"
        if substantial_text and image_dominant:
            return "searchable_scanned_page"
        if (not substantial_text) and image_dominant:
            return "image_only_scanned_page"
        return "uncertain_page"

    def _paperless_content_for_title(self, doc: dict[str, Any]) -> str:
        from .types import TITLE_SOURCE_MAX_CHARS

        content = self._sanitize_ocr_text(str(doc.get("content", "")))
        if not content:
            raise RuntimeError(
                "Paperless content is empty, so title-only processing cannot be used for this original digital PDF."
            )
        return content[:TITLE_SOURCE_MAX_CHARS].strip()

    def _describe_pdf_mode(self, pdf_mode: str) -> str:
        if pdf_mode == "original_digital_pdf":
            return "Original digital PDF detected from the original file structure. Using Paperless text and skipping OCR overwrite."
        if pdf_mode == "searchable_scanned_pdf":
            return "Searchable scanned PDF detected from the original file structure. Using Ollama OCR and overwriting Paperless content."
        if pdf_mode == "image_only_scanned_pdf":
            return "Image-only scanned PDF detected from the original file structure. Using Ollama OCR and overwriting Paperless content."
        if pdf_mode == "uncertain_pdf":
            return "PDF classification was uncertain. Falling back to Ollama OCR."
        return f"Detected PDF mode: {pdf_mode}"

    def _normalize_status_value(self, raw_value: Any, status_option_ids: dict[str, str]) -> str:
        if raw_value is None:
            return ""

        text = str(raw_value).strip()
        if not text:
            return ""
        if text in STATUS_VALUES:
            return text

        for label, option_id in status_option_ids.items():
            if text == option_id:
                return label

        return text

    def _update_status(
        self,
        document_id: int,
        *,
        custom_field_id: int,
        status_name: str,
        status_option_ids: dict[str, str],
        tags: list[int] | None = None,
    ) -> None:
        value_variants: list[Any] = []
        option_id = status_option_ids.get(status_name)
        if option_id:
            value_variants.append(option_id)
        value_variants.append(status_name)

        try:
            self.paperless.update_custom_field_value(
                document_id,
                field_id=custom_field_id,
                value_variants=value_variants,
                tags=tags,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to set `{STATUS_FIELD_NAME}` to `{status_name}` on document {document_id}: {exc}"
            ) from exc

    def _sanitize_title(self, title: str) -> str:
        cleaned = title.strip().strip("'\"")
        cleaned = re.sub(r"\s+", " ", cleaned)
        if len(cleaned) > TITLE_MAX_LENGTH:
            cleaned = cleaned[:TITLE_MAX_LENGTH].rstrip()
        return cleaned

    def _sanitize_ocr_text(self, text: str) -> str:
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = cleaned.replace("\x00", "")
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _clean_title(self, title: str) -> str:
        return re.sub(r"\s+", " ", title).strip()

    def _document_label(self, doc: dict[str, Any]) -> str:
        document_id = int(doc.get("id", 0) or 0)
        title = self._clean_title(str(doc.get("title", "")))
        if title:
            return f"document #{document_id} ({title})"
        return f"document #{document_id}"
