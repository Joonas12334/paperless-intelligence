import base64
import json
import re
from typing import Any

import httpx

from .config import PaperlessServerConfig
from .types import TITLE_SOURCE_MAX_CHARS


class PaperlessApi:
    def __init__(self, server: PaperlessServerConfig, timeout: int) -> None:
        self.server = server
        self.timeout = timeout
        self.base_url = server.base_url.rstrip("/")
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Token {server.api_token}",
                "Accept": "application/json",
            },
        )

    def _request_json(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | list[Any] | None = None,
    ) -> dict[str, Any]:
        url = self._build_url(path, params)
        try:
            response = self.client.request(method, url, json=json_data)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text or str(exc)
            raise RuntimeError(f"HTTP {exc.response.status_code} for {method} {path}: {detail}") from exc
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Paperless did not respond within {self.timeout} seconds at {self.base_url}. "
                "Increase `request_timeout_seconds`."
            )
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Could not connect to Paperless at {self.base_url}. "
                "Make sure the service is running and the URL is correct."
            ) from exc

        if not response.text.strip():
            return {}

        try:
            value = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON response for {method} {path}: {exc}") from exc

        if isinstance(value, dict):
            return value
        raise RuntimeError(f"Unexpected non-object JSON response for {method} {path}")

    def _request_bytes(self, method: str, path: str, params: dict[str, Any] | None = None) -> tuple[bytes, str]:
        url = self._build_url(path, params)
        try:
            response = self.client.request(method, url)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text or str(exc)
            raise RuntimeError(f"HTTP {exc.response.status_code} for {method} {path}: {detail}") from exc
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Paperless did not respond within {self.timeout} seconds at {self.base_url}."
            )
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Could not connect to Paperless at {self.base_url}. "
                "Make sure the service is running."
            ) from exc

        content_type = response.headers.get("content-type", "application/octet-stream")
        return response.content, content_type

    def _build_url(self, path: str, params: dict[str, Any] | None = None) -> str:
        if not path.startswith("/"):
            path = f"/{path}"

        url = f"{self.base_url}{path}"
        if not params:
            return url

        flat_params: dict[str, str] = {}
        for key, value in params.items():
            if value is not None:
                flat_params[key] = str(value)

        if not flat_params:
            return url

        return f"{url}?{httpx.QueryParams(flat_params)}"

    def iter_documents(self, *, page_size: int = 100, ordering: str = "-added"):
        page = 1
        while True:
            payload = self._request_json(
                "GET",
                "/api/documents/",
                params={"page": page, "page_size": page_size, "ordering": ordering},
            )
            results = payload.get("results", [])
            if isinstance(results, list):
                for doc in results:
                    if isinstance(doc, dict):
                        yield doc

            if not payload.get("next"):
                return
            page += 1

    def iter_tags(self, *, page_size: int = 200):
        page = 1
        while True:
            payload = self._request_json("GET", "/api/tags/", params={"page": page, "page_size": page_size})
            for tag in payload.get("results", []):
                if isinstance(tag, dict):
                    yield tag
            if not payload.get("next"):
                return
            page += 1

    def iter_custom_fields(self, *, page_size: int = 200):
        page = 1
        while True:
            payload = self._request_json(
                "GET",
                "/api/custom_fields/",
                params={"page": page, "page_size": page_size},
            )
            for field in payload.get("results", []):
                if isinstance(field, dict):
                    yield field
            if not payload.get("next"):
                return
            page += 1

    def get_document(self, document_id: int) -> dict[str, Any]:
        return self._request_json("GET", f"/api/documents/{document_id}/")

    def update_title(self, document_id: int, title: str, *, tags: list[int] | None = None) -> None:
        payload: dict[str, Any] = {"title": title}
        if tags is not None:
            payload["tags"] = tags
        self._request_json("PATCH", f"/api/documents/{document_id}/", json_data=payload)

    def update_content(self, document_id: int, content: str, *, tags: list[int] | None = None) -> None:
        payload: dict[str, Any] = {"content": content}
        if tags is not None:
            payload["tags"] = tags
        self._request_json("PATCH", f"/api/documents/{document_id}/", json_data=payload)

    def update_document_metadata(
        self,
        document_id: int,
        *,
        title: str | None = None,
        tags: list[int] | None = None,
        custom_fields: dict[int, Any] | None = None,
    ) -> None:
        base_payload: dict[str, Any] = {}
        if title is not None:
            base_payload["title"] = title
        if tags is not None:
            base_payload["tags"] = tags

        if custom_fields is None:
            self._request_json("PATCH", f"/api/documents/{document_id}/", json_data=base_payload)
            return

        variants = [
            {**base_payload, "custom_fields": {str(k): v for k, v in custom_fields.items()}},
            {
                **base_payload,
                "custom_fields": [{"field": int(k), "value": v} for k, v in custom_fields.items()],
            },
            {
                **base_payload,
                "custom_fields": [{"id": int(k), "value": v} for k, v in custom_fields.items()],
            },
        ]

        last_error: Exception | None = None
        for payload in variants:
            try:
                self._request_json("PATCH", f"/api/documents/{document_id}/", json_data=payload)
                return
            except Exception as exc:
                last_error = exc

        if last_error:
            raise last_error
        raise RuntimeError("Failed to update document metadata")

    def update_custom_field_value(
        self,
        document_id: int,
        *,
        field_id: int,
        value_variants: list[Any],
        tags: list[int] | None = None,
    ) -> None:
        last_error: Exception | None = None
        tags_part: dict[str, Any] = {"tags": tags} if tags is not None else {}

        for value in value_variants:
            payload_variants = [
                {**tags_part, "custom_fields": {str(field_id): value}},
                {**tags_part, "custom_fields": [{"field": field_id, "value": value}]},
                {**tags_part, "custom_fields": [{"id": field_id, "value": value}]},
            ]

            for payload in payload_variants:
                try:
                    self._request_json("PATCH", f"/api/documents/{document_id}/", json_data=payload)
                    return
                except Exception as exc:
                    last_error = exc

        if last_error:
            raise last_error
        raise RuntimeError("Failed to update document custom field")

    def download_thumb(self, document_id: int) -> tuple[bytes, str]:
        return self._request_bytes("GET", f"/api/documents/{document_id}/thumb/")

    def download_original(self, document_id: int) -> tuple[bytes, str]:
        return self._request_bytes("GET", f"/api/documents/{document_id}/download/?original=true")


class OllamaApi:
    def __init__(self, base_url: str, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

    def _request_json(self, method: str, path: str, json_data: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = self.client.request(method, path, json=json_data)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text or str(exc)
            raise RuntimeError(f"HTTP {exc.response.status_code} for {method} {path}: {detail}") from exc
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Ollama did not respond within {self.timeout} seconds at {self.base_url}. "
                "Increase `request_timeout_seconds`, use a faster/smaller model, or retry with a simpler document."
            )
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running and the URL is correct."
            ) from exc

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON response for {method} {path}: {exc}") from exc

    def generate_title(
        self,
        *,
        model: str,
        image_base64: str,
        context: str,
        system_prompt: str,
        title_hint: str,
    ) -> dict[str, Any]:
        user_prompt = (
            "Analyze the document image and output strict JSON with keys: "
            "title, language, confidence, ocr_text, reasoning_short. "
            "Title must be concise and in the document's native language. "
            "Make the title specific enough to identify the document later. "
            "Prefer a concrete title that combines document type with the merchant, brand, issuer, "
            "or subject when it is visible in the document. "
            "Avoid generic titles like only 'Kviitung' or only 'Arve' when the source is clear. "
            "ocr_text must contain the best-effort full OCR transcription as plain text. "
            "Do not use markdown. Preserve line breaks when they help readability. "
            "Prefer readable text over perfect formatting. "
            f"{title_hint} "
            f"Extra context: {context}"
        )

        payload = {
            "model": model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": [image_base64]},
            ],
        }

        response = self._request_json("POST", "/api/chat", json_data=payload)
        message = response.get("message", {})
        content = str(message.get("content", "")).strip()
        return self._extract_json(content)

    def generate_title_from_text(
        self,
        *,
        model: str,
        text: str,
        context: str,
        title_hint: str,
    ) -> dict[str, Any]:
        clipped_text = text[:TITLE_SOURCE_MAX_CHARS].strip()
        payload = {
            "model": model,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You generate concise document titles from extracted document text. "
                        "Return strict JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Analyze the extracted document text and output strict JSON with keys: "
                        "title, language, confidence, reasoning_short. "
                        "Title must be concise and in the document's native language. "
                        "Make the title specific enough to identify the document later. "
                        "Prefer a concrete title that combines document type with the merchant, brand, issuer, "
                        "or subject when it is visible in the text. "
                        "Avoid generic titles like only 'Kviitung' or only 'Arve' when the source is clear. "
                        "Do not use markdown. "
                        f"{title_hint} "
                        f"Extra context: {context}\n\n"
                        "Document text:\n"
                        f"{clipped_text}"
                    ),
                },
            ],
        }

        response = self._request_json("POST", "/api/chat", json_data=payload)
        message = response.get("message", {})
        content = str(message.get("content", "")).strip()
        return self._extract_json(content)

    def _extract_json(self, text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise RuntimeError("Model did not return JSON")

        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise RuntimeError("Model response JSON is not an object")
        return parsed
