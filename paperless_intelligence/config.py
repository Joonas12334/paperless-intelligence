import re
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def read_env_file() -> dict[str, str]:
    env_path = get_project_root() / ".env"
    env_vars: dict[str, str] = {}

    if not env_path.exists():
        return env_vars

    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip()

    return env_vars


class PaperlessServerConfig(BaseModel):
    name: str
    base_url: str
    api_token: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(get_project_root() / ".env"),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # First server (no number suffix)
    paperless_server_name: str | None = None
    paperless_server_url: str | None = None
    paperless_server_token: str | None = None

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:27b"
    ollama_fallback_model: str = "qwen3.5:latest"
    document_title_language: str = "Estonian"
    ollama_system_prompt: str = (
        "You are an advanced document digitization assistant. Your task is to extract text from images into a clean PLAIN TEXT format. "
        "Strictly follow these rules:\n"
        "1. LAYOUT & SPACING: Preserve the visual layout of the document. Use multiple spaces (tabulation) to separate distinct columns (e.g., Item Name      Quantity      Price). "
        "Do not merge columns into a single line of text; keep them visually distinct.\n"
        "2. Transcribe the document exactly as written. "
        "3. NO CONVERSATION: Output only the extracted text. Do not include markdown backticks (```), 'Here is the text', or JSON formatting."
    )
    ollama_title_prompt_hint: str | None = None

    # Processing options
    max_retries: int = 2
    request_timeout_seconds: int = 240

    @property
    def resolved_document_title_language(self) -> str:
        value = self.document_title_language.strip()
        return value or "Estonian"

    @property
    def resolved_ollama_title_prompt_hint(self) -> str:
        if self.ollama_title_prompt_hint:
            custom_hint = self.ollama_title_prompt_hint.strip()
            if custom_hint:
                return custom_hint

        language = self.resolved_document_title_language
        return (
            f"Based on the document content, generate a concise title in {language}.\n"
            "Format: '[Category or Main Item] - [Vendor]'.\n"
            "Examples: 'Toidulisandid - Ostrovit', 'Kütusearve - Circle K', 'Telefoniarve - Telia'.\n"
            "Rules: \n"
            f"1. Always use {language}, even if the document is in another language.\n"
            "2. Output ONLY the title string. No quotation marks or file extensions."
        )

    @property
    def servers(self) -> list[PaperlessServerConfig]:
        servers: list[PaperlessServerConfig] = []

        # Read .env file directly to get all vars
        env_vars = read_env_file()

        # First server (no number suffix)
        if self.paperless_server_name and self.paperless_server_url and self.paperless_server_token:
            servers.append(PaperlessServerConfig(
                name=self.paperless_server_name,
                base_url=self.paperless_server_url,
                api_token=self.paperless_server_token,
            ))

        # Additional servers (numbered) - find all _N_ suffixes
        suffixes: set[int] = set()

        for key in env_vars:
            match = re.match(r"PAPERLESS_SERVER_(\d+)_NAME", key)
            if match:
                suffixes.add(int(match.group(1)))

        for num in sorted(suffixes):
            name = env_vars.get(f"PAPERLESS_SERVER_{num}_NAME")
            url = env_vars.get(f"PAPERLESS_SERVER_{num}_URL")
            token = env_vars.get(f"PAPERLESS_SERVER_{num}_TOKEN")

            if name and url and token:
                servers.append(PaperlessServerConfig(name=name, base_url=url, api_token=token))

        # Return default if no servers configured
        if not servers:
            servers.append(PaperlessServerConfig(
                name="default",
                base_url="http://localhost:8000",
                api_token="replace-me"
            ))

        return servers


def get_settings() -> Settings:
    return Settings()
