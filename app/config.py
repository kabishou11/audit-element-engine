from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelEndpointConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    reasoning_split: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    vision_api_key: str = Field(alias="VISION_API_KEY")
    vision_base_url: str = Field(default="https://api-inference.modelscope.cn/v1", alias="VISION_BASE_URL")
    vision_model: str = Field(default="Qwen/Qwen3.5-35B-A3B", alias="VISION_MODEL")
    vision_temperature: float = Field(default=0.0, alias="VISION_TEMPERATURE")
    vision_max_tokens: int = Field(default=4096, alias="VISION_MAX_TOKENS")
    vision_timeout_seconds: int = Field(default=180, alias="VISION_TIMEOUT_SECONDS")

    minimax_api_key: str = Field(default="", alias="MINIMAX_API_KEY")
    minimax_base_url: str = Field(default="https://api.minimaxi.com/v1", alias="MINIMAX_BASE_URL")
    minimax_model: str = Field(default="MiniMax-M2.7", alias="MINIMAX_MODEL")
    minimax_temperature: float = Field(default=0.1, alias="MINIMAX_TEMPERATURE")
    minimax_max_tokens: int = Field(default=4096, alias="MINIMAX_MAX_TOKENS")
    minimax_timeout_seconds: int = Field(default=120, alias="MINIMAX_TIMEOUT_SECONDS")
    minimax_reasoning_split: bool = Field(default=False, alias="MINIMAX_REASONING_SPLIT")

    classifier_model_source: str = Field(default="minimax", alias="CLASSIFIER_MODEL_SOURCE")
    classifier_model_name: str = Field(default="", alias="CLASSIFIER_MODEL_NAME")
    classifier_temperature: float = Field(default=0.0, alias="CLASSIFIER_TEMPERATURE")
    classifier_max_tokens: int = Field(default=2048, alias="CLASSIFIER_MAX_TOKENS")
    classifier_timeout_seconds: int = Field(default=120, alias="CLASSIFIER_TIMEOUT_SECONDS")

    extractor_model_source: str = Field(default="minimax", alias="EXTRACTOR_MODEL_SOURCE")
    extractor_model_name: str = Field(default="", alias="EXTRACTOR_MODEL_NAME")
    extractor_temperature: float = Field(default=0.0, alias="EXTRACTOR_TEMPERATURE")
    extractor_max_tokens: int = Field(default=2048, alias="EXTRACTOR_MAX_TOKENS")
    extractor_timeout_seconds: int = Field(default=120, alias="EXTRACTOR_TIMEOUT_SECONDS")

    audit_model_source: str = Field(default="minimax", alias="AUDIT_MODEL_SOURCE")
    audit_model_name: str = Field(default="", alias="AUDIT_MODEL_NAME")
    audit_temperature: float = Field(default=0.0, alias="AUDIT_TEMPERATURE")
    audit_max_tokens: int = Field(default=2048, alias="AUDIT_MAX_TOKENS")
    audit_timeout_seconds: int = Field(default=120, alias="AUDIT_TIMEOUT_SECONDS")

    folder_model_source: str = Field(default="minimax", alias="FOLDER_MODEL_SOURCE")
    folder_model_name: str = Field(default="", alias="FOLDER_MODEL_NAME")
    folder_temperature: float = Field(default=0.0, alias="FOLDER_TEMPERATURE")
    folder_max_tokens: int = Field(default=3072, alias="FOLDER_MAX_TOKENS")
    folder_timeout_seconds: int = Field(default=180, alias="FOLDER_TIMEOUT_SECONDS")

    rules_excel_path: Path = Field(alias="RULES_EXCEL_PATH")
    classification_prompt_path: Path = Field(alias="CLASSIFICATION_PROMPT_PATH")
    visual_analysis_prompt_path: Path = Field(alias="VISUAL_ANALYSIS_PROMPT_PATH")
    attachment_root: Path = Field(alias="ATTACHMENT_ROOT")
    output_root: Path = Field(default=Path("outputs"), alias="OUTPUT_ROOT")
    rule_recall_top_k: int = Field(default=5, alias="RULE_RECALL_TOP_K")
    rule_recall_min_score: float = Field(default=0.05, alias="RULE_RECALL_MIN_SCORE")
    rule_alias_bonus: float = Field(default=3.0, alias="RULE_ALIAS_BONUS")
    rule_fragment_bonus: float = Field(default=0.4, alias="RULE_FRAGMENT_BONUS")
    rule_element_bonus: float = Field(default=1.2, alias="RULE_ELEMENT_BONUS")
    rule_bigram_weight: float = Field(default=2.0, alias="RULE_BIGRAM_WEIGHT")
    allowed_image_extensions: str = Field(
        default=".jpg,.jpeg,.png,.bmp,.webp,.tif,.tiff",
        alias="ALLOWED_IMAGE_EXTENSIONS",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    task_log_enabled: bool = Field(default=True, alias="TASK_LOG_ENABLED")
    model_max_retries: int = Field(default=2, alias="MODEL_MAX_RETRIES")
    model_retry_backoff_seconds: float = Field(default=2.0, alias="MODEL_RETRY_BACKOFF_SECONDS")
    single_image_timeout_seconds: int = Field(default=600, alias="SINGLE_IMAGE_TIMEOUT_SECONDS")

    @property
    def image_extensions(self) -> set[str]:
        return {item.strip().lower() for item in self.allowed_image_extensions.split(",") if item.strip()}

    def model_post_init(self, __context: object) -> None:
        self.rules_excel_path = self._resolve_path(self.rules_excel_path)
        self.classification_prompt_path = self._resolve_path(self.classification_prompt_path)
        self.visual_analysis_prompt_path = self._resolve_path(self.visual_analysis_prompt_path)
        self.attachment_root = self._resolve_path(self.attachment_root)
        self.output_root = self._resolve_path(self.output_root)

    @staticmethod
    def _resolve_path(value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @property
    def vision_config(self) -> ModelEndpointConfig:
        return ModelEndpointConfig(
            api_key=self.vision_api_key,
            base_url=self.vision_base_url,
            model=self.vision_model,
            temperature=self.vision_temperature,
            max_tokens=self.vision_max_tokens,
            timeout_seconds=self.vision_timeout_seconds,
            reasoning_split=False,
        )

    @property
    def minimax_config(self) -> ModelEndpointConfig:
        return ModelEndpointConfig(
            api_key=self.minimax_api_key or self.vision_api_key,
            base_url=self.minimax_base_url if self.minimax_api_key else self.vision_base_url,
            model=self.minimax_model if self.minimax_api_key else self.vision_model,
            temperature=self.minimax_temperature if self.minimax_api_key else 0.0,
            max_tokens=self.minimax_max_tokens if self.minimax_api_key else self.vision_max_tokens,
            timeout_seconds=self.minimax_timeout_seconds if self.minimax_api_key else self.vision_timeout_seconds,
            reasoning_split=self.minimax_reasoning_split if self.minimax_api_key else False,
        )

    def get_stage_text_config(self, stage_name: str) -> ModelEndpointConfig:
        source = getattr(self, f"{stage_name}_model_source")
        override_model_name = getattr(self, f"{stage_name}_model_name")
        temperature = getattr(self, f"{stage_name}_temperature")
        max_tokens = getattr(self, f"{stage_name}_max_tokens")
        timeout_seconds = getattr(self, f"{stage_name}_timeout_seconds")

        provider_config = self.minimax_config if source.lower() == "minimax" else self.vision_config
        return ModelEndpointConfig(
            api_key=provider_config.api_key,
            base_url=provider_config.base_url,
            model=override_model_name or provider_config.model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            reasoning_split=provider_config.reasoning_split,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def build_runtime_settings(overrides: dict[str, Any] | None = None) -> Settings:
    base = get_settings()
    if not overrides:
        return base
    merged = base.model_dump()
    for key, value in overrides.items():
        if key not in merged or value in (None, ""):
            continue
        merged[key] = value
    return Settings.model_validate(merged)
