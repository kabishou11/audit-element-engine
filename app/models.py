from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AttachmentRule(BaseModel):
    attachment_type: str
    attachment_description: str
    ai_audit_elements: list[str]
    audit_issue_references: list[str]
    aliases: list[str]


class VisualAnalysisResult(BaseModel):
    document_category: str | None = None
    specific_type: str | None = None
    key_entities: dict[str, Any] = Field(default_factory=dict)
    signatures_and_seals: Any = Field(default_factory=dict)
    content_summary: str | None = None
    visual_elements: Any = Field(default_factory=dict)
    raw_json: dict[str, Any] = Field(default_factory=dict)


class IssueItem(BaseModel):
    rule_issue_reference: str | None = None
    problem: str
    suggestion: str
    basis: str | None = None


class SingleImageRecord(BaseModel):
    source_folder: str
    source_file: str
    attachment_type: str
    attachment_description: str
    classification_basis: str
    selected_audit_elements: dict[str, Any]
    issues: list[IssueItem]
    visual_analysis: dict[str, Any]
    raw_outputs: dict[str, Any] = Field(default_factory=dict)


class FolderSummaryRecord(BaseModel):
    source_folder: str
    included_files: list[str]
    attachment_type: str
    attachment_type_distribution: dict[str, int] = Field(default_factory=dict)
    attachment_description: str
    classification_basis: str
    evidence_chain: list[dict[str, Any]] = Field(default_factory=list)
    selected_audit_elements: dict[str, Any]
    issues: list[IssueItem]
    folder_summary: str
    referenced_single_image_files: list[str]
    raw_outputs: dict[str, Any] = Field(default_factory=dict)


class TaskOutput(BaseModel):
    task_id: str
    created_at: datetime
    single_image_records: list[SingleImageRecord]
    folder_summary_records: list[FolderSummaryRecord]
    output_dir: str


class FolderProcessRequest(BaseModel):
    folder_name: str
    overrides: dict[str, Any] | None = None


class ProcessAllRequest(BaseModel):
    folder_names: list[str] | None = None
    overrides: dict[str, Any] | None = None


class TaskFileRecord(BaseModel):
    source_folder: str
    source_file: str
    status: str = "pending"
    error_message: str | None = None


class TaskFolderRecord(BaseModel):
    source_folder: str
    status: str = "pending"
    image_files: list[str] = Field(default_factory=list)
    folder_summary_done: bool = False
    error_message: str | None = None


class TaskState(BaseModel):
    task_id: str
    scope: str
    status: str
    created_at: datetime
    updated_at: datetime
    output_dir: str
    folder_names: list[str]
    total_images: int = 0
    completed_images: int = 0
    failed_images: int = 0
    total_folders: int = 0
    completed_folders: int = 0
    failed_folders: int = 0
    current_folder: str | None = None
    current_file: str | None = None
    error_message: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
    file_records: list[TaskFileRecord] = Field(default_factory=list)
    folder_records: list[TaskFolderRecord] = Field(default_factory=list)


class TaskStartResponse(BaseModel):
    task_id: str
    status: str
    output_dir: str


class TaskCancelResponse(BaseModel):
    task_id: str
    status: str


class TaskListItem(BaseModel):
    task_id: str
    status: str
    scope: str
    created_at: datetime
    updated_at: datetime
    total_images: int = 0
    completed_images: int = 0
    failed_images: int = 0
    total_folders: int = 0
    completed_folders: int = 0
    failed_folders: int = 0
    current_folder: str | None = None
    current_file: str | None = None
    output_dir: str
