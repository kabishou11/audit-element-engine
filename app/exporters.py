from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from app.models import FolderSummaryRecord, SingleImageRecord


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, records: Iterable[BaseModel]) -> None:
    payload = [record.model_dump(mode="json") for record in records]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_single_image_csv(path: Path, records: list[SingleImageRecord]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_folder",
                "source_file",
                "attachment_type",
                "attachment_description",
                "classification_basis",
                "selected_audit_elements",
                "issues",
                "content_summary",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_folder": record.source_folder,
                    "source_file": record.source_file,
                    "attachment_type": record.attachment_type,
                    "attachment_description": record.attachment_description,
                    "classification_basis": record.classification_basis,
                    "selected_audit_elements": json.dumps(record.selected_audit_elements, ensure_ascii=False),
                    "issues": json.dumps([item.model_dump() for item in record.issues], ensure_ascii=False),
                    "content_summary": record.visual_analysis.get("content_summary"),
                }
            )


def write_folder_summary_csv(path: Path, records: list[FolderSummaryRecord]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_folder",
                "included_files",
                "attachment_type",
                "attachment_type_distribution",
                "attachment_description",
                "classification_basis",
                "evidence_chain",
                "selected_audit_elements",
                "issues",
                "folder_summary",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_folder": record.source_folder,
                    "included_files": json.dumps(record.included_files, ensure_ascii=False),
                    "attachment_type": record.attachment_type,
                    "attachment_type_distribution": json.dumps(record.attachment_type_distribution, ensure_ascii=False),
                    "attachment_description": record.attachment_description,
                    "classification_basis": record.classification_basis,
                    "evidence_chain": json.dumps(record.evidence_chain, ensure_ascii=False),
                    "selected_audit_elements": json.dumps(record.selected_audit_elements, ensure_ascii=False),
                    "issues": json.dumps([item.model_dump() for item in record.issues], ensure_ascii=False),
                    "folder_summary": record.folder_summary,
                }
            )
