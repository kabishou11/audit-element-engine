from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.models import (
    FolderProcessRequest,
    ProcessAllRequest,
    TaskCancelResponse,
    TaskListItem,
    TaskOutput,
    TaskStartResponse,
    TaskState,
)
from app.pipeline import VoucherAttachmentPipeline
from app.task_manager import TaskManager

app = FastAPI(title="凭证附件识别审计服务", version="0.1.0")
_pipeline_instance: VoucherAttachmentPipeline | None = None
_task_manager_instance: TaskManager | None = None


def _get_pipeline() -> VoucherAttachmentPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        settings = get_settings()
        _pipeline_instance = VoucherAttachmentPipeline(settings)
    return _pipeline_instance


def _get_task_manager() -> TaskManager:
    global _task_manager_instance
    if _task_manager_instance is None:
        _task_manager_instance = TaskManager(_get_pipeline())
    return _task_manager_instance


def _resolve_folder(folder_name: str) -> Path:
    settings = get_settings()
    folder_path = settings.attachment_root / folder_name
    if not folder_path.exists() or not folder_path.is_dir():
        raise HTTPException(status_code=404, detail=f"未找到凭证文件夹: {folder_name}")
    return folder_path


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/folders")
def list_folders() -> list[dict[str, int | str]]:
    settings = get_settings()
    items: list[dict[str, int | str]] = []
    for folder_path in sorted(path for path in settings.attachment_root.iterdir() if path.is_dir()):
        image_count = len([path for path in folder_path.iterdir() if path.is_file() and path.suffix.lower() in settings.image_extensions])
        items.append({"folder_name": folder_path.name, "image_count": image_count})
    return items


@app.post("/api/v1/process/folder", response_model=TaskStartResponse)
def process_folder(request: FolderProcessRequest) -> TaskStartResponse:
    folder_path = _resolve_folder(request.folder_name)
    task_manager = _get_task_manager()
    state = task_manager.create_new_task(scope="folder", folder_paths=[folder_path], overrides=request.overrides)
    return task_manager.start_task(state, [folder_path])


@app.post("/api/v1/process/all", response_model=TaskStartResponse)
def process_all(request: ProcessAllRequest | None = None) -> TaskStartResponse:
    settings = get_settings()
    task_manager = _get_task_manager()
    if request and request.folder_names:
        folder_paths = [_resolve_folder(folder_name) for folder_name in request.folder_names]
    else:
        folder_paths = sorted(path for path in settings.attachment_root.iterdir() if path.is_dir())
    state = task_manager.create_new_task(
        scope="all",
        folder_paths=folder_paths,
        overrides=request.overrides if request else None,
    )
    return task_manager.start_task(state, folder_paths)


@app.get("/api/v1/tasks/{task_id}", response_model=TaskState)
def get_task(task_id: str) -> TaskState:
    task_manager = _get_task_manager()
    try:
        return task_manager.load_state(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务: {task_id}") from exc


@app.get("/api/v1/tasks", response_model=list[TaskListItem])
def list_tasks(limit: int = 20) -> list[TaskListItem]:
    task_manager = _get_task_manager()
    return task_manager.list_tasks(limit=limit)


@app.get("/api/v1/tasks/{task_id}/result", response_model=TaskOutput)
def get_task_result(task_id: str) -> TaskOutput:
    task_manager = _get_task_manager()
    try:
        return task_manager.load_task_output(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务: {task_id}") from exc


@app.post("/api/v1/tasks/{task_id}/resume", response_model=TaskStartResponse)
def resume_task(task_id: str) -> TaskStartResponse:
    task_manager = _get_task_manager()
    try:
        state = task_manager.load_state(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务: {task_id}") from exc

    folder_paths = [_resolve_folder(folder_name) for folder_name in state.folder_names]
    return task_manager.resume_task(task_id, folder_paths)


@app.post("/api/v1/tasks/{task_id}/cancel", response_model=TaskCancelResponse)
def cancel_task(task_id: str) -> TaskCancelResponse:
    task_manager = _get_task_manager()
    try:
        state = task_manager.cancel_task(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"未找到任务: {task_id}")
    return TaskCancelResponse(task_id=state.task_id, status=state.status)
