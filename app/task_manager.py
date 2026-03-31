from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app.config import build_runtime_settings
from app.exporters import ensure_output_dir, write_folder_summary_csv, write_json, write_single_image_csv
from app.models import (
    FolderSummaryRecord,
    SingleImageRecord,
    TaskFileRecord,
    TaskFolderRecord,
    TaskListItem,
    TaskOutput,
    TaskStartResponse,
    TaskState,
)
from app.pipeline import VoucherAttachmentPipeline
from app.task_logging import TaskLogger


class TaskManager:
    def __init__(self, pipeline: VoucherAttachmentPipeline) -> None:
        self.pipeline = pipeline
        self._threads: dict[str, threading.Thread] = {}

    def create_task(self, task_id: str, scope: str, folder_paths: list[Path], overrides: dict | None = None) -> TaskState:
        output_dir = self.pipeline.settings.output_root / task_id
        ensure_output_dir(output_dir)
        now = datetime.now()
        file_records: list[TaskFileRecord] = []
        folder_records: list[TaskFolderRecord] = []
        for folder_path in folder_paths:
            image_paths = self.pipeline._list_image_paths(folder_path)
            folder_records.append(
                TaskFolderRecord(
                    source_folder=folder_path.name,
                    image_files=[path.name for path in image_paths],
                )
            )
            for image_path in image_paths:
                file_records.append(TaskFileRecord(source_folder=folder_path.name, source_file=image_path.name))
        state = TaskState(
            task_id=task_id,
            scope=scope,
            status="pending",
            created_at=now,
            updated_at=now,
            output_dir=str(output_dir.resolve()),
            folder_names=[path.name for path in folder_paths],
            overrides=overrides or {},
            total_images=len(file_records),
            total_folders=len(folder_records),
            file_records=file_records,
            folder_records=folder_records,
        )
        self.save_state(state)
        self._write_records(output_dir, [], [])
        return state

    def create_new_task(
        self,
        scope: str,
        folder_paths: list[Path],
        overrides: dict | None = None,
        task_label: str | None = None,
    ) -> TaskState:
        label = task_label or ("all" if scope == "all" or len(folder_paths) > 1 else folder_paths[0].name)
        task_id = self.pipeline._build_task_id(label)
        return self.create_task(task_id=task_id, scope=scope, folder_paths=folder_paths, overrides=overrides)

    def start_task(self, state: TaskState, folder_paths: list[Path]) -> TaskStartResponse:
        thread = threading.Thread(target=self._run_task, args=(state.task_id, folder_paths), daemon=True)
        thread.start()
        self._threads[state.task_id] = thread
        return TaskStartResponse(task_id=state.task_id, status=state.status, output_dir=state.output_dir)

    def resume_task(self, task_id: str, folder_paths: list[Path]) -> TaskStartResponse:
        state = self.prepare_resume(task_id, keep_running=True)
        return self.start_task(state, folder_paths)

    def cancel_task(self, task_id: str) -> TaskState:
        state = self.load_state(task_id)
        if state.status in {"completed", "failed", "cancelled"}:
            return state
        state.status = "cancel_requested"
        state.updated_at = datetime.now()
        self.save_state(state)
        return state

    def run_task_sync(
        self,
        task_id: str,
        folder_paths: list[Path],
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> TaskState:
        self._execute_task(task_id, folder_paths, progress_callback=progress_callback)
        return self.load_state(task_id)

    def prepare_resume(self, task_id: str, keep_running: bool = False) -> TaskState:
        state = self.load_state(task_id)
        if state.status == "running" and keep_running:
            return state
        state.status = "pending"
        state.error_message = None
        state.updated_at = datetime.now()
        self.save_state(state)
        return state

    def load_state(self, task_id: str) -> TaskState:
        state_path = self._state_path(task_id)
        return TaskState.model_validate_json(state_path.read_text(encoding="utf-8"))

    def load_task_output(self, task_id: str) -> TaskOutput:
        state = self.load_state(task_id)
        output_dir = Path(state.output_dir)
        single_records = self._load_record_list(output_dir / "single_images.json", SingleImageRecord)
        folder_records = self._load_record_list(output_dir / "folder_summary.json", FolderSummaryRecord)
        return TaskOutput(
            task_id=state.task_id,
            created_at=state.created_at,
            single_image_records=single_records,
            folder_summary_records=folder_records,
            output_dir=state.output_dir,
        )

    def list_tasks(self, limit: int = 20) -> list[TaskListItem]:
        output_root = self.pipeline.settings.output_root
        items: list[TaskListItem] = []
        for state_path in output_root.glob("*/task_state.json"):
            try:
                state = TaskState.model_validate_json(state_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            items.append(
                TaskListItem(
                    task_id=state.task_id,
                    status=state.status,
                    scope=state.scope,
                    created_at=state.created_at,
                    updated_at=state.updated_at,
                    total_images=state.total_images,
                    completed_images=state.completed_images,
                    failed_images=state.failed_images,
                    total_folders=state.total_folders,
                    completed_folders=state.completed_folders,
                    failed_folders=state.failed_folders,
                    current_folder=state.current_folder,
                    current_file=state.current_file,
                    output_dir=state.output_dir,
                )
            )
        items.sort(key=lambda item: item.updated_at, reverse=True)
        return items[:limit]

    def save_state(self, state: TaskState) -> None:
        state.updated_at = datetime.now()
        self._write_json_atomic(self._state_path(state.task_id), state.model_dump(mode="json"))

    def _run_task(self, task_id: str, folder_paths: list[Path]) -> None:
        self._execute_task(task_id, folder_paths, progress_callback=None)

    def _execute_task(
        self,
        task_id: str,
        folder_paths: list[Path],
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        state = self.load_state(task_id)
        runtime_pipeline = VoucherAttachmentPipeline(build_runtime_settings(state.overrides))
        output_dir = Path(state.output_dir)
        task_logger = TaskLogger(
            output_dir / "task.log",
            enabled=runtime_pipeline.settings.task_log_enabled,
        )
        single_records = self._load_record_list(output_dir / "single_images.json", SingleImageRecord)
        folder_summary_records = self._load_record_list(output_dir / "folder_summary.json", FolderSummaryRecord)
        folder_map = {record.source_folder: record for record in folder_summary_records}
        resume_failed_only = state.failed_images > 0
        folders_requiring_summary: set[str] = set()
        try:
            state.status = "running"
            task_logger.log("task_started", task_id=task_id, folder_count=len(folder_paths))
            self.save_state(state)
            self._emit_progress(progress_callback, "task_started", state=state, output_dir=str(output_dir))
            self._emit_progress(progress_callback, "progress_updated", state=state)
            for folder_path in folder_paths:
                state = self.load_state(task_id)
                if state.status == "cancel_requested":
                    state.status = "cancelled"
                    state.current_folder = None
                    state.current_file = None
                    task_logger.log("task_cancelled", task_id=task_id)
                    self.save_state(state)
                    self._emit_progress(progress_callback, "task_cancelled", state=state)
                    return

                state.current_folder = folder_path.name
                folder_record = self._get_folder_record(state, folder_path.name)
                should_process_folder = self._should_process_folder(state, folder_record, resume_failed_only)
                if not should_process_folder:
                    task_logger.log("folder_skipped_completed", folder=folder_path.name)
                    continue

                previous_folder_status = folder_record.status
                folders_requiring_summary.add(folder_path.name)
                if previous_folder_status != "completed":
                    folder_record.status = "running"
                folder_record.error_message = None
                self._refresh_progress_counts(state)
                task_logger.log("folder_started", folder=folder_path.name)
                self.save_state(state)
                self._emit_progress(progress_callback, "folder_started", state=state, folder=folder_path.name)

                for image_path in runtime_pipeline._list_image_paths(folder_path):
                    state = self.load_state(task_id)
                    folder_record = self._get_folder_record(state, folder_path.name)
                    file_record = self._get_file_record(state, folder_path.name, image_path.name)
                    if file_record.status == "completed":
                        task_logger.log("image_skipped_completed", folder=folder_path.name, file=image_path.name)
                        self._emit_progress(
                            progress_callback,
                            "image_skipped_completed",
                            state=state,
                            folder=folder_path.name,
                            file=image_path.name,
                        )
                        continue
                    if state.status == "cancel_requested":
                        state.status = "cancelled"
                        state.current_file = None
                        state.current_folder = None
                        folder_record.status = "cancelled"
                        self._refresh_progress_counts(state)
                        task_logger.log("task_cancelled", task_id=task_id, folder=folder_path.name, file=image_path.name)
                        self.save_state(state)
                        self._emit_progress(progress_callback, "task_cancelled", state=state)
                        return

                    state.current_file = image_path.name
                    task_logger.log("image_started", folder=folder_path.name, file=image_path.name)
                    self.save_state(state)
                    self._emit_progress(progress_callback, "image_started", state=state, folder=folder_path.name, file=image_path.name)
                    executor: ThreadPoolExecutor | None = None
                    try:
                        executor = ThreadPoolExecutor(max_workers=1)
                        future = executor.submit(
                            runtime_pipeline.process_single_image,
                            folder_path,
                            image_path,
                            task_logger.log,
                        )
                        record = future.result(timeout=runtime_pipeline.settings.single_image_timeout_seconds)
                        single_records = [item for item in single_records if not (item.source_folder == record.source_folder and item.source_file == record.source_file)]
                        single_records.append(record)
                        single_records.sort(key=lambda item: (item.source_folder, item.source_file))
                        file_record.status = "completed"
                        file_record.error_message = None
                        folders_requiring_summary.add(folder_path.name)
                        self._refresh_progress_counts(state)
                        self._write_records(output_dir, single_records, list(folder_map.values()))
                        task_logger.log("image_completed", folder=folder_path.name, file=image_path.name)
                        self.save_state(state)
                        self._emit_progress(
                            progress_callback,
                            "image_completed",
                            state=state,
                            folder=folder_path.name,
                            file=image_path.name,
                        )
                        executor.shutdown(wait=False, cancel_futures=False)
                    except FutureTimeoutError:
                        if executor is not None:
                            executor.shutdown(wait=False, cancel_futures=True)
                        file_record.status = "failed"
                        file_record.error_message = (
                            f"单图处理超时，超过 {runtime_pipeline.settings.single_image_timeout_seconds} 秒"
                        )
                        self._refresh_progress_counts(state)
                        task_logger.log(
                            "image_timeout",
                            folder=folder_path.name,
                            file=image_path.name,
                            timeout_seconds=runtime_pipeline.settings.single_image_timeout_seconds,
                        )
                        self.save_state(state)
                        self._emit_progress(
                            progress_callback,
                            "image_timeout",
                            state=state,
                            folder=folder_path.name,
                            file=image_path.name,
                            error=file_record.error_message,
                        )
                    except Exception as exc:  # noqa: BLE001
                        if executor is not None:
                            executor.shutdown(wait=False, cancel_futures=True)
                        file_record.status = "failed"
                        file_record.error_message = str(exc)
                        self._refresh_progress_counts(state)
                        task_logger.log("image_failed", folder=folder_path.name, file=image_path.name, error=str(exc))
                        self.save_state(state)
                        self._emit_progress(
                            progress_callback,
                            "image_failed",
                            state=state,
                            folder=folder_path.name,
                            file=image_path.name,
                            error=str(exc),
                        )

                state = self.load_state(task_id)
                folder_record = self._get_folder_record(state, folder_path.name)
                current_single_records = [record for record in single_records if record.source_folder == folder_path.name]
                should_refresh_summary = (
                    folder_path.name in folders_requiring_summary
                    or not folder_record.folder_summary_done
                    or previous_folder_status != "completed"
                )
                if current_single_records and should_refresh_summary:
                    try:
                        folder_summary = runtime_pipeline.process_folder_summary(folder_path, current_single_records, task_logger.log)
                        folder_map[folder_path.name] = folder_summary
                        folder_record.folder_summary_done = True
                        folder_record.status = "completed"
                        folder_record.error_message = None
                        task_logger.log("folder_completed", folder=folder_path.name)
                        self._emit_progress(progress_callback, "folder_completed", state=state, folder=folder_path.name)
                    except Exception as exc:  # noqa: BLE001
                        folder_record.status = "failed"
                        folder_record.error_message = str(exc)
                        task_logger.log("folder_failed", folder=folder_path.name, error=str(exc))
                        self._emit_progress(
                            progress_callback,
                            "folder_failed",
                            state=state,
                            folder=folder_path.name,
                            error=str(exc),
                        )
                elif current_single_records:
                    folder_record.status = previous_folder_status
                    folder_record.error_message = None
                else:
                    folder_record.status = "failed"
                    folder_record.error_message = "未成功处理任何图片，无法生成整凭证汇总。"
                    task_logger.log("folder_failed", folder=folder_path.name, error=folder_record.error_message)
                    self._emit_progress(
                        progress_callback,
                        "folder_failed",
                        state=state,
                        folder=folder_path.name,
                        error=folder_record.error_message,
                    )

                self._refresh_progress_counts(state)
                self._write_records(output_dir, single_records, list(folder_map.values()))
                self.save_state(state)
                self._emit_progress(progress_callback, "progress_updated", state=state)

            state = self.load_state(task_id)
            state.current_folder = None
            state.current_file = None
            state.status = "completed" if state.failed_images == 0 and state.failed_folders == 0 else "completed_with_errors"
            task_logger.log("task_finished", task_id=task_id, status=state.status)
            self.save_state(state)
            self._emit_progress(progress_callback, "task_finished", state=state)
            self._emit_progress(progress_callback, "progress_updated", state=state)
        except Exception as exc:  # noqa: BLE001
            state = self.load_state(task_id)
            state.status = "failed"
            state.error_message = str(exc)
            state.current_folder = None
            state.current_file = None
            task_logger.log("task_failed", task_id=task_id, error=str(exc))
            self.save_state(state)
            self._emit_progress(progress_callback, "task_failed", state=state, error=str(exc))
            self._emit_progress(progress_callback, "progress_updated", state=state)

    def _write_records(
        self,
        output_dir: Path,
        single_records: list[SingleImageRecord],
        folder_summary_records: list[FolderSummaryRecord],
    ) -> None:
        write_json(output_dir / "single_images.json", single_records)
        write_single_image_csv(output_dir / "single_images.csv", single_records)
        write_json(output_dir / "folder_summary.json", folder_summary_records)
        write_folder_summary_csv(output_dir / "folder_summary.csv", folder_summary_records)

    def _state_path(self, task_id: str) -> Path:
        return self.pipeline.settings.output_root / task_id / "task_state.json"

    @staticmethod
    def _refresh_progress_counts(state: TaskState) -> None:
        state.completed_images = sum(1 for item in state.file_records if item.status == "completed")
        state.failed_images = sum(1 for item in state.file_records if item.status == "failed")
        state.completed_folders = sum(1 for item in state.folder_records if item.status == "completed")
        state.failed_folders = sum(1 for item in state.folder_records if item.status == "failed")

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _load_record_list(path: Path, model: Callable) -> list:
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [model.model_validate(item) for item in payload]

    @staticmethod
    def _get_folder_record(state: TaskState, folder_name: str) -> TaskFolderRecord:
        for record in state.folder_records:
            if record.source_folder == folder_name:
                return record
        raise ValueError(f"未找到文件夹记录: {folder_name}")

    @staticmethod
    def _get_file_record(state: TaskState, folder_name: str, file_name: str) -> TaskFileRecord:
        for record in state.file_records:
            if record.source_folder == folder_name and record.source_file == file_name:
                return record
        raise ValueError(f"未找到图片记录: {folder_name}/{file_name}")

    @staticmethod
    def _should_process_folder(state: TaskState, folder_record: TaskFolderRecord, resume_failed_only: bool) -> bool:
        if not resume_failed_only:
            return True
        file_records = [item for item in state.file_records if item.source_folder == folder_record.source_folder]
        has_retryable_file = any(item.status in {"failed", "pending"} for item in file_records)
        if has_retryable_file:
            return True
        return not folder_record.folder_summary_done or folder_record.status != "completed"

    @staticmethod
    def _emit_progress(
        progress_callback: Callable[[str, dict[str, Any]], None] | None,
        event: str,
        **payload: Any,
    ) -> None:
        if progress_callback is None:
            return
        progress_payload = dict(payload)
        state = progress_payload.get("state")
        if isinstance(state, TaskState):
            progress_payload["state"] = state.model_copy(deep=True)
        progress_callback(event, progress_payload)
