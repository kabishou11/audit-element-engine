from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from tqdm import tqdm

from app.config import get_settings
from app.main import _resolve_folder
from app.models import TaskState
from app.pipeline import VoucherAttachmentPipeline
from app.task_manager import TaskManager


class CliRunner:
    def __init__(self) -> None:
        self.base_settings = get_settings()
        self.pipeline = VoucherAttachmentPipeline(self.base_settings)
        self.task_manager = TaskManager(self.pipeline)

    def run(self, scope: str, folder_names: list[str] | None, task_id: str | None) -> None:
        if task_id:
            self._resume_existing_task(task_id)
            return

        folder_paths = self._resolve_scope(scope, folder_names)
        state = self.task_manager.create_new_task(
            scope=scope,
            folder_paths=folder_paths,
            overrides=None,
        )
        self._run_sync(state, folder_paths, resumed=False)

    def _resume_existing_task(self, task_id: str) -> None:
        state = self.task_manager.prepare_resume(task_id)
        folder_paths = [_resolve_folder(folder_name) for folder_name in state.folder_names]
        self._run_sync(state, folder_paths, resumed=True)

    def list_tasks(self, limit: int = 20) -> None:
        tasks = self.task_manager.list_tasks(limit=limit)
        if not tasks:
            print("未找到历史任务。")
            return
        print("最近任务列表：")
        for item in tasks:
            print(
                f"{item.task_id} | 状态={item.status} | 图片={item.completed_images + item.failed_images}/{item.total_images} "
                f"(失败{item.failed_images}) | 文件夹={item.completed_folders + item.failed_folders}/{item.total_folders} "
                f"(失败{item.failed_folders}) | 更新时间={item.updated_at:%Y-%m-%d %H:%M:%S}"
            )

    def resume_latest_task(self) -> None:
        tasks = self.task_manager.list_tasks(limit=1)
        if not tasks:
            raise SystemExit("未找到可续跑任务。")
        latest_task = tasks[0]
        print(f"续跑最近任务: {latest_task.task_id}")
        self._resume_existing_task(latest_task.task_id)

    def _resolve_scope(self, scope: str, folder_names: list[str] | None) -> list[Path]:
        if scope == "all":
            return sorted(path for path in self.base_settings.attachment_root.iterdir() if path.is_dir())
        if not folder_names:
            raise ValueError("单文件夹模式必须提供 --folder")
        return [_resolve_folder(folder_name) for folder_name in folder_names]

    def _run_sync(self, state: TaskState, folder_paths: list[Path], resumed: bool) -> None:
        output_dir = Path(state.output_dir)
        image_bar = tqdm(
            total=state.total_images,
            initial=state.completed_images + state.failed_images,
            desc="图片进度",
            dynamic_ncols=True,
            position=0,
        )
        folder_bar = tqdm(
            total=state.total_folders,
            initial=state.completed_folders + state.failed_folders,
            desc="文件夹进度",
            dynamic_ncols=True,
            position=1,
        )

        tqdm.write(f"{'续跑任务' if resumed else '新任务开始'}: {state.task_id}")
        tqdm.write(f"输出目录: {output_dir}")
        tqdm.write(f"任务日志: {output_dir / 'task.log'}")

        def progress_callback(event: str, payload: dict[str, Any]) -> None:
            callback_state = payload.get("state")
            if isinstance(callback_state, TaskState):
                image_bar.n = callback_state.completed_images + callback_state.failed_images
                folder_bar.n = callback_state.completed_folders + callback_state.failed_folders
                image_bar.refresh()
                folder_bar.refresh()

            folder = payload.get("folder")
            file_name = payload.get("file")
            error = payload.get("error")

            if event == "folder_started" and folder:
                tqdm.write(f"[文件夹开始] {folder}")
            elif event == "image_started" and folder and file_name:
                tqdm.write(f"[图片开始] {folder}/{file_name}")
            elif event == "image_completed" and folder and file_name:
                tqdm.write(f"[图片完成] {folder}/{file_name}")
            elif event in {"image_failed", "image_timeout"} and folder and file_name:
                tqdm.write(f"[图片失败] {folder}/{file_name} -> {error}")
            elif event == "folder_completed" and folder:
                tqdm.write(f"[文件夹完成] {folder}")
            elif event == "folder_failed" and folder:
                tqdm.write(f"[文件夹失败] {folder} -> {error}")
            elif event == "task_cancelled":
                tqdm.write("任务已取消")
            elif event == "task_failed":
                tqdm.write(f"任务失败: {error}")
            elif event == "task_finished" and isinstance(callback_state, TaskState):
                tqdm.write(f"任务结束: {callback_state.task_id} -> {callback_state.status}")

        try:
            final_state = self.task_manager.run_task_sync(
                task_id=state.task_id,
                folder_paths=folder_paths,
                progress_callback=progress_callback,
            )
        except KeyboardInterrupt:
            interrupted_state = self.task_manager.load_state(state.task_id)
            interrupted_state.status = "pending"
            interrupted_state.error_message = "终端手动中断，可使用 --task-id 续跑。"
            self.task_manager.save_state(interrupted_state)
            tqdm.write(f"任务已中断并保存续跑状态: {interrupted_state.task_id}")
            raise SystemExit(130)
        finally:
            image_bar.close()
            folder_bar.close()

        if final_state.status in {"failed", "cancelled"}:
            raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="凭证附件终端跑批工具")
    parser.add_argument("--task-id", default="", help="已有 task_id 时按该任务续跑")
    parser.add_argument("--all", action="store_true", help="全量跑所有文件夹")
    parser.add_argument("--folder", action="append", default=[], help="指定单个或多个文件夹名")
    parser.add_argument("--list-tasks", action="store_true", help="列出最近任务，方便查 task_id")
    parser.add_argument("--resume-latest", action="store_true", help="直接续跑最近一个任务")
    parser.add_argument("--limit", type=int, default=20, help="配合 --list-tasks 使用，控制返回数量")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    runner = CliRunner()
    try:
        if args.list_tasks:
            runner.list_tasks(limit=max(args.limit, 1))
            return
        if args.resume_latest:
            runner.resume_latest_task()
            return
        if args.task_id:
            runner.run(scope="all", folder_names=None, task_id=args.task_id)
            return
        if args.all:
            runner.run(scope="all", folder_names=None, task_id=None)
            return
        if args.folder:
            runner.run(scope="folder", folder_names=args.folder, task_id=None)
            return
        raise SystemExit("请至少提供 --all、--folder、--task-id、--list-tasks 或 --resume-latest")
    except HTTPException as exc:
        raise SystemExit(exc.detail) from exc


if __name__ == "__main__":
    main()
