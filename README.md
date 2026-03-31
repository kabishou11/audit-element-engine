# audit-element-engine

三资监管凭证附件识别服务。

基于 FastAPI + Qwen(ModelScope OpenAI 兼容接口) 多模态模型，实现：

- 单张图片三层识别分类
- 单张图片审计要素提取
- 单张图片问题与建议生成
- 同一凭证单元的整凭证汇总识别与审计
- JSON 与 CSV 导出

当前分类结构为：

1. `Qwen` 多模态模型抽取视觉事实
2. 本地规则召回候选附件类型
3. `Qwen` 仅在候选集内做最终裁决

## 目录约定

- `凭证附件/`：输入目录，每个子文件夹代表一个完整凭证单元
- `参考规则数据/`：规则与提示词目录
- `outputs/`：处理输出目录

## 推荐使用方式

当前推荐直接走终端 CLI 跑批，不再依赖前端调试页。

优点：

- 终端实时显示 `tqdm` 图片进度和文件夹进度
- 每张图片开始/完成/失败会输出一条简洁日志
- `Ctrl+C` 中断后可按 `task_id` 续跑
- 结果仍然持续写入 `outputs/<task_id>/` 下的 `json/csv`
- CLI 与 FastAPI API 共用同一套 `TaskManager` 任务执行代码

## 环境准备

项目环境约定：

- 只使用根目录下的 `venv/`
- 依赖通过 `requirements.txt` 安装

1. 创建并激活 `venv`
2. 如 `venv` 尚未创建，可执行：`python -m venv venv`
3. 安装依赖：`venv\Scripts\python.exe -m pip install -r requirements.txt`
4. 复制 `.env.example` 为 `.env` 并填写模型配置

## 终端跑批

全量处理全部凭证文件夹：

```powershell
venv\Scripts\python.exe -m app.cli --all
```

只处理指定文件夹：

```powershell
venv\Scripts\python.exe -m app.cli --folder "记67"
```

一次处理多个指定文件夹：

```powershell
venv\Scripts\python.exe -m app.cli --folder "记67" --folder "记68"
```

按已有任务 `task_id` 续跑：

```powershell
venv\Scripts\python.exe -m app.cli --task-id 20260330_143636_all_7ad534af
```

先查看最近任务列表，再决定续跑哪个：

```powershell
venv\Scripts\python.exe -m app.cli --list-tasks
```

只看最近 10 个任务：

```powershell
venv\Scripts\python.exe -m app.cli --list-tasks --limit 10
```

直接续跑最近一个任务：

```powershell
venv\Scripts\python.exe -m app.cli --resume-latest
```

## 续跑旧任务

推荐按下面顺序操作。

### 情况 1：你知道旧任务的 `task_id`

直接续跑：

```powershell
venv\Scripts\python.exe -m app.cli --task-id 20260330_143636_all_7ad534af
```

### 情况 2：你不知道 `task_id`

先列出最近任务：

```powershell
venv\Scripts\python.exe -m app.cli --list-tasks
```

终端会显示类似：

```text
20260330_143636_all_7ad534af | 状态=completed_with_errors | 图片=12/15 (失败3) | 文件夹=5/7 (失败2) | 更新时间=2026-03-30 15:11:52
```

然后复制对应 `task_id` 再续跑：

```powershell
venv\Scripts\python.exe -m app.cli --task-id 20260330_143636_all_7ad534af
```

### 情况 3：你就想直接接着最近一次任务跑

```powershell
venv\Scripts\python.exe -m app.cli --resume-latest
```

### 续跑时实际会发生什么

- 已完成的单张图片会自动跳过，不会重跑。
- 失败的单张图片会重试。
- 失败图片所在的文件夹会重新生成一次 `folder_summary.json` / `folder_summary.csv`。
- 没有失败图片且整凭证汇总已完成的文件夹，续跑时不会重跑，也不会重写其 `folder_summary`。
- 整个文件夹如果之前整凭证汇总失败，也会在当前文件夹单图处理结束后重新做整凭证汇总。
- 如果旧任务原来显示 `running`，但其实进程已经没了，CLI 会自动把它改回可续跑状态再继续。
- 如果旧任务已经 `completed`，再次续跑会很快结束，因为没有可重跑内容。
- 如果你在续跑过程中再次按 `Ctrl+C`，任务会再次保存为可续跑状态。

### 我建议你的实际用法

如果你是日常跑批，直接用下面两条就够了：

1. 先看最近任务

```powershell
venv\Scripts\python.exe -m app.cli --list-tasks
```

2. 续跑指定任务或最近任务

```powershell
venv\Scripts\python.exe -m app.cli --task-id <task_id>
```

或者：

```powershell
venv\Scripts\python.exe -m app.cli --resume-latest
```

运行时说明：

- 终端会显示两个进度条：
  - `图片进度`
  - `文件夹进度`
- 同时会输出少量关键日志：
  - `[文件夹开始]`
  - `[图片开始]`
  - `[图片完成]`
  - `[图片失败]`
  - `[文件夹完成]`
  - `[文件夹失败]`
- 详细模型重试与阶段日志写入 `outputs/<task_id>/task.log`
- 模型如果返回空响应、`<think>` 包裹内容或不可解析 JSON，会自动按重试策略再次请求
- 如果按 `Ctrl+C` 中断，任务会保存为可续跑状态，后续可用 `--task-id` 继续

## 如需 API 服务

如果你仍然需要 FastAPI 接口，再单独启动服务：

```powershell
venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 主要接口

- `GET /health`
- `POST /api/v1/process/folder`
- `POST /api/v1/process/all`
- `GET /api/v1/tasks`
- `GET /api/v1/tasks/{task_id}`
- `GET /api/v1/tasks/{task_id}/result`
- `POST /api/v1/tasks/{task_id}/resume`
- `POST /api/v1/tasks/{task_id}/cancel`

## 任务机制

- 启动接口立即返回 `task_id`，后台持续处理。
- 每处理完一张图片，都会立即落盘更新：
  - `task_state.json`
  - `single_images.json`
  - `single_images.csv`
- 每处理完一个凭证文件夹，都会落盘更新：
  - `folder_summary.json`
  - `folder_summary.csv`
- 支持中断后续跑：调用 `resume` 会跳过已完成图片，只继续未完成或失败项目。
- 支持取消：调用 `cancel` 会将任务标记为 `cancel_requested`，当前图片处理结束后停止。
- 状态文件保存在 `outputs/<task_id>/task_state.json`。

## 关键配置

- `VISION_API_KEY` / `VISION_BASE_URL` / `VISION_MODEL`
- `MINIMAX_API_KEY` / `MINIMAX_BASE_URL` / `MINIMAX_MODEL`
- `CLASSIFIER_MODEL_SOURCE` / `CLASSIFIER_MODEL_NAME`
- `EXTRACTOR_MODEL_SOURCE` / `EXTRACTOR_MODEL_NAME`
- `AUDIT_MODEL_SOURCE` / `AUDIT_MODEL_NAME`
- `FOLDER_MODEL_SOURCE` / `FOLDER_MODEL_NAME`
- `VISION_TEMPERATURE` / `MINIMAX_TEMPERATURE`
- `RULES_EXCEL_PATH`
- `CLASSIFICATION_PROMPT_PATH`
- `VISUAL_ANALYSIS_PROMPT_PATH`
- `RULE_RECALL_TOP_K`
- `RULE_RECALL_MIN_SCORE`
- `RULE_ALIAS_BONUS`
- `RULE_FRAGMENT_BONUS`
- `RULE_ELEMENT_BONUS`
- `RULE_BIGRAM_WEIGHT`
- `TASK_LOG_ENABLED`
- `MODEL_MAX_RETRIES`
- `MODEL_RETRY_BACKOFF_SECONDS`
- `SINGLE_IMAGE_TIMEOUT_SECONDS`

其中各阶段的 `*_MODEL_SOURCE` 可选：

- `minimax`：走 `MINIMAX_*` 配置
- `vision`：走 `VISION_*` 配置，也就是当前的 Qwen / ModelScope 配置

## 当前建议模型组合

- 多模态识别：`Qwen/Qwen3.5-35B-A3B`
- 单图分类：`MiniMax-M2.7`
- 要素提取：`MiniMax-M2.7`
- 单图审计问题：`MiniMax-M2.7`
- 整凭证汇总：`MiniMax-M2.7`

如果 `MINIMAX_API_KEY` 留空，文本阶段会自动回退到 Qwen。

## 推荐起跑配置

建议先用当前 `.env` 默认组合直接跑批：

- 多模态识别：Qwen
- 单图分类：MiniMax
- 要素提取：MiniMax
- 单图审计问题：MiniMax
- 整凭证汇总：MiniMax

这样能把最重的“看图”留给多模态模型，把其余文本阶段分流给 MiniMax，整体更稳，也更省多模态资源。

## API 跑批示例

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/process/all"
```

查看最近任务列表：

```powershell
curl "http://127.0.0.1:8000/api/v1/tasks?limit=20"
```

查看单个任务状态：

```powershell
curl "http://127.0.0.1:8000/api/v1/tasks/20260330_143636_all_7ad534af"
```

恢复指定任务：

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/tasks/20260330_143636_all_7ad534af/resume"
```

## 输出位置

每个任务会在 `outputs/<task_id>/` 下生成：

- `task_state.json`：任务状态与进度
- `task.log`：任务级执行日志，包含每张图开始/完成/失败、模型请求重试、整凭证汇总等事件
- `single_images.json`
- `single_images.csv`
- `folder_summary.json`
- `folder_summary.csv`

## 输出说明

每次任务会在 `outputs/<task_id>/` 生成：

- `single_images.json`
- `single_images.csv`
- `folder_summary.json`
- `folder_summary.csv`

所有记录均带来源字段，便于追溯图片与凭证单元。
