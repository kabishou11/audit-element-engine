from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from app.config import Settings
from app.exporters import ensure_output_dir, write_folder_summary_csv, write_json, write_single_image_csv
from app.llm_clients import OpenAICompatibleJsonClient
from app.models import FolderSummaryRecord, IssueItem, SingleImageRecord, TaskOutput, VisualAnalysisResult
from app.rules import (
    build_rule_prompt_lines,
    load_rules,
    load_text,
    match_rule_by_type,
    recall_candidate_rules,
)


class VoucherAttachmentPipeline:
    CANDIDATE_CLASSIFICATION_SYSTEM_PROMPT = (
        "你是农村财务审计附件分类裁决器。"
        "你只能从用户提供的候选附件类型中选择最终结果。"
        "严禁输出候选列表之外的类型。"
        "只输出 JSON。"
    )

    PRIMARY_TYPE_PRIORITY = {
        "银行回单 / 电子回单": 100,
        "发票 / 电子发票": 95,
        "合同 / 协议": 90,
        "大额支出审批表": 85,
        "经济组织内部结算凭证": 85,
        "拨付请示": 80,
        "会议纪要": 80,
        "会议决议或成员大会记录": 80,
        "项目资金支付清单": 78,
        "费用付款凭证": 76,
        "报价单": 40,
        "费用明细": 35,
        "发货单或送货单": 35,
        "入库单或出库单": 35,
        "其他": 0,
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.visual_client = OpenAICompatibleJsonClient(
            settings.vision_config,
            max_retries=settings.model_max_retries,
            retry_backoff_seconds=settings.model_retry_backoff_seconds,
        )
        self.classification_client = OpenAICompatibleJsonClient(
            settings.get_stage_text_config("classifier"),
            max_retries=settings.model_max_retries,
            retry_backoff_seconds=settings.model_retry_backoff_seconds,
        )
        self.extractor_client = OpenAICompatibleJsonClient(
            settings.get_stage_text_config("extractor"),
            max_retries=settings.model_max_retries,
            retry_backoff_seconds=settings.model_retry_backoff_seconds,
        )
        self.audit_client = OpenAICompatibleJsonClient(
            settings.get_stage_text_config("audit"),
            max_retries=settings.model_max_retries,
            retry_backoff_seconds=settings.model_retry_backoff_seconds,
        )
        self.folder_client = OpenAICompatibleJsonClient(
            settings.get_stage_text_config("folder"),
            max_retries=settings.model_max_retries,
            retry_backoff_seconds=settings.model_retry_backoff_seconds,
        )
        self.rules = load_rules(settings.rules_excel_path)
        self.classification_system_prompt = load_text(settings.classification_prompt_path)
        self.visual_analysis_system_prompt = load_text(settings.visual_analysis_prompt_path)
        self.rule_prompt_lines = build_rule_prompt_lines(self.rules)

    def process_folder(self, folder_path: Path) -> TaskOutput:
        task_id = self._build_task_id(folder_path.name)
        output_dir = self.settings.output_root / task_id
        ensure_output_dir(output_dir)

        single_image_records: list[SingleImageRecord] = []
        image_paths = self._list_image_paths(folder_path)
        for image_path in image_paths:
            single_image_records.append(self.process_single_image(folder_path, image_path))

        folder_summary_record = self.process_folder_summary(folder_path, single_image_records)

        write_json(output_dir / "single_images.json", single_image_records)
        write_single_image_csv(output_dir / "single_images.csv", single_image_records)
        write_json(output_dir / "folder_summary.json", [folder_summary_record])
        write_folder_summary_csv(output_dir / "folder_summary.csv", [folder_summary_record])

        return TaskOutput(
            task_id=task_id,
            created_at=datetime.now(),
            single_image_records=single_image_records,
            folder_summary_records=[folder_summary_record],
            output_dir=str(output_dir.resolve()),
        )

    def process_multiple_folders(self, folder_paths: list[Path]) -> TaskOutput:
        task_id = self._build_task_id("all")
        output_dir = self.settings.output_root / task_id
        ensure_output_dir(output_dir)

        single_image_records: list[SingleImageRecord] = []
        folder_summary_records: list[FolderSummaryRecord] = []
        for folder_path in folder_paths:
            folder_output = self.process_folder(folder_path)
            single_image_records.extend(folder_output.single_image_records)
            folder_summary_records.extend(folder_output.folder_summary_records)

        write_json(output_dir / "single_images.json", single_image_records)
        write_single_image_csv(output_dir / "single_images.csv", single_image_records)
        write_json(output_dir / "folder_summary.json", folder_summary_records)
        write_folder_summary_csv(output_dir / "folder_summary.csv", folder_summary_records)

        return TaskOutput(
            task_id=task_id,
            created_at=datetime.now(),
            single_image_records=single_image_records,
            folder_summary_records=folder_summary_records,
            output_dir=str(output_dir.resolve()),
        )

    def process_single_image(
        self,
        folder_path: Path,
        image_path: Path,
        event_logger: Callable[..., None] | None = None,
    ) -> SingleImageRecord:
        image_started = time.perf_counter()
        visual_user_prompt = (
            "请分析当前财务附件图片并按 JSON 输出。"
            "请尽最大努力识别图片标题、关键正文、金额、日期、主体名称、账号、印章、签名、摘要等可见信息。"
            "如果是银行回单、发票、合同、会议纪要、审批表等明显票据或文书，必须明确写出。"
            "不要返回空对象。"
        )
        self._bind_event_logger(event_logger)
        self._log_stage(event_logger, "single_image_start", folder=folder_path.name, file=image_path.name)
        visual_raw = self.visual_client.analyze_image(
            system_prompt=self.visual_analysis_system_prompt,
            user_prompt=visual_user_prompt,
            image_path=image_path,
        )
        self._log_stage(event_logger, "visual_stage_done", folder=folder_path.name, file=image_path.name)
        visual_result = self._normalize_visual_result(visual_raw)
        candidate_rules = self._recall_candidate_rules_for_image(visual_result)
        candidate_prompt_lines = build_rule_prompt_lines(rule for rule, _ in candidate_rules)

        classification_user_prompt = f"""
请结合下列候选规则清单，对当前图片做受限分类。
要求：
1. 只能从候选规则清单中的“附件类型”选择一个最终结果。
2. 输出字段：
{{
  "attachment_type": "必须是规则清单中的原始附件类型文本",
  "classification_basis": "结合图片可见事实与规则描述的分类依据"
}}
3. 不允许输出候选规则清单外的新类型。

候选规则清单：
{candidate_prompt_lines}

候选召回分数：
{json.dumps([{ "attachment_type": rule.attachment_type, **details } for rule, details in candidate_rules], ensure_ascii=False, indent=2)}

参考分类原则文本：
{self.classification_system_prompt}

视觉识别结果：
{json.dumps(visual_result.model_dump(mode="json"), ensure_ascii=False, indent=2)}
""".strip()
        classification_raw = self.classification_client.analyze_text(
            system_prompt=self.CANDIDATE_CLASSIFICATION_SYSTEM_PROMPT,
            user_prompt=classification_user_prompt,
        )
        self._log_stage(event_logger, "classification_stage_done", folder=folder_path.name, file=image_path.name)
        classification_raw = self._normalize_candidate_classification(classification_raw, candidate_rules)

        rule = match_rule_by_type(self.rules, classification_raw["attachment_type"])
        elements_raw = self._extract_elements(rule, visual_result, image_path)
        self._log_stage(event_logger, "element_stage_done", folder=folder_path.name, file=image_path.name)
        issues_raw = self._generate_image_issues(rule, visual_result, elements_raw, image_path)
        self._log_stage(event_logger, "issue_stage_done", folder=folder_path.name, file=image_path.name)

        normalized_issues = self._normalize_issue_items(issues_raw, rule.audit_issue_references)
        self._log_stage(
            event_logger,
            "single_image_success",
            folder=folder_path.name,
            file=image_path.name,
            elapsed_seconds=round(time.perf_counter() - image_started, 3),
        )

        return SingleImageRecord(
            source_folder=folder_path.name,
            source_file=image_path.name,
            attachment_type=rule.attachment_type,
            attachment_description=rule.attachment_description,
            classification_basis=classification_raw.get("classification_basis", ""),
            selected_audit_elements=self._normalize_selected_audit_elements(elements_raw),
            issues=normalized_issues,
            visual_analysis=visual_result.model_dump(mode="json"),
            raw_outputs={
                "classification": classification_raw,
                "elements": elements_raw,
                "issues": issues_raw,
            },
        )

    def process_folder_summary(
        self,
        folder_path: Path,
        single_image_records: list[SingleImageRecord],
        event_logger: Callable[..., None] | None = None,
    ) -> FolderSummaryRecord:
        self._bind_event_logger(event_logger)
        self._log_stage(event_logger, "folder_summary_start", folder=folder_path.name, image_count=len(single_image_records))
        summary_input = [
            {
                "source_file": record.source_file,
                "attachment_type": record.attachment_type,
                "attachment_description": record.attachment_description,
                "classification_basis": record.classification_basis,
                "selected_audit_elements": record.selected_audit_elements,
                "issues": [item.model_dump(mode="json") for item in record.issues],
                "content_summary": record.visual_analysis.get("content_summary"),
                "key_entities": record.visual_analysis.get("key_entities"),
            }
            for record in single_image_records
        ]
        candidate_rules = self._recall_candidate_rules_for_folder(single_image_records)
        candidate_prompt_lines = build_rule_prompt_lines(rule for rule, _ in candidate_rules)
        classification_prompt = f"""
你正在处理一个完整凭证单元的整凭证审计。输入是该文件夹内每张图片已经处理完成的结果。
请重新做一次整凭证级别的受限分类，但不要把多附件凭证错误压缩成只有一种附件。

要求：
1. 必须输出一个 `primary_attachment_type` 作为整凭证主类型，它应代表该凭证单元中最核心、最能代表该笔业务的附件类型。
2. 同时输出 `attachment_type_distribution`，列出当前凭证单元中实际出现过的附件类型及数量。
3. 同时输出 `evidence_chain`，按图片列出每张图片在整凭证中的作用。
4. 不允许判断缺件、资料不全、应附未附。
5. 输出 JSON：
{{
  "primary_attachment_type": "规则清单中的原始附件类型文本",
  "attachment_type_distribution": {{
    "附件类型": 1
  }},
  "evidence_chain": [
    {{
      "source_file": "图片文件名",
      "attachment_type": "该图附件类型",
      "role_in_voucher": "该图在整凭证中的作用，如付款依据、审批依据、合同依据、报价依据"
    }}
  ],
  "classification_basis": "整凭证层面的分类依据",
  "folder_summary": "对该凭证单元当前已提供图片的客观概括"
}}

候选规则清单：
{candidate_prompt_lines}

候选召回分数：
{json.dumps([{ "attachment_type": rule.attachment_type, **details } for rule, details in candidate_rules], ensure_ascii=False, indent=2)}

参考分类原则文本：
{self.classification_system_prompt}

单图结果：
{json.dumps(summary_input, ensure_ascii=False, indent=2)}
""".strip()
        classification_raw = self.folder_client.analyze_text(
            system_prompt=self.CANDIDATE_CLASSIFICATION_SYSTEM_PROMPT,
            user_prompt=classification_prompt,
        )
        self._log_stage(event_logger, "folder_summary_classification_done", folder=folder_path.name)
        classification_raw = self._normalize_folder_candidate_classification(classification_raw, candidate_rules)
        attachment_type_distribution = self._normalize_attachment_type_distribution(
            classification_raw.get("attachment_type_distribution"),
            single_image_records,
        )
        primary_type = self._choose_primary_attachment_type(
            classification_raw.get("primary_attachment_type") or classification_raw.get("attachment_type"),
            attachment_type_distribution,
        )
        rule = match_rule_by_type(self.rules, primary_type)
        if rule.ai_audit_elements:
            elements_prompt = f"""
你正在根据整凭证数据提取审计要素。
只能从以下要素中选择并填值，不允许新增字段：
{json.dumps(rule.ai_audit_elements, ensure_ascii=False)}

输出 JSON：
{{
  "selected_audit_elements": {{
    "要素名": "识别值或null"
  }}
}}

要求：
1. 所有键必须来自给定要素列表。
2. 没有明确依据的值填 null。
3. 只基于给定的单图结果汇总，不做额外推断。

单图结果汇总：
{json.dumps(summary_input, ensure_ascii=False, indent=2)}
""".strip()
            elements_raw = self.folder_client.analyze_text(system_prompt="你是审计要素抽取助手。只输出 JSON。", user_prompt=elements_prompt)
        else:
            elements_raw = {"selected_audit_elements": {}}
        self._log_stage(event_logger, "folder_summary_elements_done", folder=folder_path.name)

        if rule.audit_issue_references:
            issues_prompt = f"""
你正在对整凭证单元生成可能存在的问题与建议。
请严格从给定“审计问题参考”中选择命中的规则问题，不允许自造新的问题类型。
补充约束：
1. 银行回单中的“交易日期”和“打印时间”不同，不能单独判定为异常。
2. 只有当同类核心字段存在明显冲突或缺乏必要识别依据时，才输出问题。
3. 如果没有命中的规则问题，返回空列表。

审计问题参考：
{json.dumps(rule.audit_issue_references, ensure_ascii=False, indent=2)}

输出 JSON：
{{
  "issues": [
    {{
      "rule_issue_reference": "必须是审计问题参考中的原文之一",
      "problem": "可能存在的问题描述",
      "suggestion": "对应建议",
      "basis": "来自当前整凭证数据的依据"
    }}
  ]
}}

整凭证分类结果：
{json.dumps(classification_raw, ensure_ascii=False, indent=2)}

整凭证要素：
{json.dumps(elements_raw, ensure_ascii=False, indent=2)}

单图结果汇总：
{json.dumps(summary_input, ensure_ascii=False, indent=2)}
""".strip()
            issues_raw = self.folder_client.analyze_text(system_prompt="你是审计问题生成助手。只输出 JSON。", user_prompt=issues_prompt)
        else:
            issues_raw = {"issues": []}
        self._log_stage(event_logger, "folder_summary_issues_done", folder=folder_path.name)

        normalized_issues = self._normalize_issue_items(issues_raw, rule.audit_issue_references)
        self._log_stage(event_logger, "folder_summary_success", folder=folder_path.name)

        return FolderSummaryRecord(
            source_folder=folder_path.name,
            included_files=[record.source_file for record in single_image_records],
            attachment_type=rule.attachment_type,
            attachment_type_distribution=attachment_type_distribution,
            attachment_description=rule.attachment_description,
            classification_basis=classification_raw.get("classification_basis", ""),
            evidence_chain=self._normalize_evidence_chain(
                classification_raw.get("evidence_chain"),
                single_image_records,
            ),
            selected_audit_elements=self._normalize_selected_audit_elements(elements_raw),
            issues=normalized_issues,
            folder_summary=classification_raw.get("folder_summary", ""),
            referenced_single_image_files=[record.source_file for record in single_image_records],
            raw_outputs={
                "classification": classification_raw,
                "elements": elements_raw,
                "issues": issues_raw,
            },
        )

    def _extract_elements(
        self,
        rule,
        visual_result: VisualAnalysisResult,
        image_path: Path,
    ) -> dict:
        if not rule.ai_audit_elements:
            return {"selected_audit_elements": {}}
        prompt = f"""
你正在对单张凭证附件图片提取审计要素。
图片文件名：{image_path.name}
限定要素列表：
{json.dumps(rule.ai_audit_elements, ensure_ascii=False)}

输出 JSON：
{{
  "selected_audit_elements": {{
    "要素名": "识别值或null"
  }}
}}

要求：
1. 所有键必须来自限定要素列表。
2. 不允许新增任何其他字段。
3. 没有明确识别依据的值填 null。
4. 只依据当前图片可见事实。

视觉识别结果：
{json.dumps(visual_result.model_dump(mode="json"), ensure_ascii=False, indent=2)}
""".strip()
        return self.extractor_client.analyze_text(system_prompt="你是审计要素抽取助手。只输出 JSON。", user_prompt=prompt)

    def _generate_image_issues(self, rule, visual_result: VisualAnalysisResult, elements_raw: dict, image_path: Path) -> dict:
        if not rule.audit_issue_references:
            return {"issues": []}
        prompt = f"""
你正在对单张凭证附件图片做一轮审计。
请严格从以下审计问题模式中选择命中的规则问题，不允许自造新的问题类型。
不要输出“缺失、应附未附、资料不全”这类需要知道完整凭证范围才能判断的问题。
只根据当前图片的可见事实，输出可能存在的问题描述和建议。
补充约束：
1. 银行回单中的“交易日期”和“打印时间”不同，不能单独判定为异常。
2. 若图片信息清晰且关键字段完整，不必强行生成问题，可返回空列表。

审计问题参考：
{json.dumps(rule.audit_issue_references, ensure_ascii=False, indent=2)}

输出 JSON：
{{
  "issues": [
    {{
      "rule_issue_reference": "必须是审计问题参考中的原文之一",
      "problem": "可能存在的问题描述",
      "suggestion": "对应建议",
      "basis": "来自当前图片的依据"
    }}
  ]
}}

图片文件名：
{image_path.name}

视觉识别结果：
{json.dumps(visual_result.model_dump(mode="json"), ensure_ascii=False, indent=2)}

审计要素：
{json.dumps(elements_raw, ensure_ascii=False, indent=2)}
""".strip()
        return self.audit_client.analyze_text(system_prompt="你是审计问题生成助手。只输出 JSON。", user_prompt=prompt)

    @staticmethod
    def _normalize_selected_audit_elements(payload: dict[str, Any]) -> dict[str, Any]:
        selected = payload.get("selected_audit_elements", payload)
        return selected if isinstance(selected, dict) else {}

    @staticmethod
    def _normalize_issue_items(payload: dict[str, Any], allowed_rule_refs: list[str] | None = None) -> list[IssueItem]:
        issue_items = payload.get("issues", payload)
        if isinstance(issue_items, dict) and "problem" in issue_items:
            issue_items = [issue_items]
        if not isinstance(issue_items, list):
            return []
        allowed = set(allowed_rule_refs or [])
        normalized: list[IssueItem] = []
        for item in issue_items:
            if not isinstance(item, dict):
                continue
            if "problem" not in item or "suggestion" not in item:
                continue
            rule_issue_reference = item.get("rule_issue_reference")
            if allowed and rule_issue_reference not in allowed:
                continue
            normalized.append(IssueItem(**item))
        return normalized

    @staticmethod
    def _normalize_attachment_type_distribution(
        payload: Any,
        single_image_records: list[SingleImageRecord],
    ) -> dict[str, int]:
        if isinstance(payload, dict):
            normalized: dict[str, int] = {}
            for key, value in payload.items():
                try:
                    normalized[str(key)] = int(value)
                except (TypeError, ValueError):
                    continue
            if normalized:
                return normalized
        return dict(Counter(record.attachment_type for record in single_image_records))

    @staticmethod
    def _normalize_evidence_chain(
        payload: Any,
        single_image_records: list[SingleImageRecord],
    ) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            normalized = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                source_file = item.get("source_file")
                attachment_type = item.get("attachment_type")
                role_in_voucher = item.get("role_in_voucher")
                if source_file and attachment_type and role_in_voucher:
                    normalized.append(
                        {
                            "source_file": source_file,
                            "attachment_type": attachment_type,
                            "role_in_voucher": role_in_voucher,
                        }
                    )
            if normalized:
                return normalized
        fallback = []
        for record in single_image_records:
            fallback.append(
                {
                    "source_file": record.source_file,
                    "attachment_type": record.attachment_type,
                    "role_in_voucher": VoucherAttachmentPipeline._default_role_for_attachment_type(record.attachment_type),
                }
            )
        return fallback

    @classmethod
    def _choose_primary_attachment_type(
        cls,
        model_primary_type: Any,
        attachment_type_distribution: dict[str, int],
    ) -> str:
        if not attachment_type_distribution:
            return str(model_primary_type or "其他")
        candidate_types = list(attachment_type_distribution.keys())
        if model_primary_type in candidate_types and cls.PRIMARY_TYPE_PRIORITY.get(str(model_primary_type), 50) >= 60:
            return str(model_primary_type)
        return max(
            candidate_types,
            key=lambda item: (cls.PRIMARY_TYPE_PRIORITY.get(item, 50), attachment_type_distribution.get(item, 0), item),
        )

    @staticmethod
    def _default_role_for_attachment_type(attachment_type: str) -> str:
        role_map = {
            "银行回单 / 电子回单": "付款结果依据",
            "发票 / 电子发票": "支出合法性依据",
            "合同 / 协议": "业务约定依据",
            "大额支出审批表": "内部审批依据",
            "报价单": "采购比价依据",
            "经济组织内部结算凭证": "内部结算依据",
            "拨付请示": "资金申请依据",
            "会议纪要": "会议决策依据",
            "会议决议或成员大会记录": "集体决策依据",
            "项目资金支付清单": "付款明细依据",
            "费用付款凭证": "付款事项依据",
            "发货单或送货单": "货物交付依据",
            "入库单或出库单": "物资流转依据",
        }
        return role_map.get(attachment_type, "附件事实依据")

    @staticmethod
    def _normalize_visual_result(visual_raw: dict[str, Any]) -> VisualAnalysisResult:
        key_entities_raw = visual_raw.get("key_entities", {})
        key_entities = key_entities_raw if isinstance(key_entities_raw, dict) else {}

        date_found: list[str] = []
        for key in ("date_found", "date", "print_time"):
            value = key_entities.get(key)
            if isinstance(value, list):
                date_found.extend([str(item) for item in value if item])
            elif value:
                date_found.append(str(value))
        if date_found:
            key_entities["date_found"] = date_found

        signatures_raw = visual_raw.get("signatures_and_seals", {})
        if isinstance(signatures_raw, list):
            signatures = {
                "has_official_seal": bool(signatures_raw),
                "seal_details": signatures_raw,
                "has_handwritten_signature": False,
                "signature_details": [],
                "has_digital_signature_image": bool(signatures_raw),
            }
        else:
            signatures = signatures_raw if signatures_raw else {}

        visual_elements_raw = visual_raw.get("visual_elements", {})
        if isinstance(visual_elements_raw, list):
            visual_elements = {
                "notes": visual_elements_raw,
                "is_multi_page_scan": False,
                "image_quality": "清晰",
                "text_layout": "混合布局",
            }
        else:
            visual_elements = visual_elements_raw if visual_elements_raw else {}

        return VisualAnalysisResult(
            document_category=visual_raw.get("document_category"),
            specific_type=visual_raw.get("specific_type"),
            key_entities=key_entities,
            signatures_and_seals=signatures,
            content_summary=visual_raw.get("content_summary"),
            visual_elements=visual_elements,
            raw_json=visual_raw,
        )

    def _recall_candidate_rules_for_image(
        self,
        visual_result: VisualAnalysisResult,
    ) -> list[tuple[Any, dict[str, float]]]:
        source_text = self._build_image_source_text(visual_result)
        return recall_candidate_rules(
            rules=self.rules,
            source_text=source_text,
            top_k=self.settings.rule_recall_top_k,
            min_score=self.settings.rule_recall_min_score,
            alias_bonus=self.settings.rule_alias_bonus,
            fragment_bonus=self.settings.rule_fragment_bonus,
            element_bonus=self.settings.rule_element_bonus,
            bigram_weight=self.settings.rule_bigram_weight,
        )

    def _recall_candidate_rules_for_folder(
        self,
        single_image_records: list[SingleImageRecord],
    ) -> list[tuple[Any, dict[str, float]]]:
        chunks: list[str] = []
        for record in single_image_records:
            chunks.extend(
                [
                    record.attachment_type,
                    record.attachment_description,
                    record.classification_basis,
                    json.dumps(record.selected_audit_elements, ensure_ascii=False),
                    record.visual_analysis.get("content_summary") or "",
                ]
            )
        source_text = "\n".join(chunks)
        return recall_candidate_rules(
            rules=self.rules,
            source_text=source_text,
            top_k=self.settings.rule_recall_top_k,
            min_score=self.settings.rule_recall_min_score,
            alias_bonus=self.settings.rule_alias_bonus,
            fragment_bonus=self.settings.rule_fragment_bonus,
            element_bonus=self.settings.rule_element_bonus,
            bigram_weight=self.settings.rule_bigram_weight,
        )

    @staticmethod
    def _build_image_source_text(visual_result: VisualAnalysisResult) -> str:
        key_entities = visual_result.key_entities if isinstance(visual_result.key_entities, dict) else {}
        flattened_entities = " ".join(f"{key}:{value}" for key, value in key_entities.items())
        flattened_signatures = json.dumps(visual_result.signatures_and_seals, ensure_ascii=False)
        flattened_visuals = json.dumps(visual_result.visual_elements, ensure_ascii=False)
        return "\n".join(
            [
                visual_result.document_category or "",
                visual_result.specific_type or "",
                visual_result.content_summary or "",
                flattened_entities,
                flattened_signatures,
                flattened_visuals,
            ]
        )

    @staticmethod
    def _normalize_candidate_classification(
        payload: dict[str, Any],
        candidate_rules: list[tuple[Any, dict[str, float]]],
    ) -> dict[str, Any]:
        allowed = {rule.attachment_type for rule, _ in candidate_rules}
        attachment_type = payload.get("attachment_type")
        if attachment_type in allowed:
            return payload
        top_rule = candidate_rules[0][0]
        return {
            **payload,
            "attachment_type": top_rule.attachment_type,
            "classification_basis": payload.get("classification_basis") or "模型输出超出候选范围，已回退到本地召回第一候选类型。",
        }

    @staticmethod
    def _normalize_folder_candidate_classification(
        payload: dict[str, Any],
        candidate_rules: list[tuple[Any, dict[str, float]]],
    ) -> dict[str, Any]:
        allowed = {rule.attachment_type for rule, _ in candidate_rules}
        primary_attachment_type = payload.get("primary_attachment_type") or payload.get("attachment_type")
        if primary_attachment_type in allowed:
            payload["primary_attachment_type"] = primary_attachment_type
            return payload
        top_rule = candidate_rules[0][0]
        payload["primary_attachment_type"] = top_rule.attachment_type
        if not payload.get("classification_basis"):
            payload["classification_basis"] = "模型输出超出候选范围，已回退到本地召回第一候选类型。"
        return payload

    def _list_image_paths(self, folder_path: Path) -> list[Path]:
        return sorted(
            [
                path
                for path in folder_path.iterdir()
                if path.is_file() and path.suffix.lower() in self.settings.image_extensions
            ]
        )

    @staticmethod
    def _build_task_id(scope: str) -> str:
        return f"{datetime.now():%Y%m%d_%H%M%S}_{scope}_{uuid4().hex[:8]}"

    def _bind_event_logger(self, event_logger: Callable[..., None] | None) -> None:
        for client in (
            self.visual_client,
            self.classification_client,
            self.extractor_client,
            self.audit_client,
            self.folder_client,
        ):
            client.event_logger = event_logger

    @staticmethod
    def _log_stage(event_logger: Callable[..., None] | None, event: str, **payload: Any) -> None:
        if event_logger:
            event_logger(event, **payload)
