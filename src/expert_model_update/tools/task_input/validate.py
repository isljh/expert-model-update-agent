"""Validate the minimal TaskInput v0.2 contract without third-party packages."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "0.2"
DOC_PATH = "docs/product-specs/task-intake.md"
DATA_USES = {"analysis", "training", "validation", "evaluation"}
GOAL_STATUSES = {"missing", "proposed", "confirmed"}
PRESERVATION_MODES = {"unconfirmed", "specified", "none"}
REQUIRED_QUESTION_STATUSES = {"unanswered", "answered"}
OPTIONAL_QUESTION_STATUSES = {"unanswered", "answered", "skipped"}
CONFIRMATION_STATUSES = {"pending", "confirmed", "rejected"}


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    path: str
    problem: str
    fix: str

    def render(self) -> str:
        return (
            f"[{self.code}] {self.path}: {self.problem}\n"
            f"  修复：{self.fix}\n"
            f"  参考：{DOC_PATH}"
        )


@dataclass(frozen=True)
class RequirementGap:
    target_fields: tuple[str, ...]
    problem: str
    fix: str


class Checker:
    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []

    def add(self, code: str, path: str, problem: str, fix: str) -> None:
        self.issues.append(ValidationIssue(code, path, problem, fix))

    def object(
        self,
        value: Any,
        path: str,
        required: Iterable[str],
        allowed: Iterable[str],
    ) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            self.add("TYPE", path, "应为对象。", "按 TaskInput v0.2 结构提供键值对象。")
            return None
        required_set = set(required)
        allowed_set = set(allowed)
        for key in sorted(required_set - value.keys()):
            self.add("REQUIRED", f"{path}.{key}", "缺少必填字段。", "补充该字段；业务值未知时使用规范允许的 null、空数组或未确认状态。")
        for key in sorted(value.keys() - allowed_set):
            self.add("UNKNOWN", f"{path}.{key}", "存在 v0.2 未定义的字段。", "删除该字段，或先修订契约再使用。")
        return value

    def string(self, value: Any, path: str, *, nullable: bool = False) -> None:
        if value is None and nullable:
            return
        if not isinstance(value, str):
            self.add("TYPE", path, "应为字符串。", "提供文本；未知且允许为空时使用 null。")
            return
        if not value.strip():
            self.add("EMPTY", path, "字符串不能为空白。", "提供明确内容，或在允许时使用 null 并生成对应必答问题。")

    def integer(self, value: Any, path: str, *, minimum: int = 1) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            self.add("TYPE", path, "应为整数。", f"提供不小于 {minimum} 的整数。")
        elif value < minimum:
            self.add("RANGE", path, f"不能小于 {minimum}。", f"提供不小于 {minimum} 的整数。")

    def enum(self, value: Any, path: str, allowed: Iterable[str]) -> None:
        allowed_set = set(allowed)
        if value not in allowed_set:
            values = "、".join(sorted(allowed_set))
            self.add("ENUM", path, f"值 {value!r} 不在允许范围内。", f"使用以下值之一：{values}。")

    def string_list(self, value: Any, path: str, *, allowed: set[str] | None = None) -> None:
        if not isinstance(value, list):
            self.add("TYPE", path, "应为数组。", "使用 JSON 数组保存零个或多个值。")
            return
        seen: set[str] = set()
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            self.string(item, item_path)
            if isinstance(item, str):
                if allowed is not None and item not in allowed:
                    self.enum(item, item_path, allowed)
                if item in seen:
                    self.add("DUPLICATE", item_path, f"值 {item!r} 重复。", "删除重复值。")
                seen.add(item)


def value_at(data: Any, *keys: str) -> Any:
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_shape(data: Any) -> list[ValidationIssue]:
    checker = Checker()
    root_fields = {
        "schema_version",
        "task_id",
        "task_input_version",
        "request",
        "assets",
        "requirements",
        "intake",
    }
    root = checker.object(data, "$", root_fields, root_fields)
    if root is None:
        return checker.issues

    if root.get("schema_version") != SCHEMA_VERSION:
        checker.add("VERSION", "$.schema_version", f"当前校验器只支持 {SCHEMA_VERSION}。", f"将值设为 {SCHEMA_VERSION}，或使用匹配的校验器。")
    checker.string(root.get("task_id"), "$.task_id")
    checker.integer(root.get("task_input_version"), "$.task_input_version")

    request = checker.object(
        root.get("request"),
        "$.request",
        {"raw_prompt", "update_goal"},
        {"raw_prompt", "update_goal"},
    )
    if request is not None:
        checker.string(request.get("raw_prompt"), "$.request.raw_prompt", nullable=True)
        goal = checker.object(
            request.get("update_goal"),
            "$.request.update_goal",
            {"summary", "reason", "status"},
            {"summary", "reason", "status"},
        )
        if goal is not None:
            checker.string(goal.get("summary"), "$.request.update_goal.summary", nullable=True)
            checker.string(goal.get("reason"), "$.request.update_goal.reason", nullable=True)
            checker.enum(goal.get("status"), "$.request.update_goal.status", GOAL_STATUSES)

    assets = checker.object(
        root.get("assets"),
        "$.assets",
        {"target_project", "base_model", "new_data"},
        {"target_project", "base_model", "new_data"},
    )
    if assets is not None:
        project = checker.object(
            assets.get("target_project"),
            "$.assets.target_project",
            {"location"},
            {"location"},
        )
        if project is not None:
            checker.string(project.get("location"), "$.assets.target_project.location", nullable=True)

        model = checker.object(
            assets.get("base_model"),
            "$.assets.base_model",
            {"checkpoint", "discovery_authorized"},
            {"checkpoint", "config", "discovery_authorized"},
        )
        if model is not None:
            checker.string(model.get("checkpoint"), "$.assets.base_model.checkpoint", nullable=True)
            checker.string(model.get("config"), "$.assets.base_model.config", nullable=True)
            discovery = model.get("discovery_authorized")
            if discovery is not None and not isinstance(discovery, bool):
                checker.add("TYPE", "$.assets.base_model.discovery_authorized", "应为布尔值或 null。", "使用 true、false；尚未回答时使用 null。")

        new_data = checker.object(
            assets.get("new_data"),
            "$.assets.new_data",
            {"location", "allowed_uses"},
            {"location", "allowed_uses"},
        )
        if new_data is not None:
            checker.string(new_data.get("location"), "$.assets.new_data.location", nullable=True)
            checker.string_list(new_data.get("allowed_uses"), "$.assets.new_data.allowed_uses", allowed=DATA_USES)

    requirements = checker.object(
        root.get("requirements"),
        "$.requirements",
        {"capabilities_to_preserve", "success_expectation"},
        {"capabilities_to_preserve", "success_expectation"},
    )
    if requirements is not None:
        preservation = checker.object(
            requirements.get("capabilities_to_preserve"),
            "$.requirements.capabilities_to_preserve",
            {"mode", "items"},
            {"mode", "items"},
        )
        if preservation is not None:
            checker.enum(preservation.get("mode"), "$.requirements.capabilities_to_preserve.mode", PRESERVATION_MODES)
            checker.string_list(preservation.get("items"), "$.requirements.capabilities_to_preserve.items")
        checker.string(requirements.get("success_expectation"), "$.requirements.success_expectation", nullable=True)

    intake = checker.object(
        root.get("intake"),
        "$.intake",
        {"required_questions", "optional_questions", "human_confirmation"},
        {"required_questions", "optional_questions", "human_confirmation"},
    )
    if intake is not None:
        _validate_required_questions(checker, intake.get("required_questions"))
        _validate_optional_questions(checker, intake.get("optional_questions"))
        confirmation = checker.object(
            intake.get("human_confirmation"),
            "$.intake.human_confirmation",
            {"status", "confirmed_task_input_version", "notes"},
            {"status", "confirmed_task_input_version", "notes"},
        )
        if confirmation is not None:
            checker.enum(confirmation.get("status"), "$.intake.human_confirmation.status", CONFIRMATION_STATUSES)
            confirmed_version = confirmation.get("confirmed_task_input_version")
            if confirmed_version is not None:
                checker.integer(confirmed_version, "$.intake.human_confirmation.confirmed_task_input_version")
            checker.string(confirmation.get("notes"), "$.intake.human_confirmation.notes", nullable=True)

    return checker.issues


def _validate_required_questions(checker: Checker, questions: Any) -> None:
    path = "$.intake.required_questions"
    if not isinstance(questions, list):
        checker.add("TYPE", path, "应为数组。", "没有必答问题时使用空数组。")
        return
    ids: set[str] = set()
    targets_seen: set[tuple[str, ...]] = set()
    allowed_fields = {field for target in REQUIRED_TARGET_GROUPS for field in target}
    for index, item in enumerate(questions):
        item_path = f"{path}[{index}]"
        question = checker.object(
            item,
            item_path,
            {"id", "target_fields", "question", "status", "answer"},
            {"id", "target_fields", "question", "status", "answer"},
        )
        if question is None:
            continue
        checker.string(question.get("id"), f"{item_path}.id")
        checker.string(question.get("question"), f"{item_path}.question")
        checker.enum(question.get("status"), f"{item_path}.status", REQUIRED_QUESTION_STATUSES)
        checker.string(question.get("answer"), f"{item_path}.answer", nullable=True)
        target_fields = question.get("target_fields")
        if not isinstance(target_fields, list) or not target_fields:
            checker.add("TYPE", f"{item_path}.target_fields", "应为非空字段路径数组。", "按固定问题映射填写 target_fields。")
        else:
            seen: set[str] = set()
            for target_index, field in enumerate(target_fields):
                field_path = f"{item_path}.target_fields[{target_index}]"
                checker.string(field, field_path)
                if isinstance(field, str):
                    if field not in allowed_fields:
                        checker.add("QUESTION_SCOPE", field_path, "该字段不属于 TaskInput 最小必答范围。", "删除此必答问题，必要时改为可选问题。")
                    if field in seen:
                        checker.add("DUPLICATE", field_path, "目标字段重复。", "删除重复字段。")
                    seen.add(field)
            target_key = tuple(target_fields) if all(isinstance(field, str) for field in target_fields) else ()
            if target_key:
                if target_key in targets_seen:
                    checker.add("DUPLICATE", f"{item_path}.target_fields", "同一最小缺口出现了多个必答问题。", "每个固定缺口只保留一个必答问题。")
                targets_seen.add(target_key)
        question_id = question.get("id")
        if isinstance(question_id, str):
            if question_id in ids:
                checker.add("DUPLICATE", f"{item_path}.id", "问题编号重复。", "为每个问题使用唯一编号。")
            ids.add(question_id)
        if question.get("status") == "answered" and not nonempty(question.get("answer")):
            checker.add("STATE", f"{item_path}.answer", "问题已标记 answered，但没有非空答案。", "填写答案，或把状态改为 unanswered。")


def _validate_optional_questions(checker: Checker, questions: Any) -> None:
    path = "$.intake.optional_questions"
    if not isinstance(questions, list):
        checker.add("TYPE", path, "应为数组。", "没有可选问题时使用空数组。")
        return
    if len(questions) > 3:
        checker.add("OPTIONAL_LIMIT", path, f"有 {len(questions)} 个可选问题，超过上限 3。", "只保留最有价值的 3 个问题。")
    ids: set[str] = set()
    for index, item in enumerate(questions):
        item_path = f"{path}[{index}]"
        question = checker.object(
            item,
            item_path,
            {"id", "question", "reason", "status", "answer"},
            {"id", "question", "reason", "status", "answer"},
        )
        if question is None:
            continue
        checker.string(question.get("id"), f"{item_path}.id")
        checker.string(question.get("question"), f"{item_path}.question")
        checker.string(question.get("reason"), f"{item_path}.reason")
        checker.enum(question.get("status"), f"{item_path}.status", OPTIONAL_QUESTION_STATUSES)
        checker.string(question.get("answer"), f"{item_path}.answer", nullable=True)
        question_id = question.get("id")
        if isinstance(question_id, str):
            if question_id in ids:
                checker.add("DUPLICATE", f"{item_path}.id", "问题编号重复。", "为每个问题使用唯一编号。")
            ids.add(question_id)
        if question.get("status") == "answered" and not nonempty(question.get("answer")):
            checker.add("STATE", f"{item_path}.answer", "问题已标记 answered，但没有非空答案。", "填写答案，或改为 unanswered 或 skipped。")


def missing_requirements(data: dict[str, Any]) -> list[RequirementGap]:
    gaps: list[RequirementGap] = []

    if not nonempty(value_at(data, "request", "raw_prompt")):
        gaps.append(RequirementGap(("request.raw_prompt",), "缺少用户原始请求。", "请用户描述本次模型更新请求。"))

    goal_summary = value_at(data, "request", "update_goal", "summary")
    goal_status = value_at(data, "request", "update_goal", "status")
    if not nonempty(goal_summary) or goal_status == "missing":
        gaps.append(RequirementGap(("request.update_goal",), "缺少可理解的更新目标。", "请用户说明希望模型获得或改善什么能力。"))

    if not nonempty(value_at(data, "assets", "target_project", "location")):
        gaps.append(RequirementGap(("assets.target_project.location",), "缺少目标项目位置。", "请用户提供目标项目的位置。"))

    checkpoint = value_at(data, "assets", "base_model", "checkpoint")
    discovery_authorized = value_at(data, "assets", "base_model", "discovery_authorized")
    if not nonempty(checkpoint) and discovery_authorized is not True:
        gaps.append(
            RequirementGap(
                ("assets.base_model.checkpoint", "assets.base_model.discovery_authorized"),
                "既没有原模型 checkpoint，也未授权后续只读发现。",
                "请用户提供 checkpoint，或明确授权后续阶段只读发现。",
            )
        )

    if not nonempty(value_at(data, "assets", "new_data", "location")):
        gaps.append(RequirementGap(("assets.new_data.location",), "缺少新数据位置。", "请用户提供新数据的位置。"))

    allowed_uses = value_at(data, "assets", "new_data", "allowed_uses")
    if not isinstance(allowed_uses, list) or not allowed_uses:
        gaps.append(RequirementGap(("assets.new_data.allowed_uses",), "尚未明确新数据的允许用途。", "请用户确认可用于分析、训练、验证、评估中的哪些用途。"))

    preservation_mode = value_at(data, "requirements", "capabilities_to_preserve", "mode")
    if preservation_mode == "unconfirmed" or preservation_mode not in PRESERVATION_MODES:
        gaps.append(RequirementGap(("requirements.capabilities_to_preserve",), "尚未确认需要保留的能力。", "请用户列出需要保留的能力，或明确回答无。"))

    if not nonempty(value_at(data, "requirements", "success_expectation")):
        gaps.append(RequirementGap(("requirements.success_expectation",), "缺少业务成功期望。", "请用户说明业务上怎样算本次更新成功。"))

    return gaps


REQUIRED_TARGET_GROUPS = {
    ("request.raw_prompt",),
    ("request.update_goal",),
    ("assets.target_project.location",),
    ("assets.base_model.checkpoint", "assets.base_model.discovery_authorized"),
    ("assets.new_data.location",),
    ("assets.new_data.allowed_uses",),
    ("requirements.capabilities_to_preserve",),
    ("requirements.success_expectation",),
}


def validate_semantics(data: dict[str, Any], *, mode: str = "draft") -> list[ValidationIssue]:
    checker = Checker()
    gaps = missing_requirements(data)
    gaps_by_target = {gap.target_fields: gap for gap in gaps}
    questions = value_at(data, "intake", "required_questions")
    questions = questions if isinstance(questions, list) else []
    question_by_target: dict[tuple[str, ...], tuple[int, dict[str, Any]]] = {}

    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            continue
        targets = question.get("target_fields")
        if not isinstance(targets, list) or not all(isinstance(item, str) for item in targets):
            continue
        target_key = tuple(targets)
        if target_key not in REQUIRED_TARGET_GROUPS:
            checker.add(
                "QUESTION_MAPPING",
                f"$.intake.required_questions[{index}].target_fields",
                "必答问题没有使用固定最小字段映射，或字段顺序不正确。",
                "按任务受理规格第 4 节使用完整且精确的 target_fields；其他问题移至 optional_questions。",
            )
            continue
        question_by_target[target_key] = (index, question)

    for target_key, gap in gaps_by_target.items():
        entry = question_by_target.get(target_key)
        if entry is None:
            checker.add("MISSING_QUESTION", "$.intake.required_questions", gap.problem, gap.fix)
            continue
        index, question = entry
        if question.get("status") != "unanswered":
            checker.add(
                "UNSYNCED_ANSWER",
                f"$.intake.required_questions[{index}]",
                "问题已标记为已回答，但对应最小字段仍然缺失。",
                "把答案同步到 target_fields，或将问题恢复为 unanswered。",
            )

    for target_key, (index, question) in question_by_target.items():
        if target_key not in gaps_by_target and question.get("status") == "unanswered":
            checker.add(
                "STALE_QUESTION",
                f"$.intake.required_questions[{index}]",
                "对应最小字段已经满足，但问题仍标记为未回答。",
                "将已获得的答案记录为 answered，或删除不再需要的问题。",
            )

    preservation_mode = value_at(data, "requirements", "capabilities_to_preserve", "mode")
    preservation_items = value_at(data, "requirements", "capabilities_to_preserve", "items")
    if preservation_mode == "specified" and (not isinstance(preservation_items, list) or not preservation_items):
        checker.add(
            "STATE",
            "$.requirements.capabilities_to_preserve.items",
            "mode 为 specified 时必须列出至少一项需要保留的能力。",
            "补充能力列表，或把 mode 改为 unconfirmed 或 none。",
        )
    if preservation_mode == "none" and isinstance(preservation_items, list) and preservation_items:
        checker.add(
            "STATE",
            "$.requirements.capabilities_to_preserve.items",
            "mode 为 none 时能力列表必须为空。",
            "清空 items，或把 mode 改为 specified。",
        )

    if mode == "confirmed":
        _validate_confirmation_gate(checker, data, gaps, questions)
    elif mode != "draft":
        checker.add("MODE", "$", f"未知校验模式 {mode!r}。", "使用 draft 或 confirmed。")

    return checker.issues


def _validate_confirmation_gate(
    checker: Checker,
    data: dict[str, Any],
    gaps: list[RequirementGap],
    questions: list[Any],
) -> None:
    for gap in gaps:
        checker.add("CONFIRMATION_GATE", ".".join(("$",) + gap.target_fields), gap.problem, gap.fix)

    if value_at(data, "request", "update_goal", "status") != "confirmed":
        checker.add("CONFIRMATION_GATE", "$.request.update_goal.status", "更新目标尚未由人类确认。", "请人类核对目标，并将状态改为 confirmed。")

    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            continue
        if question.get("status") != "answered" or not nonempty(question.get("answer")):
            checker.add(
                "CONFIRMATION_GATE",
                f"$.intake.required_questions[{index}]",
                "存在尚未回答的必答问题。",
                "获得人类答复、同步正式字段，并把问题标为 answered。",
            )

    confirmation = value_at(data, "intake", "human_confirmation")
    if not isinstance(confirmation, dict):
        checker.add("CONFIRMATION_GATE", "$.intake.human_confirmation", "缺少人工确认记录。", "为当前版本记录人工确认。")
        return
    if confirmation.get("status") != "confirmed":
        checker.add("CONFIRMATION_GATE", "$.intake.human_confirmation.status", "当前版本尚未人工确认。", "人类确认后将状态设为 confirmed。")
    if confirmation.get("confirmed_task_input_version") != data.get("task_input_version"):
        checker.add(
            "CONFIRMATION_GATE",
            "$.intake.human_confirmation.confirmed_task_input_version",
            "人工确认版本与当前 TaskInput 版本不一致。",
            "重新确认当前版本，并记录相同的版本号。",
        )
def validate_confirmed(data: dict[str, Any]) -> list[ValidationIssue]:
    """Backward-compatible entry point for callers that check the confirmation gate."""
    return validate_semantics(data, mode="confirmed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验 TaskInput v0.2 JSON 文件。")
    parser.add_argument("input", nargs="?", type=Path, help="TaskInput JSON 文件路径")
    parser.add_argument(
        "--stdin",
        action="store_true",
        dest="read_stdin",
        help="从标准输入读取 TaskInput JSON；用于 Harness 工具适配。",
    )
    parser.add_argument(
        "--mode",
        choices=("draft", "confirmed"),
        default="draft",
        help="draft 检查结构与最小问题映射；confirmed 额外检查进入策略规划的门槛",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="输出人类可读文本或机器可读 JSON。",
    )
    args = parser.parse_args()
    if args.read_stdin == (args.input is not None):
        parser.error("必须且只能选择一个输入来源：文件路径或 --stdin。")
    return args


def result_payload(
    data: Any,
    mode: str,
    issues: list[ValidationIssue],
    *,
    execution_error: bool = False,
) -> dict[str, Any]:
    task_id = data.get("task_id") if isinstance(data, dict) else None
    task_input_version = data.get("task_input_version") if isinstance(data, dict) else None
    return {
        "valid": not issues and not execution_error,
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "task_id": task_id if isinstance(task_id, str) else None,
        "task_input_version": (
            task_input_version
            if isinstance(task_input_version, int) and not isinstance(task_input_version, bool)
            else None
        ),
        "issues": [
            {
                "code": issue.code,
                "path": issue.path,
                "problem": issue.problem,
                "fix": issue.fix,
            }
            for issue in issues
        ],
        **({"execution_error": True} if execution_error else {}),
    }


def emit_input_error(args: argparse.Namespace, code: str, problem: str, fix: str) -> int:
    issue = ValidationIssue(code, "$", problem, fix)
    if args.output == "json":
        print(
            json.dumps(
                result_payload(None, args.mode, [issue], execution_error=True),
                ensure_ascii=False,
            )
        )
    else:
        print(issue.render(), file=sys.stderr)
    return 2


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    args = parse_args()
    try:
        if args.read_stdin:
            data = json.load(sys.stdin)
        else:
            with args.input.open("r", encoding="utf-8") as stream:
                data = json.load(stream)
    except FileNotFoundError:
        return emit_input_error(
            args,
            "INPUT_NOT_FOUND",
            f"找不到文件：{args.input}",
            "提供存在的 TaskInput JSON 路径。",
        )
    except (OSError, json.JSONDecodeError) as exc:
        return emit_input_error(
            args,
            "INVALID_JSON",
            f"无法读取 TaskInput：{exc}",
            "确认输入是 UTF-8 编码的有效 JSON。",
        )

    issues = validate_shape(data)
    if not issues:
        issues.extend(validate_semantics(data, mode=args.mode))

    if args.output == "json":
        print(
            json.dumps(
                result_payload(data, args.mode, issues),
                ensure_ascii=False,
            )
        )
        return 1 if issues else 0

    if issues:
        print(f"TaskInput 校验失败，共 {len(issues)} 个问题：", file=sys.stderr)
        for issue in issues:
            print(issue.render(), file=sys.stderr)
        return 1

    print(f"TaskInput 校验通过：mode={args.mode}, task_id={data['task_id']}, version={data['task_input_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
