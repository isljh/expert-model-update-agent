"""Revalidate and load a declared model delivery in a local Python process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

try:
    from .validate import validate_model_delivery
    from .record import RecordWriteError, save_model_analysis_record
except ImportError:
    from validate import validate_model_delivery
    from record import RecordWriteError, save_model_analysis_record


REPORT_TYPE = "ModelInstantiationReport"
REPORT_VERSION = "0.2"
MAX_WORKER_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_PLAN_BYTES = 4 * 1024 * 1024
WORKER_TIMEOUT_SECONDS = 120
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RECORD_ROOT = PROJECT_ROOT / "run-records" / "model-structure"


def _blocked(code: str, message: str, *, load_spec_id: str | None = None) -> dict[str, Any]:
    return {
        "report_type": REPORT_TYPE,
        "schema_version": REPORT_VERSION,
        "load_spec_id": load_spec_id,
        "status": "blocked",
        "steps": [],
        "checkpoint": {"status": "not_attempted"},
        "model": None,
        "structure_analysis": None,
        "error": {"code": code, "message": message},
    }


def _prepare_plan(payload: Any, base_directory: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(payload, dict) or set(payload) != {"delivery_inputs", "validation_report"}:
        return None, _blocked("INPUT_CONTRACT_INVALID", "必须提供 delivery_inputs 与对应 validation_report。")
    supplied = payload["validation_report"]
    if not isinstance(supplied, dict):
        return None, _blocked("VALIDATION_REPORT_INVALID", "validation_report 必须是对象。")
    try:
        current = validate_model_delivery(payload["delivery_inputs"], base_directory=base_directory)
    except (OSError, RuntimeError, ValueError) as exc:
        return None, _blocked("REVALIDATION_FAILED", f"重新核验模型交付物失败：{type(exc).__name__}: {exc}")
    if not current["readiness"]["ready_for_model_load"]:
        return None, _blocked("MODEL_LOAD_NOT_READY", "当前模型交付物不满足加载条件。")
    plan = current["validated_load_plan"]
    assert isinstance(plan, dict)
    if (
        supplied.get("report_type") != "ModelDeliveryValidationReport"
        or supplied.get("schema_version") != "0.1"
        or supplied.get("task_binding") != current.get("task_binding")
        or supplied.get("status") != current.get("status")
        or supplied.get("validated_load_plan") != plan
        or supplied.get("readiness") != current.get("readiness")
    ):
        return None, _blocked("VALIDATION_REPORT_MISMATCH", "传入的验证报告与当前重新核验结果不一致。", load_spec_id=plan["load_spec_id"])
    return plan, None


def _run_bounded(argv: list[str], plan_bytes: bytes, *, cwd: str, env: dict[str, str]) -> tuple[int, bytes, bool]:
    process = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    output = bytearray()
    overflow = threading.Event()
    stderr_bytes = 0

    def read_stream(stream: Any, *, keep: bool) -> None:
        nonlocal stderr_bytes
        try:
            while chunk := stream.read(65536):
                if keep:
                    if len(output) + len(chunk) > MAX_WORKER_OUTPUT_BYTES:
                        overflow.set()
                    else:
                        output.extend(chunk)
                else:
                    stderr_bytes += len(chunk)
                    if stderr_bytes > MAX_WORKER_OUTPUT_BYTES:
                        overflow.set()
                if overflow.is_set():
                    process.kill()
        except OSError:
            pass

    def write_input() -> None:
        try:
            process.stdin.write(plan_bytes)
            process.stdin.close()
        except OSError:
            pass

    threads = [
        threading.Thread(target=read_stream, args=(process.stdout,), kwargs={"keep": True}, daemon=True),
        threading.Thread(target=read_stream, args=(process.stderr,), kwargs={"keep": False}, daemon=True),
        threading.Thread(target=write_input, daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        returncode = process.wait(timeout=WORKER_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    finally:
        for thread in threads:
            thread.join(timeout=2)
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                stream.close()
            except OSError:
                pass
    return returncode, bytes(output), overflow.is_set()


def _valid_worker_report(report: Any, plan: dict[str, Any], *, analyze_structure: bool = False) -> bool:
    if not isinstance(report, dict) or report.get("report_type") != REPORT_TYPE or report.get("schema_version") != REPORT_VERSION:
        return False
    if report.get("load_spec_id") != plan["load_spec_id"] or report.get("status") not in {"completed", "failed"}:
        return False
    checkpoint = report.get("checkpoint")
    if not isinstance(report.get("steps"), list) or not isinstance(checkpoint, dict):
        return False
    if checkpoint.get("status") not in {"loaded", "failed", "not_attempted"}:
        return False
    if report["status"] == "completed":
        model = report.get("model")
        structure = report.get("structure_analysis")
        return (
            isinstance(model, dict)
            and model.get("target_is_torch_module") is True
            and isinstance(model.get("has_trainable_parameters"), bool)
            and report.get("error") is None
            and checkpoint.get("status") == "loaded"
            and checkpoint.get("weights_only") is True
            and checkpoint.get("strict") is True
            and checkpoint.get("fully_matched") is True
            and checkpoint.get("missing_keys") == []
            and checkpoint.get("unexpected_keys") == []
            and (
                not analyze_structure
                or (
                    isinstance(structure, dict)
                    and structure.get("report_type") == "ModelStructureAnalysisReport"
                    and structure.get("schema_version") == "0.2"
                    and structure.get("status") == "passed"
                    and structure.get("load_spec_id") == plan["load_spec_id"]
                    and structure.get("source_model_state", {}).get("checkpoint_fully_matched") is True
                )
            )
        )
    error = report.get("error")
    return isinstance(error, dict) and isinstance(error.get("code"), str) and isinstance(error.get("message"), str)


def load_validated_model(
    payload: Any,
    *,
    base_directory: Path | None = None,
    analyze_structure: bool = False,
    record_root: Path | None = None,
) -> dict[str, Any]:
    base = (base_directory or Path.cwd()).resolve(strict=True)
    plan, error = _prepare_plan(payload, base)
    if error is not None:
        return error
    assert plan is not None
    delivery_root = Path(plan["model"]["path"]).resolve(strict=True).parent
    destination = record_root if record_root is not None else DEFAULT_RECORD_ROOT
    if analyze_structure:
        resolved_destination = destination.resolve(strict=False)
        if resolved_destination == delivery_root or delivery_root in resolved_destination.parents:
            return _blocked("RECORD_ROOT_IN_MODEL_DELIVERY", "运行记录目录位于模型交付目录内，拒绝执行。", load_spec_id=plan["load_spec_id"])
        if record_root is None:
            workspace = PROJECT_ROOT.resolve(strict=True)
            if workspace not in resolved_destination.parents:
                return _blocked("RECORD_ROOT_OUTSIDE_WORKSPACE", "运行记录目录未解析到智能体项目内，拒绝执行。", load_spec_id=plan["load_spec_id"])
    worker = Path(__file__).with_name("instantiate_worker.py").resolve(strict=True)
    argv = [sys.executable, "-B", "-X", "utf8", str(worker)]
    if analyze_structure:
        argv.append("--analyze-structure")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA"}
    }
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    plan_bytes = json.dumps(plan, ensure_ascii=False).encode("utf-8")
    if len(plan_bytes) > MAX_PLAN_BYTES:
        return _blocked("PLAN_TOO_LARGE", "模型加载计划超过 4 MiB。", load_spec_id=plan["load_spec_id"])
    try:
        with tempfile.TemporaryDirectory(prefix="model-loading-") as scratch:
            returncode, output, overflow = _run_bounded(argv, plan_bytes, cwd=scratch, env=environment)
    except subprocess.TimeoutExpired:
        return _blocked("WORKER_EXECUTION_TIMEOUT", "本地模型加载超过 120 秒；子进程已终止。", load_spec_id=plan["load_spec_id"])
    except OSError as exc:
        return _blocked("WORKER_START_FAILED", f"无法启动本地模型加载进程：{exc}", load_spec_id=plan["load_spec_id"])
    if overflow:
        return _blocked("WORKER_OUTPUT_TOO_LARGE", "模型加载进程输出超过 4 MiB。", load_spec_id=plan["load_spec_id"])
    try:
        report = json.loads(output.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _blocked("WORKER_OUTPUT_INVALID", "模型加载进程没有返回有效 JSON。", load_spec_id=plan["load_spec_id"])
    if not _valid_worker_report(report, plan, analyze_structure=analyze_structure):
        return _blocked("WORKER_OUTPUT_INVALID", "模型加载进程返回的报告不符合协议。", load_spec_id=plan["load_spec_id"])
    if (returncode == 0) != (report["status"] == "completed"):
        return _blocked("WORKER_EXIT_STATUS_MISMATCH", "模型加载进程退出码与报告状态不一致。", load_spec_id=plan["load_spec_id"])
    report["execution_boundary"] = {
        "kind": "local_subprocess_without_os_isolation",
        "model_code_imported_in_host": False,
        "model_code_may_have_executed_in_worker": True,
        "model_assets_read_only_guaranteed": False,
        "network_blocked_guaranteed": False,
        "child_process_cleanup_guaranteed": False,
    }
    if analyze_structure:
        report["task_binding"] = payload["validation_report"]["task_binding"]
        report["delivery"] = {
            "model_file": plan["model"]["path"],
            "config": plan["config"]["path"],
            "checkpoint": plan["checkpoint"]["path"],
        }
        try:
            return save_model_analysis_record(
                report,
                record_root=destination,
                model_delivery_root=delivery_root,
            )
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = {
                "code": exc.code if isinstance(exc, RecordWriteError) else "REPORT_WRITE_FAILED",
                "message": f"保存模型分析报告失败：{type(exc).__name__}: {exc}",
            }
    return report


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="在本地子进程加载已验证模型交付物；不提供操作系统级隔离。")
    parser.add_argument("--stdin", action="store_true", required=True)
    parser.add_argument("--analyze-structure", action="store_true")
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        report = load_validated_model(payload, analyze_structure=args.analyze_structure)
    except Exception as exc:
        report = _blocked("HOST_EXECUTION_FAILED", f"{type(exc).__name__}: {exc}")
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
