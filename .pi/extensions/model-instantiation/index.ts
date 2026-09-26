import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

interface InstantiationReport {
	report_type: "ModelInstantiationReport";
	schema_version: "0.1";
	status: "completed" | "failed" | "blocked";
	proposal_id: string | null;
	checkpoint: { status: string; tensor_count?: number; strict?: boolean; missing_keys?: string[]; unexpected_keys?: string[] };
	model: unknown;
	project?: { source_project_path: string; source_revision?: string | null };
	structure_analysis?: {
		report_type: "ModelStructureAnalysisReport";
		schema_version: "0.1";
		status: "passed";
		proposal_id: string;
		general?: { model_family?: string };
		parameter_summary?: { total_parameter_count?: number };
		module_tree?: unknown[];
		buffers?: unknown[];
		forward_trace?: { status?: string };
	} | null;
	record?: { run_id: string; recorded_at_utc: string; project_name: string; json_report_path: string; structure_tree_path: string };
	execution_boundary?: unknown;
	error: { code: string; message: string } | null;
}

const EXTENSION_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(EXTENSION_DIRECTORY, "..", "..", "..");
const HOST_PATH = resolve(
	PROJECT_ROOT,
	"src", "expert_model_update", "tools", "model_creation", "instantiate.py",
);
const MAX_INPUT_BYTES = 4_000_000;
const MAX_OUTPUT_BYTES = 5_000_000;
const TOOL_TIMEOUT_MS = 130_000;
const WINDOWS_MODEL_INSTANTIATION_PYTHON = "D:\\software\\envs\\model-instantiation-safe\\python.exe";

function pythonExecutable(): string {
	const configured = process.env.MODEL_INSTANTIATION_PYTHON?.trim();
	if (configured) return configured;
	if (process.platform === "win32") {
		if (!existsSync(WINDOWS_MODEL_INSTANTIATION_PYTHON)) {
			throw new Error("未找到已选定的 model-instantiation-safe 环境：" + WINDOWS_MODEL_INSTANTIATION_PYTHON);
		}
		return WINDOWS_MODEL_INSTANTIATION_PYTHON;
	}
	return "python3";
}

function isReport(value: unknown): value is InstantiationReport {
	if (!value || typeof value !== "object") return false;
	const item = value as Partial<InstantiationReport>;
	return item.report_type === "ModelInstantiationReport" &&
		item.schema_version === "0.1" &&
		(item.status === "completed" || item.status === "failed" || item.status === "blocked") &&
		Boolean(item.checkpoint && typeof item.checkpoint.status === "string");
}

async function runInstantiation(input: unknown, signal?: AbortSignal, analyzeStructure = false): Promise<InstantiationReport> {
	if (!existsSync(HOST_PATH)) throw new Error("模型实例化工具不存在：" + HOST_PATH);
	const serialized = JSON.stringify(input);
	if (typeof serialized !== "string" || Buffer.byteLength(serialized, "utf8") > MAX_INPUT_BYTES) {
		throw new Error("实例化输入必须是小于 4 MB 的 JSON 对象。");
	}
	if (signal?.aborted) throw new Error("模型实例化已取消。");
	return new Promise((resolvePromise, rejectPromise) => {
		const child = spawn(
			pythonExecutable(),
			["-B", "-X", "utf8", HOST_PATH, "--stdin", ...(analyzeStructure ? ["--analyze-structure"] : [])],
			{ cwd: PROJECT_ROOT, stdio: ["pipe", "pipe", "pipe"], windowsHide: true },
		);
		let stdout = "";
		let stderr = "";
		let settled = false;
		let timedOut = false;
		let outputTooLarge = false;
		const finishError = (message: string) => {
			if (settled) return;
			settled = true;
			cleanup();
			rejectPromise(new Error(message));
		};
		const onAbort = () => {
			child.kill();
			finishError("模型实例化已取消。");
		};
		const timer = setTimeout(() => {
			timedOut = true;
			child.kill();
		}, TOOL_TIMEOUT_MS);
		const cleanup = () => {
			clearTimeout(timer);
			signal?.removeEventListener("abort", onAbort);
		};
		signal?.addEventListener("abort", onAbort, { once: true });
		child.stdout.on("data", (chunk: Buffer) => {
			stdout += chunk.toString("utf8");
			if (Buffer.byteLength(stdout, "utf8") > MAX_OUTPUT_BYTES) {
				outputTooLarge = true;
				child.kill();
			}
		});
		child.stderr.on("data", (chunk: Buffer) => {
			stderr += chunk.toString("utf8");
			if (Buffer.byteLength(stderr, "utf8") > MAX_OUTPUT_BYTES) {
				outputTooLarge = true;
				child.kill();
			}
		});
		child.once("error", error => finishError("无法启动实例化工具：" + error.message));
		child.once("close", code => {
			if (settled) return;
			settled = true;
			cleanup();
			if (timedOut) return rejectPromise(new Error("模型实例化超过 130 秒。"));
			if (outputTooLarge) return rejectPromise(new Error("实例化工具输出超过大小上限。"));
			let parsed: unknown;
			try { parsed = JSON.parse(stdout.trim()); }
			catch { return rejectPromise(new Error("实例化工具未返回有效 JSON：" + (stderr.trim() || String(code)))); }
			if (!isReport(parsed)) return rejectPromise(new Error("实例化工具报告结构不符合约定。"));
			if (analyzeStructure && parsed.status === "completed" &&
				(!parsed.structure_analysis || parsed.structure_analysis.report_type !== "ModelStructureAnalysisReport" ||
				parsed.structure_analysis.schema_version !== "0.1" || parsed.structure_analysis.status !== "passed" ||
				parsed.structure_analysis.proposal_id !== parsed.proposal_id ||
				!parsed.record || !parsed.record.json_report_path || !parsed.record.structure_tree_path)) {
				return rejectPromise(new Error("结构解析工具未返回有效结构报告。"));
			}
			if (code !== 0 && code !== 1) return rejectPromise(new Error("实例化工具异常退出：" + String(code)));
			if ((code === 0) !== (parsed.status === "completed")) {
				return rejectPromise(new Error("实例化工具退出码与报告状态不一致。"));
			}
			resolvePromise(parsed);
		});
		child.stdin.once("error", error => finishError("无法传入实例化计划：" + error.message));
		child.stdin.end(serialized, "utf8");
	});
}

const instantiateModelTool = defineTool({
	name: "instantiate_validated_model",
	label: "Instantiate Validated Model",
	description: "重新核对模型创建方案后在本地 Python 子进程导入并调用目标项目代码；无操作系统级隔离，项目代码可能产生写入、网络或其他副作用。",
	promptSnippet: "Execute only a revalidated model creation plan in a local subprocess without OS isolation",
	promptGuidelines: [
		"Use only after validate_model_creation_spec selected exactly one proposal and returned ready_for_model_construction=true.",
		"Pass the original ModelCreationProposalSet and matching ModelCreationValidationReport unchanged; this tool revalidates and compares them before execution.",
		"This tool executes target project Python code locally without OS isolation; do not describe its subprocess as a sandbox or claim the target project is read-only.",
		"A blocked result is not a model instance; a completed result with checkpoint=loaded is required before claiming original weights were restored.",
	],
	parameters: Type.Object({
		proposal_set: Type.Any({ description: "已经通过静态验证的原始 ModelCreationProposalSet v0.2。" }),
		validation_report: Type.Any({ description: "对应的 ModelCreationValidationReport v0.2 完整输出。" }),
	}, { additionalProperties: false }),
	async execute(_toolCallId, params, signal) {
		const result = await runInstantiation(params, signal);
		return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }], details: result };
	},
});

const analyzeModelStructureTool = defineTool({
	name: "analyze_validated_model_structure",
	label: "Analyze Validated Model Structure",
	description: "重新核验创建方案，在同一本地 Python 子进程中创建模型、加载声明权重并解析内存中的模型结构；不执行 forward，无操作系统级隔离。",
	promptSnippet: "Construct a revalidated model and inspect its live structure in the same local worker",
	promptGuidelines: [
		"Use only after validate_model_creation_spec selected exactly one proposal and returned ready_for_model_construction=true.",
		"Pass the original proposal set and matching validation report unchanged; this tool revalidates both before project execution.",
		"This tool creates the model as part of the same call. Do not call instantiate_validated_model first just to obtain an instance: that instance exits with its worker.",
		"The host saves the complete JSON report and fully expanded readable module tree outside the target project. Use the returned record paths as authoritative artifacts; read the JSON file if downstream reasoning needs full details.",
		"Do not claim a forward trace, successful original-weight restoration, or OS isolation unless the report explicitly proves it.",
	],
	parameters: Type.Object({
		proposal_set: Type.Any({ description: "已经通过静态验证的原始 ModelCreationProposalSet v0.2。" }),
		validation_report: Type.Any({ description: "对应的 ModelCreationValidationReport v0.2 完整输出。" }),
	}, { additionalProperties: false }),
	async execute(_toolCallId, params, signal) {
		const result = await runInstantiation(params, signal, true);
		const structure = result.structure_analysis;
		const summary = {
			report_type: result.report_type,
			status: result.status,
			proposal_id: result.proposal_id,
			project: result.project?.source_project_path ?? null,
			checkpoint: {
				status: result.checkpoint.status,
				strict: result.checkpoint.strict,
				tensor_count: result.checkpoint.tensor_count,
				missing_key_count: result.checkpoint.missing_keys?.length,
				unexpected_key_count: result.checkpoint.unexpected_keys?.length,
			},
			structure_analysis: structure ? {
				status: structure.status,
				model_family: structure.general?.model_family,
				module_record_count: structure.module_tree?.length,
				parameter_count: structure.parameter_summary?.total_parameter_count,
				buffer_record_count: structure.buffers?.length,
				forward_trace_status: structure.forward_trace?.status,
			} : null,
			record: result.record ?? null,
			execution_boundary: result.execution_boundary,
			error: result.error,
		};
		return { content: [{ type: "text", text: JSON.stringify(summary, null, 2) }], details: summary };
	},
});

export default function modelInstantiationExtension(pi: ExtensionAPI) {
	pi.registerTool(instantiateModelTool);
	pi.registerTool(analyzeModelStructureTool);
}
