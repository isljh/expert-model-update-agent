import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

interface InstantiationReport {
	report_type: "ModelInstantiationReport";
	schema_version: "0.2";
	status: "completed" | "failed" | "blocked";
	load_spec_id: string | null;
	checkpoint: {
		status: string;
		tensor_count?: number;
		strict?: boolean;
		fully_matched?: boolean;
		missing_keys?: string[];
		unexpected_keys?: string[];
	};
	model: {
		class_name?: string;
		parameter_count?: number;
		trainable_parameter_count?: number;
		has_trainable_parameters?: boolean;
	} | null;
	delivery?: { model_file: string; config: string; checkpoint: string };
	structure_analysis?: {
		report_type: "ModelStructureAnalysisReport";
		schema_version: "0.2";
		status: "passed";
		load_spec_id: string;
		general?: { model_family?: string };
		parameter_summary?: { total_parameter_count?: number; trainable_parameter_count?: number };
		module_tree?: unknown[];
		buffers?: unknown[];
		forward_trace?: { status?: string };
	} | null;
	record?: { run_id: string; json_report_path: string; structure_tree_path: string };
	execution_boundary?: unknown;
	error: { code: string; message: string } | null;
}

const EXTENSION_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(EXTENSION_DIRECTORY, "..", "..", "..");
const HOST_PATH = resolve(PROJECT_ROOT, "src", "expert_model_update", "tools", "model_creation", "instantiate.py");
const MAX_INPUT_BYTES = 8_000_000;
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
		item.schema_version === "0.2" &&
		(item.status === "completed" || item.status === "failed" || item.status === "blocked") &&
		Boolean(item.checkpoint && typeof item.checkpoint.status === "string");
}

async function runInstantiation(input: unknown, signal?: AbortSignal, analyzeStructure = false): Promise<InstantiationReport> {
	if (!existsSync(HOST_PATH)) throw new Error("模型加载工具不存在：" + HOST_PATH);
	const serialized = JSON.stringify(input);
	if (typeof serialized !== "string" || Buffer.byteLength(serialized, "utf8") > MAX_INPUT_BYTES) {
		throw new Error("模型加载输入必须是小于 8 MB 的 JSON 对象。");
	}
	if (signal?.aborted) throw new Error("模型加载已取消。");
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
			finishError("模型加载已取消。");
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
		child.once("error", error => finishError("无法启动模型加载工具：" + error.message));
		child.once("close", code => {
			if (settled) return;
			settled = true;
			cleanup();
			if (timedOut) return rejectPromise(new Error("模型加载超过 130 秒。"));
			if (outputTooLarge) return rejectPromise(new Error("模型加载工具输出超过大小上限。"));
			let parsed: unknown;
			try { parsed = JSON.parse(stdout.trim()); }
			catch { return rejectPromise(new Error("模型加载工具未返回有效 JSON：" + (stderr.trim() || String(code)))); }
			if (!isReport(parsed)) return rejectPromise(new Error("模型加载报告结构不符合约定。"));
			if (analyzeStructure && parsed.status === "completed" &&
				(!parsed.structure_analysis || parsed.structure_analysis.report_type !== "ModelStructureAnalysisReport" ||
					parsed.structure_analysis.schema_version !== "0.2" || parsed.structure_analysis.status !== "passed" ||
					parsed.structure_analysis.load_spec_id !== parsed.load_spec_id ||
					!parsed.record?.json_report_path || !parsed.record?.structure_tree_path)) {
				return rejectPromise(new Error("结构分析工具未返回有效结构报告。"));
			}
			if (code !== 0 && code !== 1) return rejectPromise(new Error("模型加载工具异常退出：" + String(code)));
			if ((code === 0) !== (parsed.status === "completed")) {
				return rejectPromise(new Error("模型加载工具退出码与报告状态不一致。"));
			}
			resolvePromise(parsed);
		});
		child.stdin.once("error", error => finishError("无法传入模型加载计划：" + error.message));
		child.stdin.end(serialized, "utf8");
	});
}

const parameters = Type.Object({
	delivery_inputs: Type.Any({ description: "传给 validate_model_delivery 的四项原始输入。" }),
	validation_report: Type.Any({ description: "对应的 ModelDeliveryValidationReport v0.1 完整输出。" }),
}, { additionalProperties: false });

const loadModelTool = defineTool({
	name: "load_validated_model",
	label: "Load Validated Model",
	description: "重新验证声明的模型、配置和 checkpoint 后，在本地 Python 子进程导入模型类、用声明参数创建实例并严格加载权重；无操作系统级隔离。",
	promptSnippet: "Load one validated model delivery and require a fully matched checkpoint",
	promptGuidelines: [
		"Use only after validate_model_delivery returned ready_for_model_load=true.",
		"Pass the original delivery inputs and validation report unchanged; this tool revalidates both before execution.",
		"Do not describe the subprocess as a sandbox. Importing the supplied model file and calling its constructor may have side effects.",
		"Do not claim restoration unless status=completed, checkpoint.status=loaded, strict=true, and fully_matched=true.",
	],
	parameters,
	async execute(_toolCallId, params, signal) {
		const result = await runInstantiation(params, signal);
		return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }], details: result };
	},
});

const analyzeModelStructureTool = defineTool({
	name: "analyze_validated_model_structure",
	label: "Analyze Validated Model Structure",
	description: "在同一本地子进程中创建声明模型、严格加载 checkpoint，并分析真实内存模型的结构和可训练参数；不执行 forward，无操作系统级隔离。",
	promptSnippet: "Load a validated model and inspect structure and trainable parameters in one worker",
	promptGuidelines: [
		"Use only after validate_model_delivery returned ready_for_structure_analysis=true.",
		"Do not call load_validated_model first merely to obtain an instance; that instance exits with its worker.",
		"Use the saved JSON and structure tree paths as authoritative runtime evidence.",
		"Do not claim forward compatibility or operating-system isolation; forward is not executed.",
	],
	parameters,
	async execute(_toolCallId, params, signal) {
		const result = await runInstantiation(params, signal, true);
		const structure = result.structure_analysis;
		const summary = {
			report_type: result.report_type,
			status: result.status,
			load_spec_id: result.load_spec_id,
			delivery: result.delivery ?? null,
			checkpoint: {
				status: result.checkpoint.status,
				strict: result.checkpoint.strict,
				fully_matched: result.checkpoint.fully_matched,
				tensor_count: result.checkpoint.tensor_count,
				missing_key_count: result.checkpoint.missing_keys?.length,
				unexpected_key_count: result.checkpoint.unexpected_keys?.length,
			},
			model: result.model,
			structure_analysis: structure ? {
				status: structure.status,
				model_family: structure.general?.model_family,
				module_record_count: structure.module_tree?.length,
				parameter_count: structure.parameter_summary?.total_parameter_count,
				trainable_parameter_count: structure.parameter_summary?.trainable_parameter_count,
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
	pi.registerTool(loadModelTool);
	pi.registerTool(analyzeModelStructureTool);
}
