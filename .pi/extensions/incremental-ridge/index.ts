import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

interface ToolReport {
	report_type: "IncrementalRidgeToolReport";
	schema_version: "0.3";
	method_id: "incremental_ridge_closed_form";
	method_version: "0.3";
	action: "initialize" | "inspect" | "update";
	status: "passed" | "completed" | "rejected" | "failed";
	issues: Array<{ code: string; path: string; message: string }>;
	state_sha256: string | null;
	state: unknown;
	summary: unknown;
}

const EXTENSION_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(EXTENSION_DIRECTORY, "..", "..", "..");
const CLI_PATH = resolve(
	PROJECT_ROOT,
	"src", "expert_model_update", "methods", "incremental_ridge", "cli.py",
);
const MAX_INPUT_BYTES = 4_000_000;
const MAX_OUTPUT_BYTES = 8_000_000;
const TOOL_TIMEOUT_MS = 60_000;

function pythonExecutable(): string {
	const configured = process.env.INCREMENTAL_METHODS_PYTHON?.trim();
	if (configured) return configured;
	return process.platform === "win32" ? "python" : "python3";
}

function isReport(value: unknown, action: "initialize" | "inspect" | "update"): value is ToolReport {
	if (!value || typeof value !== "object") return false;
	const item = value as Partial<ToolReport>;
	return item.report_type === "IncrementalRidgeToolReport" &&
		item.schema_version === "0.3" &&
		item.method_id === "incremental_ridge_closed_form" &&
		item.method_version === "0.3" &&
		item.action === action &&
		["passed", "completed", "rejected", "failed"].includes(item.status ?? "") &&
		Array.isArray(item.issues);
}

async function runTool(action: "initialize" | "inspect" | "update", payload: unknown, signal?: AbortSignal): Promise<ToolReport> {
	if (!existsSync(CLI_PATH)) throw new Error("解析增量岭回归工具不存在：" + CLI_PATH);
	const serialized = JSON.stringify(payload);
	if (typeof serialized !== "string" || Buffer.byteLength(serialized, "utf8") > MAX_INPUT_BYTES) {
		throw new Error("解析增量岭回归输入必须是小于 4 MB 的 JSON 对象。");
	}
	if (signal?.aborted) throw new Error("解析增量岭回归工具已取消。");
	return new Promise((resolvePromise, rejectPromise) => {
		const child = spawn(
			pythonExecutable(),
			["-B", "-X", "utf8", CLI_PATH, action, "--stdin"],
			{ cwd: PROJECT_ROOT, stdio: ["pipe", "pipe", "pipe"], windowsHide: true },
		);
		let stdout = "";
		let stderr = "";
		let settled = false;
		let timedOut = false;
		let outputTooLarge = false;
		const cleanup = () => {
			clearTimeout(timer);
			signal?.removeEventListener("abort", onAbort);
		};
		const rejectOnce = (error: Error) => {
			if (settled) return;
			settled = true;
			cleanup();
			rejectPromise(error);
		};
		const onAbort = () => {
			child.kill();
			rejectOnce(new Error("解析增量岭回归工具已取消。"));
		};
		const append = (current: string, chunk: Buffer): string => {
			const next = current + chunk.toString("utf8");
			if (Buffer.byteLength(next, "utf8") > MAX_OUTPUT_BYTES) {
				outputTooLarge = true;
				child.kill();
			}
			return next;
		};
		const timer = setTimeout(() => {
			timedOut = true;
			child.kill();
		}, TOOL_TIMEOUT_MS);
		signal?.addEventListener("abort", onAbort, { once: true });
		child.stdout.on("data", (chunk: Buffer) => { stdout = append(stdout, chunk); });
		child.stderr.on("data", (chunk: Buffer) => { stderr = append(stderr, chunk); });
		child.once("error", (error) => rejectOnce(new Error("无法启动解析增量岭回归工具：" + error.message)));
		child.once("close", (code) => {
			if (settled) return;
			settled = true;
			cleanup();
			if (timedOut) return rejectPromise(new Error("解析增量岭回归工具超过 60 秒，已终止。"));
			if (outputTooLarge) return rejectPromise(new Error("解析增量岭回归工具输出超过 8 MB，已终止。"));
			let report: unknown;
			try {
				report = JSON.parse(stdout.trim());
			} catch {
				return rejectPromise(new Error("解析增量岭回归工具没有返回有效 JSON。诊断：" + (stderr.trim() || stdout.trim() || String(code))));
			}
			if (!isReport(report, action)) return rejectPromise(new Error("解析增量岭回归工具返回结构不符合约定。"));
			if (report.status === "failed" || (code !== 0 && code !== 1)) {
				return rejectPromise(new Error("解析增量岭回归工具执行失败：" + (report.issues[0]?.message || stderr.trim())));
			}
			if ((code === 0) !== ["passed", "completed"].includes(report.status)) {
				return rejectPromise(new Error("解析增量岭回归工具退出码与报告状态不一致。"));
			}
			resolvePromise(report);
		});
		child.stdin.once("error", (error) => rejectOnce(new Error("无法发送解析增量岭回归输入：" + error.message)));
		child.stdin.end(serialized, "utf8");
	});
}

const inspectTool = defineTool({
	name: "inspect_incremental_ridge_state",
	label: "Inspect Incremental Ridge State",
	description: "检查已经初始化的解析任务头状态、bias增广约定、G/C统计量、当前权重与偏置及摘要是否一致；不读取项目或数据路径，不修改任何资产。",
	promptSnippet: "Inspect a bounded incremental ridge state before project adaptation or update",
	promptGuidelines: [
		"Use only for an UpdatePlan that selected incremental_ridge_closed_form v0.3.",
		"Do not invent sufficient statistics from checkpoint weights.",
		"Treat rejected status as a hard stop or replan trigger.",
	],
	parameters: Type.Object({ state: Type.Any({ description: "符合 incremental-ridge-state.schema.json 的完整状态。" }) }, { additionalProperties: false }),
	async execute(_toolCallId, params, signal) {
		const result = await runTool("inspect", params.state, signal);
		return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }], details: result };
	},
});

const initializeTool = defineTool({
	name: "initialize_incremental_ridge_state",
	label: "Initialize Incremental Ridge State",
	description: "根据固定特征提取器在原训练数据上形成的G/C统计量，直接求解解析任务头的初始权重和偏置；不要求使用原checkpoint中的任务头参数。",
	promptSnippet: "Initialize an analytic task head from historical G and C before incremental updates",
	promptGuidelines: [
		"Use after the execution Agent has identified and frozen the feature extractor and confirmed the task-head input dimension.",
		"For classification, pass C computed from fixed-order one-hot targets; for regression, pass numeric target columns.",
		"Do not compare the original gradient-trained task head with the analytic solution; the returned parameters are intended to replace that task head after evaluation.",
		"The returned state is not a saved checkpoint; save and strictly reload a new artifact without overwriting the original.",
	],
	parameters: Type.Object({ request: Type.Any({ description: "符合 incremental-ridge-initialization-request.schema.json 的完整请求。" }) }, { additionalProperties: false }),
	async execute(_toolCallId, params, signal) {
		const result = await runTool("initialize", params.request, signal);
		return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }], details: result };
	},
});

const updateTool = defineTool({
	name: "update_incremental_ridge",
	label: "Update Incremental Ridge",
	description: "对固定特征提取器产生的小规模新批次按bias约定构造设计矩阵，累计G/C并更新解析任务头；支持固定类别顺序的one-hot分类目标和连续回归目标，不扫描项目、不写checkpoint。",
	promptSnippet: "Run a bounded closed-form incremental ridge update after execution preconditions are verified",
	promptGuidelines: [
		"Call only after human plan approval and successful analytic-head initialization or state inspection.",
		"Pass the complete request bound to task_id, plan_version, and UpdatePlan SHA-256.",
		"Use one-hot target rows for fixed-class classification and numeric target rows for regression; do not update the frozen feature extractor here.",
		"The returned state is not a saved project checkpoint; the execution Agent must save and verify project artifacts without overwriting the original.",
	],
	parameters: Type.Object({ request: Type.Any({ description: "符合 incremental-ridge-update-request.schema.json 的完整请求。" }) }, { additionalProperties: false }),
	async execute(_toolCallId, params, signal) {
		const result = await runTool("update", params.request, signal);
		return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }], details: result };
	},
});

export default function incrementalRidgeExtension(pi: ExtensionAPI) {
	pi.registerTool(initializeTool);
	pi.registerTool(inspectTool);
	pi.registerTool(updateTool);
}
