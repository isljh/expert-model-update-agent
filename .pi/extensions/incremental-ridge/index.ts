import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

interface ToolReport {
	report_type: "IncrementalRidgeToolReport";
	schema_version: "0.1";
	method_id: "incremental_ridge_closed_form";
	method_version: "0.1";
	action: "inspect" | "update";
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

function isReport(value: unknown, action: "inspect" | "update"): value is ToolReport {
	if (!value || typeof value !== "object") return false;
	const item = value as Partial<ToolReport>;
	return item.report_type === "IncrementalRidgeToolReport" &&
		item.schema_version === "0.1" &&
		item.method_id === "incremental_ridge_closed_form" &&
		item.method_version === "0.1" &&
		item.action === action &&
		["passed", "completed", "rejected", "failed"].includes(item.status ?? "") &&
		Array.isArray(item.issues);
}

async function runTool(action: "inspect" | "update", payload: unknown, signal?: AbortSignal): Promise<ToolReport> {
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
	description: "检查解析增量岭回归旧状态、固定目标函数、充分统计量、当前参数和摘要是否一致；不读取项目或数据路径，不修改任何资产。",
	promptSnippet: "Inspect a bounded incremental ridge state before project adaptation or update",
	promptGuidelines: [
		"Use only for an UpdatePlan that selected incremental_ridge_closed_form v0.1.",
		"Do not invent sufficient statistics from checkpoint weights.",
		"Treat rejected status as a hard stop or replan trigger.",
	],
	parameters: Type.Object({ state: Type.Any({ description: "符合 incremental-ridge-state.schema.json 的完整状态。" }) }, { additionalProperties: false }),
	async execute(_toolCallId, params, signal) {
		const result = await runTool("inspect", params.state, signal);
		return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }], details: result };
	},
});

const updateTool = defineTool({
	name: "update_incremental_ridge",
	label: "Update Incremental Ridge",
	description: "对已标准化的小规模新批次累计充分统计量并求解固定的平方误差加 L2 正则目标；不扫描项目、不写 checkpoint、不改变目标函数。",
	promptSnippet: "Run a bounded closed-form incremental ridge update after execution preconditions are verified",
	promptGuidelines: [
		"Call only after human plan approval and execution-agent verification of the project, objective, feature contract, and old state binding.",
		"Pass the complete request bound to task_id, plan_version, and UpdatePlan SHA-256.",
		"Do not use this tool for classification, non-squared losses, neural-network Linear layers, changed feature contracts, or checkpoint-only state.",
		"The returned state is not a saved project checkpoint; the execution Agent must save and verify project artifacts without overwriting the original.",
	],
	parameters: Type.Object({ request: Type.Any({ description: "符合 incremental-ridge-update-request.schema.json 的完整请求。" }) }, { additionalProperties: false }),
	async execute(_toolCallId, params, signal) {
		const result = await runTool("update", params.request, signal);
		return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }], details: result };
	},
});

export default function incrementalRidgeExtension(pi: ExtensionAPI) {
	pi.registerTool(inspectTool);
	pi.registerTool(updateTool);
}
