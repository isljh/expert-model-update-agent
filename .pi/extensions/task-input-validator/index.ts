import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { StringEnum, Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

type ValidationMode = "draft" | "confirmed";

interface ValidationIssue {
	code: string;
	path: string;
	problem: string;
	fix: string;
}

interface ValidatorResponse {
	valid: boolean;
	schema_version: string;
	mode: ValidationMode;
	task_id: string | null;
	task_input_version: number | null;
	issues: ValidationIssue[];
	execution_error?: boolean;
}

interface AssetPreflightCheck {
	field: string;
	status: "passed" | "failed" | "skipped";
	message: string;
}

interface AssetPreflightResponse {
	report_type: "AssetPreflightReport";
	schema_version: "0.1";
	task_id: string;
	task_input_version: number;
	asset_input_digest: string;
	base_directory: string;
	overall_status: "passed" | "failed";
	checks: AssetPreflightCheck[];
	execution_error?: boolean;
}

const EXTENSION_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(EXTENSION_DIRECTORY, "..", "..", "..");
const VALIDATOR_PATH = resolve(
	PROJECT_ROOT,
	"src",
	"expert_model_update",
	"tools",
	"task_input",
	"validate.py",
);
const ASSET_PREFLIGHT_PATH = resolve(
	PROJECT_ROOT,
	"src",
	"expert_model_update",
	"tools",
	"task_assets",
	"preflight.py",
);
const MAX_INPUT_BYTES = 1_000_000;
const MAX_OUTPUT_BYTES = 1_000_000;
const TOOL_TIMEOUT_MS = 10_000;

function pythonExecutable(): string {
	const configured = process.env.TASK_INPUT_PYTHON?.trim();
	if (configured) return configured;
	return process.platform === "win32" ? "python" : "python3";
}

function isValidatorResponse(value: unknown): value is ValidatorResponse {
	if (!value || typeof value !== "object") return false;
	const candidate = value as Partial<ValidatorResponse>;
	return (
		typeof candidate.valid === "boolean" &&
		typeof candidate.schema_version === "string" &&
		(candidate.mode === "draft" || candidate.mode === "confirmed") &&
		Array.isArray(candidate.issues)
	);
}

function isAssetPreflightResponse(
	value: unknown,
): value is AssetPreflightResponse {
	if (!value || typeof value !== "object") return false;
	const candidate = value as Partial<AssetPreflightResponse>;
	return (
		candidate.report_type === "AssetPreflightReport" &&
		candidate.schema_version === "0.1" &&
		typeof candidate.task_id === "string" &&
		Number.isInteger(candidate.task_input_version) &&
		typeof candidate.asset_input_digest === "string" &&
		typeof candidate.base_directory === "string" &&
		(candidate.overall_status === "passed" ||
			candidate.overall_status === "failed") &&
		Array.isArray(candidate.checks)
	);
}

async function runPythonJsonTool<T>(
	label: string,
	scriptPath: string,
	args: string[],
	input: unknown,
	signal: AbortSignal | undefined,
	isResponse: (value: unknown) => value is T,
	exitCodeMatches: (response: T, code: number | null) => boolean,
): Promise<T> {
	if (!existsSync(scriptPath)) {
		throw new Error(label + "不存在：" + scriptPath);
	}

	const serialized = JSON.stringify(input);
	if (typeof serialized !== "string") {
		throw new Error(label + "输入无法序列化为 JSON。");
	}
	if (Buffer.byteLength(serialized, "utf8") > MAX_INPUT_BYTES) {
		throw new Error(label + "输入超过工具允许的 1 MB 上限。");
	}
	if (signal?.aborted) {
		throw new Error(label + "已取消。");
	}

	return new Promise<T>((resolvePromise, rejectPromise) => {
		const executable = pythonExecutable();
		const child = spawn(
			executable,
			[
				"-B",
				"-X",
				"utf8",
				scriptPath,
				...args,
			],
			{
				cwd: PROJECT_ROOT,
				stdio: ["pipe", "pipe", "pipe"],
				windowsHide: true,
			},
		);

		let stdout = "";
		let stderr = "";
		let settled = false;
		let timedOut = false;
		let outputTooLarge = false;

		const cleanup = () => {
			clearTimeout(timeout);
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
			rejectOnce(new Error(label + "已取消。"));
		};
		const appendOutput = (current: string, chunk: Buffer): string => {
			const next = current + chunk.toString("utf8");
			if (Buffer.byteLength(next, "utf8") > MAX_OUTPUT_BYTES) {
				outputTooLarge = true;
				child.kill();
			}
			return next;
		};

		const timeout = setTimeout(() => {
			timedOut = true;
			child.kill();
		}, TOOL_TIMEOUT_MS);

		signal?.addEventListener("abort", onAbort, { once: true });
		child.stdout.on("data", (chunk: Buffer) => {
			stdout = appendOutput(stdout, chunk);
		});
		child.stderr.on("data", (chunk: Buffer) => {
			stderr = appendOutput(stderr, chunk);
		});
		child.once("error", (error) => {
			rejectOnce(
				new Error(
					"无法启动" + label + "（" +
						executable +
						"）：" +
						error.message,
				),
			);
		});
		child.once("close", (code) => {
			if (settled) return;
			settled = true;
			cleanup();

			if (timedOut) {
				rejectPromise(new Error(label + "超过 10 秒，已终止。"));
				return;
			}
			if (outputTooLarge) {
				rejectPromise(new Error(label + "输出超过 1 MB，已终止。"));
				return;
			}

			let response: unknown;
			try {
				response = JSON.parse(stdout.trim());
			} catch {
				const diagnostic = stderr.trim() || stdout.trim() || "无输出";
				rejectPromise(
					new Error(
						label + "没有返回有效 JSON。退出码：" +
							String(code) +
							"；诊断：" +
							diagnostic,
					),
				);
				return;
			}

			if (
				Boolean(
					response &&
						typeof response === "object" &&
						(response as { execution_error?: boolean }).execution_error,
				) ||
				(code !== 0 && code !== 1)
			) {
				const errorMessage =
					response && typeof response === "object"
						? (response as { error?: { message?: unknown } }).error?.message
						: undefined;
				const diagnostic =
					(typeof errorMessage === "string" ? errorMessage : "") ||
					stderr.trim() ||
					JSON.stringify(response);
				rejectPromise(new Error(label + "执行失败：" + diagnostic));
				return;
			}
			if (!isResponse(response)) {
				rejectPromise(new Error(label + "返回了不符合约定的结果结构。"));
				return;
			}
			if (!exitCodeMatches(response, code)) {
				rejectPromise(new Error(label + "的退出码与结果状态不一致。"));
				return;
			}

			resolvePromise(response);
		});

		child.stdin.once("error", (error) => {
			rejectOnce(new Error("无法向" + label + "发送输入：" + error.message));
		});
		child.stdin.end(serialized, "utf8");
	});
}

async function runPythonValidator(
	taskInput: unknown,
	mode: ValidationMode,
	signal: AbortSignal | undefined,
): Promise<ValidatorResponse> {
	return runPythonJsonTool(
		"TaskInput 校验器",
		VALIDATOR_PATH,
		["--stdin", "--mode", mode, "--output", "json"],
		taskInput,
		signal,
		isValidatorResponse,
		(response, code) => (code === 0) === response.valid,
	);
}

async function runAssetPreflight(
	taskInput: unknown,
	signal: AbortSignal | undefined,
): Promise<AssetPreflightResponse> {
	return runPythonJsonTool(
		"资产入口预检器",
		ASSET_PREFLIGHT_PATH,
		["--stdin", "--output", "json"],
		taskInput,
		signal,
		isAssetPreflightResponse,
		(response, code) =>
			(code === 0) === (response.overall_status === "passed"),
	);
}

const validateTaskInputTool = defineTool({
	name: "validate_task_input",
	label: "Validate TaskInput",
	description:
		"确定性校验 TaskInput v0.3 的结构、固定必答问题映射和人工确认门槛，不读取模型、权重、配置或数据。",
	promptSnippet: "Validate a TaskInput draft or confirmation gate deterministically",
	promptGuidelines: [
		"Use validate_task_input in draft mode after creating or changing a TaskInput draft.",
		"Use validate_task_input in confirmed mode before handing the task to model delivery inspection.",
		"Do not infer that validation passed when validate_task_input is unavailable or returns an execution error.",
	],
	parameters: Type.Object(
		{
			task_input: Type.Any({
				description: "完整的 TaskInput JSON 对象。",
			}),
			mode: StringEnum(["draft", "confirmed"] as const, {
				description:
					"draft 检查草案结构与问题映射；confirmed 额外检查进入模型交付检查的确认门槛。",
			}),
		},
		{ additionalProperties: false },
	),

	async execute(_toolCallId, params, signal) {
		const result = await runPythonValidator(
			params.task_input,
			params.mode as ValidationMode,
			signal,
		);
		return {
			content: [
				{
					type: "text",
					text: JSON.stringify(result, null, 2),
				},
			],
			details: result,
		};
	},
});

const preflightTaskAssetsTool = defineTool({
	name: "preflight_task_assets",
	label: "Preflight Task Assets",
	description:
		"只读检查 TaskInput 声明的模型文件、构造配置、checkpoint 和新数据路径是否存在、类型正确且当前可读；不扫描内容、不导入模型、不加载权重、不解析配置。",
	promptSnippet: "Preflight declared TaskInput asset paths before human confirmation",
	promptGuidelines: [
		"Use preflight_task_assets after draft validation passes and all required questions have been answered and synchronized to their formal fields.",
		"Do not ask for final human confirmation while the asset preflight result is failed.",
		"Run it again immediately before model delivery inspection, and require the result to match the current task_id, task_input_version, and asset declarations.",
		"Do not replace this tool with shell commands or infer that a path is available from its text.",
	],
	parameters: Type.Object(
		{
			task_input: Type.Any({
				description: "需要检查其 assets 声明的完整 TaskInput JSON 对象。",
			}),
		},
		{ additionalProperties: false },
	),

	async execute(_toolCallId, params, signal) {
		const result = await runAssetPreflight(params.task_input, signal);
		return {
			content: [
				{
					type: "text",
					text: JSON.stringify(result, null, 2),
				},
			],
			details: result,
		};
	},
});

export default function taskInputValidatorExtension(pi: ExtensionAPI) {
	pi.registerTool(validateTaskInputTool);
	pi.registerTool(preflightTaskAssetsTool);
}
