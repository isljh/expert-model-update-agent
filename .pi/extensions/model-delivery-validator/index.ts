// Pi adapter for the declared three-file model delivery validator.
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

interface ModelDeliveryValidationResponse {
	report_type: "ModelDeliveryValidationReport";
	schema_version: "0.1";
	status: "validated" | "failed" | "execution_error";
	findings: unknown[];
	validated_load_plan: unknown;
	readiness: {
		ready_for_model_load: boolean;
		ready_for_structure_analysis: boolean;
		blockers: string[];
	};
	execution_error?: boolean;
	error?: { code?: string; message?: string };
}

const EXTENSION_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = resolve(EXTENSION_DIRECTORY, "..", "..", "..");
const VALIDATOR_PATH = resolve(
	PROJECT_ROOT,
	"src",
	"expert_model_update",
	"tools",
	"model_creation",
	"validate.py",
);
const MAX_INPUT_BYTES = 8_000_000;
const MAX_OUTPUT_BYTES = 8_000_000;
const TOOL_TIMEOUT_MS = 60_000;

function pythonExecutable(): string {
	const configured = process.env.MODEL_CREATION_PYTHON?.trim();
	if (configured) return configured;
	return process.platform === "win32" ? "python" : "python3";
}

function isResponse(value: unknown): value is ModelDeliveryValidationResponse {
	if (!value || typeof value !== "object") return false;
	const candidate = value as Partial<ModelDeliveryValidationResponse>;
	return (
		candidate.report_type === "ModelDeliveryValidationReport" &&
		candidate.schema_version === "0.1" &&
		(candidate.status === "validated" ||
			candidate.status === "failed" ||
			candidate.status === "execution_error") &&
		Array.isArray(candidate.findings) &&
		Boolean(candidate.readiness && typeof candidate.readiness === "object") &&
		typeof candidate.readiness?.ready_for_model_load === "boolean" &&
		typeof candidate.readiness?.ready_for_structure_analysis === "boolean" &&
		Array.isArray(candidate.readiness?.blockers)
	);
}

async function runValidator(
	input: unknown,
	signal: AbortSignal | undefined,
): Promise<ModelDeliveryValidationResponse> {
	if (!existsSync(VALIDATOR_PATH)) {
		throw new Error("模型交付验证器不存在：" + VALIDATOR_PATH);
	}
	const serialized = JSON.stringify(input);
	if (typeof serialized !== "string" || Buffer.byteLength(serialized, "utf8") > MAX_INPUT_BYTES) {
		throw new Error("模型交付验证输入必须是小于 8 MB 的 JSON 对象。");
	}
	if (signal?.aborted) throw new Error("模型交付验证已取消。");

	return new Promise((resolvePromise, rejectPromise) => {
		const child = spawn(
			pythonExecutable(),
			["-B", "-X", "utf8", VALIDATOR_PATH, "--stdin", "--output", "json"],
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
			rejectOnce(new Error("模型交付验证已取消。"));
		};
		const timer = setTimeout(() => {
			timedOut = true;
			child.kill();
		}, TOOL_TIMEOUT_MS);
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
		child.once("error", error => rejectOnce(new Error("无法启动模型交付验证器：" + error.message)));
		child.once("close", code => {
			if (settled) return;
			settled = true;
			cleanup();
			if (timedOut) return rejectPromise(new Error("模型交付验证超过 60 秒。"));
			if (outputTooLarge) return rejectPromise(new Error("模型交付验证输出超过 8 MB。"));
			let response: unknown;
			try { response = JSON.parse(stdout.trim()); }
			catch {
				return rejectPromise(new Error("模型交付验证器没有返回有效 JSON：" + (stderr.trim() || String(code))));
			}
			if (!isResponse(response)) return rejectPromise(new Error("模型交付验证报告结构不符合约定。"));
			if (response.execution_error || (code !== 0 && code !== 1)) {
				return rejectPromise(new Error("模型交付验证器执行失败：" + (response.error?.message || stderr.trim() || JSON.stringify(response))));
			}
			if ((code === 0) !== response.readiness.ready_for_model_load) {
				return rejectPromise(new Error("模型交付验证器退出码与就绪状态不一致。"));
			}
			resolvePromise(response);
		});
		child.stdin.once("error", error => rejectOnce(new Error("无法向模型交付验证器发送输入：" + error.message)));
		child.stdin.end(serialized, "utf8");
	});
}

const validateModelDeliveryTool = defineTool({
	name: "validate_model_delivery",
	label: "Validate Model Delivery",
	description:
		"确定性验证唯一声明的模型文件、构造配置和 checkpoint：绑定已确认 TaskInput 与检查报告，静态核对模型类和构造参数，并确认声明的 state_dict 路径；不创建实例。",
	promptSnippet: "Validate the declared model file, construction config, and checkpoint evidence",
	promptGuidelines: [
		"Call only after TaskInput v0.3 confirmation, matching asset preflight, inspect_model_config, and inspect_model_checkpoint.",
		"Do not search a project, generate model candidates, or infer missing constructor values from weight shapes.",
		"Do not claim the model can be loaded unless ready_for_model_load is true.",
		"Actual class import, construction, state_dict loading, and trainability checks belong to load_validated_model or analyze_validated_model_structure.",
	],
	parameters: Type.Object({
		task_input: Type.Any({ description: "已确认的 TaskInput v0.3。" }),
		asset_preflight_report: Type.Any({ description: "与 TaskInput 匹配且已通过的资产预检报告。" }),
		config_inspection_report: Type.Any({ description: "inspect_model_config 的完整成功输出。" }),
		checkpoint_inspection_report: Type.Any({ description: "inspect_model_checkpoint 的完整成功输出。" }),
	}, { additionalProperties: false }),
	async execute(_toolCallId, params, signal) {
		const result = await runValidator(params, signal);
		return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }], details: result };
	},
});

export default function modelDeliveryValidatorExtension(pi: ExtensionAPI) {
	pi.registerTool(validateModelDeliveryTool);
}
