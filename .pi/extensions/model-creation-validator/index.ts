import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

interface ModelCreationValidationResponse {
	report_type: "ModelCreationValidationReport";
	schema_version: "0.2";
	proposal_results: unknown[];
	selection: {
		status: string;
		selected_proposal_id: string | null;
	};
	readiness: {
		ready_for_model_construction: boolean;
		ready_for_checkpoint_load_attempt: boolean;
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
const MAX_INPUT_BYTES = 2_000_000;
const MAX_OUTPUT_BYTES = 8_000_000;
const TOOL_TIMEOUT_MS = 60_000;

function pythonExecutable(): string {
	const configured = process.env.MODEL_CREATION_PYTHON?.trim();
	if (configured) return configured;
	return process.platform === "win32" ? "python" : "python3";
}

function isResponse(value: unknown): value is ModelCreationValidationResponse {
	if (!value || typeof value !== "object") return false;
	const candidate = value as Partial<ModelCreationValidationResponse>;
	return (
		candidate.report_type === "ModelCreationValidationReport" &&
		candidate.schema_version === "0.2" &&
		Array.isArray(candidate.proposal_results) &&
		Boolean(candidate.selection && typeof candidate.selection === "object") &&
		Boolean(candidate.readiness && typeof candidate.readiness === "object") &&
		typeof candidate.readiness?.ready_for_model_construction === "boolean" &&
		typeof candidate.readiness?.ready_for_checkpoint_load_attempt === "boolean" &&
		Array.isArray(candidate.readiness?.blockers)
	);
}

async function runValidator(
	proposalSet: unknown,
	signal: AbortSignal | undefined,
): Promise<ModelCreationValidationResponse> {
	if (!existsSync(VALIDATOR_PATH)) {
		throw new Error("模型创建方案静态验证器不存在：" + VALIDATOR_PATH);
	}
	const serialized = JSON.stringify(proposalSet);
	if (typeof serialized !== "string") {
		throw new Error("ModelCreationProposalSet 无法序列化为 JSON。");
	}
	if (Buffer.byteLength(serialized, "utf8") > MAX_INPUT_BYTES) {
		throw new Error("ModelCreationProposalSet 超过工具允许的 2 MB 上限。");
	}
	if (signal?.aborted) throw new Error("模型创建方案静态验证已取消。");

	return new Promise((resolvePromise, rejectPromise) => {
		const child = spawn(
			pythonExecutable(),
			["-B", "-X", "utf8", VALIDATOR_PATH, "--stdin", "--output", "json"],
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
			rejectOnce(new Error("模型创建方案静态验证已取消。"));
		};
		const append = (current: string, chunk: Buffer): string => {
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
			stdout = append(stdout, chunk);
		});
		child.stderr.on("data", (chunk: Buffer) => {
			stderr = append(stderr, chunk);
		});
		child.once("error", (error) => {
			rejectOnce(new Error("无法启动模型创建方案静态验证器：" + error.message));
		});
		child.once("close", (code) => {
			if (settled) return;
			settled = true;
			cleanup();
			if (timedOut) {
				rejectPromise(new Error("模型创建方案静态验证超过 60 秒，已终止。"));
				return;
			}
			if (outputTooLarge) {
				rejectPromise(new Error("模型创建方案静态验证输出超过 8 MB，已终止。"));
				return;
			}
			let response: unknown;
			try {
				response = JSON.parse(stdout.trim());
			} catch {
				rejectPromise(
					new Error(
						"模型创建方案静态验证器没有返回有效 JSON。退出码：" +
							String(code) +
							"；诊断：" +
							(stderr.trim() || stdout.trim() || "无输出"),
					),
				);
				return;
			}
			if (!isResponse(response)) {
				rejectPromise(new Error("模型创建方案静态验证器返回结构不符合约定。"));
				return;
			}
			if (response.execution_error || (code !== 0 && code !== 1)) {
				rejectPromise(
					new Error(
						"模型创建方案静态验证器执行失败：" +
							(response.error?.message || stderr.trim() || JSON.stringify(response)),
					),
				);
				return;
			}
			if ((code === 0) !== response.readiness.ready_for_model_construction) {
				rejectPromise(new Error("模型创建方案静态验证器的退出码与就绪状态不一致。"));
				return;
			}
			resolvePromise(response);
		});
		child.stdin.once("error", (error) => {
			rejectOnce(new Error("无法向模型创建方案静态验证器发送输入：" + error.message));
		});
		child.stdin.end(serialized, "utf8");
	});
}

const validateModelCreationSpecTool = defineTool({
	name: "validate_model_creation_spec",
	label: "Validate Model Creation Spec",
	description:
		"对项目理解 Agent 提出的完整对象创建链做定向、确定性的契约与源码锚点检查；不打开 checkpoint，不重新发现模型候选，不执行目标项目，不创建模型实例。",
	promptSnippet: "Validate proposed model construction fields before safe instantiation",
	promptGuidelines: [
		"Call this tool only after TaskInput confirmation, asset preflight, and project understanding have produced one or more concrete ModelCreationProposalSet v0.2 entries.",
		"Pass the matching passed AssetPreflightReport so the tool can bind the proposal to confirmed asset locations without reopening or inspecting checkpoint contents.",
		"Do not pass the whole prose project-understanding report; construct only the ModelCreationProposalSet contract.",
		"Represent every externally reproducible creation or wrapper call as an ordered construction_steps entry; use object_ref arguments to connect later steps to earlier results.",
		"Keep result.runtime_object, result.target_model, lifecycle-operation targets, and checkpoint.load_target as explicit object references; do not collapse wrapper and inner model into one callable.",
		"Do not claim an instance can be created unless ready_for_model_construction is true.",
		"Treat ready_for_checkpoint_load_attempt only as confirmation that the Agent-proposed loading recipe is structurally bound to source anchors, object references, and authorized locations; actual checkpoint inspection and loading belong to their dedicated stages.",
	],
	parameters: Type.Object(
		{
			proposal_set: Type.Any({
				description:
					"符合 schemas/model-creation-proposal.schema.json 的 ModelCreationProposalSet v0.2 完整对象创建链。",
			}),
		},
		{ additionalProperties: false },
	),

	async execute(_toolCallId, params, signal) {
		const result = await runValidator(params.proposal_set, signal);
		return {
			content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
			details: result,
		};
	},
});

export default function modelCreationValidatorExtension(pi: ExtensionAPI) {
	pi.registerTool(validateModelCreationSpecTool);
}
