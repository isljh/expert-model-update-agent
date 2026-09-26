# 模型项目理解输出模板

本模板定义项目理解报告的逻辑内容，供人类和更新决策 Agent 阅读。静态验证工具不解析这份自然语言报告；机器交接必须另外生成符合 [`../../../schemas/model-creation-proposal.schema.json`](../../../schemas/model-creation-proposal.schema.json) 的 `ModelCreationProposalSet`。

## 顶层结构

```yaml
analysis_scope:
  task_id: string
  task_input_version: integer
  source_project_path: string
  source_revision: string | null
  checkpoint_reference: string | null
  declared_config_reference: string | null

declared_asset_evidence:
  config_inspection:
    status: inspected | unavailable | failed | not_provided
    path: string | null
    sha256: string | null
    evidence_refs: [string]
  checkpoint_inspection:
    status: inspected | unavailable | failed | not_provided
    path: string | null
    evidence_refs: [string]
  conflicts: [string]

model_candidates:
  - candidate_id: string
    proposed_role: string
    role_status: observed | inferred | unresolved

    construction:
      construction_callable: string | null
      concrete_model_definition: string | null
      construction_call_site: string | null
      result_extraction_path: string | null
      call_expression: string | null

    arguments:
      - parameter: string
        source_expression: string
        source_kind: config | constant | function_argument | cli_default | environment | dataset_derived | checkpoint_derived | runtime | unknown
        resolved_value: any | null
        status: observed | inferred | unresolved
        evidence_refs: [string]

    lifecycle_operations:
      - order: integer
        operation: string
        effect: string
        status: observed | inferred | unresolved
        evidence_refs: [string]

    checkpoint_binding:
      load_call_site: string | null
      load_target: string | null
      state_dict_selector: string | null
      key_transform: string | null
      status: observed | inferred | unresolved
      evidence_refs: [string]

    evidence_refs: [string]
    contradictions: [string]
    unresolved: [string]

candidate_assessment:
  recommended_candidate_id: string | null
  recommendation_status: observed | inferred | unresolved
  rationale: [string]
  alternatives_not_excluded: [string]

evidence:
  - evidence_id: string
    status: observed | inferred | unresolved
    path: string
    line: integer | null
    symbol: string | null
    fact: string
    reasoning: string | null

unresolved_questions:
  - question: string
    blocks_static_validation: boolean
    next_evidence_needed: string

handoff:
  ready_for_static_validation: boolean
  not_ready_reasons: [string]
```

## 字段边界

### `construction_callable`

实际准备调用的类、函数或工厂。它对应旧接口中 `model_definition_site` 的实际用途，但名称不应暗示它必然是最终 `nn.Module` 类定义。

### `concrete_model_definition`

能够静态确定时，记录工厂或注册表最终选择的具体模型实现。不能确定时保持 `null`，由静态验证继续解析。

### `construction_call_site`

原执行路径中发生创建调用的模块、函数或类方法。它对应旧接口中的 `model_instantiation_site`，但必须附调用表达式和源码证据。

### `result_extraction_path`

从构造结果取得目标 `torch.nn.Module` 的候选路径，对应旧接口中的 `model_instance_path`。构造结果本身就是目标模型时使用 `$`；无法确定时使用 `null`，不能默认假设 `$`。

### `recommended_candidate_id`

它只是 Agent 基于源码语义提出的推荐项。后续静态验证工具可以否决它；没有充分排他证据时必须为 `null`。

### `ready_for_static_validation`

只有满足以下条件时为 `true`：

- 至少存在一个具有实际源码调用证据的候选；
- 候选的构造入口或创建位置不是纯名称猜测；
- 已明确列出构造参数来源和未解决项；
- 已调查创建后结构变化与 checkpoint 绑定；
- 所有保留候选都有可定位证据。

这个状态不能命名或解释为 `ready_for_instantiation`。

## 机器交接边界

项目理解报告完成后，只把创建实例所需字段转换为 `ModelCreationProposalSet`：

- 当前任务、版本和资产预检绑定；
- 项目路径和 revision；
- 每个具体方案中按执行顺序排列的 `construction_steps`；
- 每一步的 callable、位置参数、关键字参数和独立源码锚点；
- 后续步骤参数到前序步骤结果的 `object_ref`；
- 最终运行时对象与目标模型各自的 `{step_id, path}`；
- checkpoint 加载前后的必要操作；
- checkpoint 路径、选择路径、键转换、加载目标和 `strict`；
- 创建调用和 checkpoint 加载调用的源码锚点。

不要把候选理由、文档冲突、排除过程或完整证据正文复制到机器交接。一个模型角色对应多个仍可行的 callable 或参数集合时，分别生成多个 proposal；没有上游选择证据时保持 `selected_proposal_id: null`。

下面只演示“工厂结果被 `DataParallel` 包装”时的对象关系；路径、符号、行号和证据必须替换为目标项目事实：

```yaml
schema_version: "0.2"
proposals:
  - proposal_id: P1
    construction_steps:
      - step_id: base_model
        callable: {origin: project, module: resnet, symbol: resnet20}
        positional_arguments: []
        keyword_arguments: []
        source_anchor:
          path: trainer.py
          scope: main
          line: 70
          call_expression: resnet.__dict__[args.arch]()
        evidence_refs: [E3]
      - step_id: parallel_wrapper
        callable: {origin: pytorch, module: torch.nn, symbol: DataParallel}
        positional_arguments:
          - value:
              kind: object_ref
              object: {step_id: base_model, path: "$"}
            source_expression: resnet.__dict__[args.arch]()
        keyword_arguments: []
        source_anchor:
          path: trainer.py
          scope: main
          line: 70
          call_expression: torch.nn.DataParallel(resnet.__dict__[args.arch]())
        evidence_refs: [E3]
    result:
      runtime_object: {step_id: parallel_wrapper, path: "$"}
      target_model: {step_id: parallel_wrapper, path: "$.module"}
```

这不是可以复制到任何项目的答案。它只说明嵌套源码表达式也必须拆成两步，并明确“运行时包装对象”和“内部目标模型”不是同一对象引用。

## 最小证据示例

```yaml
evidence:
  - evidence_id: E1
    status: observed
    path: trainer.py
    line: 42
    symbol: _train
    fact: get_model(args.model_name, args) 的返回结果被赋给 model
    reasoning: null

  - evidence_id: E2
    status: inferred
    path: trainer.py
    line: 55
    symbol: _train
    fact: model._network 是 checkpoint 的候选加载目标
    reasoning: 同一对象随后传给 load_state_dict，但尚未运行时验证其类型
```

不要引用本模板中的示例作为目标项目证据。
