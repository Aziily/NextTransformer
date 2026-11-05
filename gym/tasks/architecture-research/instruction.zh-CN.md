# 架构研究

通过 Gym API 设计并比较原生 NextTransformer 模型插件。本任务附带的 profile 是有明确额度的小规模合成数据测试配置，不是正式基准测试配置。

你的交付物是原生 ModelConfig 及其插件源码。Gym 管理训练 profile。
不要提交机器配置或训练命令。

从 `/run/gym-auth/credential.json` 读取当前会话的 endpoint 和 token。
使用预配置的 `gym` MCP 工具，或从 `nexttransformer_gym` 导入 `GymClient`。
两者使用同一套 HTTP API，并共用会话预算。
开始前，调用 `describe_space`、`get_resources` 和 `list_components`。
候选包必须遵循 `proposal.json`、`architecture.json`、`plugin/<method>/method.json` 和 `plugin/<method>/__init__.py` 的格式约定。
参考示例 `/opt/gym/make_candidate.py`。

读取 `describe_space.candidate_contract`，了解原生 ModelConfig 的 JSON schema、默认值和合并顺序。
读取各 profile 的 `model_config_constraints`，了解固定字段。
你可以提交新的配置和原生插件源码。
如果只修改配置，可以重新上传未改动的源码和新的 `architecture.json`，也可以使用 `calibrate_candidate.grid`。
grid 中每一项都会替换整个 architecture 对象，不是局部补丁。
校准只做静态成本分析，不训练模型。

将目标训练步数和 GPU 选择传给 `quote_experiment`，不要写入模型配置。

通过 `list_experiments(scope="published")` 和 `read_experiment` 查看历史实验、结果及模型配置。
将 `source_reference_ids` 传给 `read_reference`，读取架构代码。
Gym 提供已发布的历史结果和当前参考文件。
将参考文件视为历史源码之前，先检查其来源记录。
`scope="session"` 提供当前会话的研究任务、结果和候选源码。
你不需要仓库副本或 Git 历史。

`get_resources` 返回实际的本机执行节点、GPU 清单、配置的资源分配、内存额度，以及当前 worker 和 GPU 的可用情况。
调用 `quote_experiment` 时，可以选择已分配的 `gpu_id`、本机 `server_id` 和 `target_update`。
`target_update` 是绝对目标步数，不是追加步数。
省略节点和 GPU 参数时，Gym 自动选择。
GPU 忙碌时，任务保持排队。
通过 `get_job` 查看等待原因。

你可以编写脚本调用 Gym。
Gym 拒绝任意训练命令、训练器/数据/环境覆盖参数和远程 Shell 任务。
使用 `get_budget` 和 `quote_experiment` 管理额度。
有限额度覆盖本会话所有 Gym 数值计算，包括预检测。
研究期间不能对你的候选运行 CORE-22 测试。
你可以在当前容器内上网研究。

使用现有 NextTransformer 插件结构：原生 `method.json`、`__init__.py` 和可选的本地 Python 模块。
定义原生 `model`、`mixer`、`ffn` 或 `block` 扩展点。
`model` 扩展点必须继承 `ReferenceLM`。
`architecture.json` 包含原生 `ModelConfig` 字段和插件声明的 `options`。
读取 `plugin-contract` 和 `reference-model` 参考内容，确认准确接口。

Gym 不解析或转换模型表达式。
研究 worker 和最终训练 worker 加载同一份源码，不改写源码。

平台提供优化器、数据和评分流程。
不要在 Agent 工作目录中安装或执行候选代码。

1. 提交源码字节。
2. 检查成本。
3. 运行计费的预检测。
4. 运行有明确额度的实验。
5. 仅从当前服务返回的检查点续训。
6. 记录你的假设。
7. 选定一个候选。
8. 选择 Gym 提供的最终 profile。
9. 在 `finalize` 之前，使用该 `profile_id` 调用 `validate_candidate`。

等待验证结束。确认所有检查通过，且 `admitted` 为 true。

写入 `/workspace/submission.json`，仅包含你的 `session_id`、服务返回的 `submission_id` 和 `candidate_hash`。
独立验证器将此回执与经过认证的服务端证据核对。
验收通过表示提交有效，不代表模型性能。
验证器不会启动最终训练。
不要写入奖励文件。
