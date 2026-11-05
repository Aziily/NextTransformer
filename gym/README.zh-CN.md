# Gym 使用文档

[English](README.md) · [设计文档](DESIGN.zh-CN.md)

Gym 在宿主机上运行架构实验。Harbor 在独立容器中运行 Claude Code 和提交验证器。
示例使用合成数据，不用于衡量生产 ClimbMix 或 CORE-22 成绩。

Harbor 读取 [Agent 任务指令](tasks/architecture-research/instruction.md)。

## 安装

宿主机需要 Linux、Docker、bubblewrap，以及非特权用户命名空间。Gym 需要 Python 3.12 或 3.13。
在仓库根目录执行：

```sh
uv sync --project gym --extra worker
docker build -f gym/Dockerfile --target client -t nt-gym-client:local .
docker build -f gym/Dockerfile --target verifier -t nt-gym-verifier:local .
```

`worker` 依赖组安装 CPU 运行依赖。GPU 训练进程需要 CUDA Python 环境，依赖版本见 `gym/pyproject.toml`。
Gym 不需要 CUDA 容器镜像。部署时，用经过检查的 digest 固定 Docker 基础镜像。
启动器要求镜像已存在于本机，并记录其不可变 ID。

## 启动研究会话

在控制器环境中设置模型供应商凭据。Harbor 将凭据传给 Claude Code。
示例配置指定模型、CLI 版本、时间上限、轮次上限和 Agent 美元费用上限。

Agent 美元费用上限是可选项。不配置 `agent.max_budget_usd` 时，启动器不设置 API 费用上限。
`--max-budget-usd` 可覆盖配置中的额度。显式设置的额度必须是大于零的有限数值。

`budget.research_flops_limit` 是整个会话共用的研究 FLOPs 额度。最终训练使用独立额度。

```sh
uv run --project gym python gym/run_trial.py --config gym/examples/claude-code.toml --dry-run
uv run --project gym python gym/run_trial.py --config gym/examples/claude-code.toml
```

使用 DeepSeek Flash 时，选择 `gym/examples/deepseek-v4.1-flash.toml`。
将 `GYM_RESEARCH_MODEL` 设为供应商实际提供的模型 ID。
配置 `ANTHROPIC_BASE_URL` 和 `ANTHROPIC_API_KEY`。接口必须支持 Anthropic Messages。
严格的费用上限应由供应商控制。CLI 美元额度依赖供应商提供的用量和价格信息。

使用 GPU 时，先创建本地配置：

```sh
mkdir -p .agent-work
cp gym/examples/resources.gpu.json .agent-work/resources.local.json
```

将 `gpu_ids` 设为已分配的 GPU 索引或 UUID。用 `gpu_memory_limit_bytes` 设置每张卡的显存额度。
保持 `max_concurrent_jobs` 为 `1`。然后启动会话：

```sh
uv run --project gym python gym/run_trial.py \
  --config gym/examples/deepseek-v4.1-flash.toml \
  --execution-config .agent-work/resources.local.json \
  --worker-python /path/to/cuda-environment/bin/python
```

CPU 示例可以省略 GPU 配置。同一时间只启动一个研究会话。
每份 Gym 状态有一个训练任务执行槽。不同状态目录不共用整台机器的调度锁。
GPU 忙时，作业保持排队。Gym 不会把 GPU 作业转到 CPU。

<a id="gym-api"></a>

## Gym 接口

在 Agent 容器中读取 `/run/gym-auth/credential.json`。文件包含 `endpoint`、`token` 和 `session_id`。
Harbor 会注册 `gym` MCP 工具。Python `GymClient` 使用同一套 HTTP API。
外部 MCP 客户端可以启动 `python -m nexttransformer_gym.mcp_server --credential <credential.json>`。

直接调用 HTTP 时，向 `POST <endpoint>/v1/call` 发送 JSON。
请求头包含 `Content-Type: application/json` 和 `Authorization: Bearer <token>`。

```json
{"schema_version": "gym.api.v1", "operation": "describe_space", "params": {}}
```

成功的 HTTP 响应包含 `schema_version` 和 `result`。SDK 与 MCP 直接返回 result 的值。

先调用 `describe_space`。其中的 `operation_schemas` 给出参数类型、必填字段和限制。
`candidate_contract` 给出模型配置 schema 和合并规则。各个执行配置（profile）给出固定字段值。

| 操作 | 参数 | 返回内容 |
| --- | --- | --- |
| `describe_space` | 无 | API schema、候选规则和执行配置 |
| `get_resources` | 可选 `resource_profile_id` | 宿主资源、GPU 分配与可用状态 |
| `list_components` | 可选 `cursor`、`limit` | 允许读取的参考资料 ID |
| `read_reference` | `resource_id` | 参考资料正文 |
| `list_experiments` | 可选 `scope`、`cursor`、`limit` | 实验摘要 |
| `read_experiment` | `experiment_id`，可选 `scope` | 结果、模型配置和源码引用 |
| `submit_candidate` | `package_base64` | 候选 ID、哈希和元数据 |
| `list_candidates` | 可选 `cursor`、`limit` | 当前会话的候选 |
| `get_candidate` | `candidate_id` | 候选元数据和已提交的配置 |
| `inspect_cost` | `candidate_id`、`profile_id` | 跟踪得到的 FLOPs、参数量和覆盖情况 |
| `calibrate_candidate` | `candidate_id`、`profile_id`、`grid` | 包含派生候选和成本的静态分析作业 |
| `validate_candidate` | `candidate_id`、`profile_id` | 完整检查作业，包含最终准入检查 |
| `debug_candidate` | `candidate_id`、`profile_id`、`check`，可选 `shape` | 指定诊断作业 |
| `quote_experiment` | `candidate_id`、`profile_id`，以及下述可选资源与续训参数 | 成本上界及额度是否足够 |
| `run_experiment` | `quote_id` | 已排队的作业 |
| `get_job` | `job_id` | 作业状态、进度和执行信息 |
| `get_result` | `job_id` | 已完成的结果 |
| `cancel_job` | `job_id` | 更新后的取消状态 |
| `read_events` | 可选整数 `cursor`、`limit` | 当前会话的事件 |
| `get_budget` | 无 | 总额、已消耗、已预留和可用 FLOPs |
| `finalize` | `candidate_id`、`profile_id` | 已冻结的提交 |
| `get_submission` | `submission_id` | 提交状态和最终训练状态 |

`scope="published"` 读取公开实验，`scope="session"` 读取本会话实验。默认值是 `published`。
分页游标通常是字符串，只有 `read_events` 使用整数游标。`limit` 范围为 1–100，默认 50。

以下操作需要 `idempotency_key`：`submit_candidate`、`calibrate_candidate`、`validate_candidate`、`debug_candidate`、`run_experiment`、`cancel_job`、`finalize`。
键长为 1–128 个字符。重试相同输入时复用原键。输入改变时使用新键。
报价会创建记录，但不预留 FLOPs。`run_experiment` 才会预留报价额度。

报价可选参数为 `target_update`、`checkpoint_id`、`resource_profile_id`、`server_id` 和 `gpu_id`。
`target_update` 是绝对目标步数，不是追加步数。默认值为 1，不能超过执行配置的上限。
续训时，目标步数必须大于检查点步数。
从 `get_resources` 获取本机节点 ID，以及已分配 GPU 的 UUID 或索引字符串。
省略资源选择参数时，Gym 自动选择已分配的空闲 GPU。

诊断名称为 `construction`、`numerical`、`causal`、`checkpoint` 和 `cost_coverage`。
若提供 `shape`，其批大小和序列长度必须与执行配置一致。
诊断会扣除所需算力，但不会授予最终准入资格。

## 提交原生候选

候选包含原生插件源码，以及部分或完整的原生 `ModelConfig` 字段。
ZIP 结构如下：

```text
proposal.json
architecture.json
plugin/<method>/method.json
plugin/<method>/__init__.py
plugin/<method>/<optional local Python modules>
```

ZIP 文件和解压后的总内容分别不能超过 1 MiB。最多包含 32 个 UTF-8 文本文件，不要加入目录条目。
不要加入权重、安装程序或训练脚本。上传时，Gym 保存源码，但不导入源码。
打包示例见 [make_candidate.py](examples/make_candidate.py)。

`proposal.json` 示例：

```json
{
  "schema_version": "gym.candidate.v1",
  "sdk_version": "reference_plugin_v1",
  "method": "research_model",
  "parent_ids": [],
  "hypothesis": "Compare a deeper model under the same training protocol."
}
```

前三个字段必填。方法名称必须与插件目录和 `method.json.name` 一致，且不能覆盖已注册方法。
可选的 `parent_ids` 最多包含 16 个本会话候选 ID。Gym 只将研究假设保存为元数据。

示例插件对应的 `architecture.json`：

```json
{
  "layers": 4,
  "mlp_width": 128,
  "window_pattern": "L",
  "options": {"activation": "silu", "output_scale": 1.0}
}
```

使用原生 `ModelConfig` 字段，以及 `method.json.options` 声明的选项。
选项值可以是字符串、有限数值、布尔值，或这些类型组成的嵌套列表。
Gym 不解析模型表达式。字段组合仍须满足原生检查。
原生 `model` hook 必须继承 `ReferenceLM`。通过 `plugin-contract` 和 `reference-model` 参考资料读取 hook 接口。

| 字段 | 输入规则 |
| --- | --- |
| `layers`、`mlp_width`、`window_pattern` 等架构字段 | 满足公开类型和原生检查。部分字段只适用于特定方法。 |
| `method` | 省略，或与 proposal 中的方法名称一致。 |
| `width`、`heads`、`sequence_len`、`vocab_size`、`scale` | 省略，或与所选执行配置一致。 |
| `routing_seed` | 省略，或设为 1729。 |
| 训练设置、路径、命令和环境变量 | 不属于模型配置。Gym 会按未知字段拒绝。 |

整数字段不接受字符串和布尔值。数值字段必须是有限数值。不要用 `null` 代替省略字段。
Gym 按以下顺序合并缺省值，后面的值优先：

1. 原生 `ModelConfig` 默认值。
2. Gym 默认值：`layers=24`、`mlp_width=6144`、`window_pattern="L"`。
3. `method.json.shell`。
4. `architecture.json`。

选项合并顺序是 `method.json.options`、debug profile 下的 `debug_options`，最后是 `architecture.json.options`。
Gym 补齐省略的 profile 固定字段。如果显式值与 profile 不同，Gym 会拒绝。

只改配置时，用原源码和新的 `architecture.json` 再次上传。
也可以向 `calibrate_candidate.grid` 传入 1–32 个完整的 architecture 对象。
每个对象替换整个 architecture，不是对父候选打补丁。
对返回的作业 ID 调用 `get_result`，读取 `derived[].candidate` 和 `derived[].cost`。
校准不会训练模型或测量吞吐。源码或配置改变后会生成新候选，新候选不能沿用父候选的检查点。

## 运行实验并提交

在 Agent 容器中使用下面的示例。先创建候选包：

```sh
python /opt/gym/make_candidate.py /workspace/candidate.zip
```

下面的 profile 名称对应合成数据示例。

```python
import json
from pathlib import Path
from nexttransformer_gym import GymClient

binding = json.loads(Path("/run/gym-auth/credential.json").read_text())
gym = GymClient(binding["endpoint"], binding["token"])
candidate = gym.submit_candidate(Path("/workspace/candidate.zip").read_bytes(), "idea-1")
quote = gym.quote_experiment(
    candidate_id=candidate["id"], profile_id="research-fixture", target_update=2
)
job = gym.run_experiment(quote_id=quote["id"], idempotency_key="run-1")
result = gym.wait_result(job["id"])
checks = gym.validate_candidate(
    candidate_id=candidate["id"], profile_id="final-fixture", idempotency_key="checks-1"
)
assert gym.wait_result(checks["id"])["admitted"]
submission = gym.finalize(
    candidate_id=candidate["id"], profile_id="final-fixture", idempotency_key="final-1"
)
Path("/workspace/submission.json").write_text(json.dumps({
    "session_id": binding["session_id"],
    "submission_id": submission["id"],
    "candidate_hash": candidate["hash"],
}))
```

SDK 的 `submit_candidate` 接收 ZIP 字节。直接调用 HTTP 时，将字节编码为 base64，传入 `package_base64`。
响应包含 `id`、`hash`、`configuration_hash`、`architecture`、`declaration`、`method`、`parent_ids` 和 `contract`。
`architecture` 是上传时的部分配置。Gym 根据所选 profile 补齐默认值。

读取结果前先检查作业状态。结果尚未生成时，`get_result` 返回 `not_found`。
执行成功的检查作业仍可能包含未通过的检查。提交前读取 `checks` 和 `admitted`。
调用 `finalize` 前，先结算或取消所有作业。账户不能有未结算的预留额度。
Gym 保留源码。验证器只检查提交回执，不启动最终训练。
资源选择示例见 [research_with_placement.py](examples/research_with_placement.py)。

## 错误与恢复

HTTP 的 `error` 对象包含 `code`、`hint`、`phase`、`retryable` 和 `details`。
SDK 抛出 `GymError`。MCP 保留原错误，并设置 `isError=true`。

| 错误码 | 含义 |
| --- | --- |
| `invalid_request` | 操作参数不符合 API schema 或请求规则。 |
| `invalid_artifact` | 包格式、模型字段名称或字段类型不正确。 |
| `incompatible_contract` | proposal 方法或路由随机种子不符合约定。 |
| `conflict` | 相同幂等键对应了不同输入，或发生其他状态冲突。 |
| `incomplete_coverage` | 静态审计失败。该路径中的原生配置或 profile 检查失败也使用此错误码。 |

保留会话的宿主状态、源码版本、profiles 和 worker Python 环境。
运行控制器命令时，使用冻结的 `<trial>.host/execution.json`。该文件已将 GPU 索引转换为 UUID。
保留原 server ID。运行 `serve` 时，还需通过 `--host` 指定原绑定地址。监听配置只保存端口。

崩溃后，先确认 worker 已退出，再运行受信任的控制器命令。将占位符替换为实际路径和 ID：

```sh
gym/.venv/bin/python -m nexttransformer_gym.cli \
  --state <retained-state> --profiles <profiles.json> --platform . \
  --execution-config <trial>.host/execution.json --worker-python <worker-python> \
  reconcile --session-id <owner-id> --job-id <job-id> --worker-confirmed-stopped
```

若其他调度器仍持有执行槽，控制器会拒绝结算。结算会扣除剩余的保守预留额度。
重复执行不会重复扣费。确认参数只声明 worker 已退出，不会终止进程。
通过 `gym/.venv/bin/python -m nexttransformer_gym.cli --help` 查看控制器命令。
命令包括 `prepare-final`、`launch-final`、`work-once`、`get-final`、`snapshot` 和 `serve`。
只有受信任的控制器命令才能分配独立额度并启动最终训练。

## 检查安装

在仓库根目录执行：

```sh
PYTHONPATH=gym/src:. NANOCHAT_DTYPE=float32 TORCHDYNAMO_DISABLE=1 \
  uv run --project gym --extra worker python -m pytest -q gym/tests
PYTEST_XDIST_AUTO_NUM_WORKERS=2 uv run --extra cpu python -m pytest -q -n auto
gym/.venv/bin/python -m nexttransformer.checks.changes --base origin/main
uvx --from ruff==0.14.10 ruff check --select F gym
uvx --from ruff==0.14.10 ruff format --check gym
uv run --project gym python gym/smoke.py
```

Smoke 使用原生 Claude Code 和本地模型桩，不调用外部模型。
如果客户端镜像没有固定版本的 CLI，Harbor 会安装它。
安装站点不可达时，通过 `--client-image <image>` 指定已包含该版本 CLI 和 Gym 客户端文件的镜像。

进行短 GPU 检查时，选择已分配的显卡：

```sh
PYTHONPATH=gym/src:. gym/.venv/bin/python gym/gpu_smoke.py \
  --state-dir .agent-work/gpu-smoke \
  --gpu <allocated-index-or-uuid> --worker-python <cuda-python>
```

这些检查验证集成行为。生产数据和最终基准成绩需要另行验收，详见[设计限制](DESIGN.zh-CN.md)。
