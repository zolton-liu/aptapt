# Agent 框架改造：上下文、记忆与验收

更新：2026-09-23。实现位于 `src/qfa_agent`，没有新增运行依赖，也没有训练模型权重。

## 开源参考与取舍

本轮直接核对了以下项目的官方仓库、源码和文档。下述为设计借鉴，新增模块为本项目实现，没有把这些项目作为依赖整体引入。

| 参考 | 公开实现中的思路 | 本项目落地 | 本轮没有引入 |
|---|---|---|---|
| [Deep Agents](https://github.com/langchain-ai/deepagents)、[上下文工程](https://docs.langchain.com/oss/python/deepagents/context-engineering) | 大工具输出移入文件，按需读取；控制不断增长的上下文 | `ContextManager`：固定任务指令、预算打包、观察摘要、可分页证据文件 | 额外模型总结调用、子 Agent 调度 |
| [LangGraph memory](https://docs.langchain.com/oss/python/concepts/memory) | 区分线程内状态与跨线程长期记忆 | `WorkingMemory`：任务隔离的证据、失败、修复状态；另置版本化通用修复规则 | 自动跨题写入经验库、向量数据库、完整断点恢复 |
| [TradingAgents](https://github.com/TauricResearch/TradingAgents)、[reflection.py](https://github.com/TauricResearch/TradingAgents/blob/main/tradingagents/graph/reflection.py) | 专业角色分工；根据实际结果生成反馈供后续参考 | 在原有 builder/executor/repair/audit 状态机上强化验收门槛；修复记录由执行证据更新 | 多空辩论、交易接口、根据收益训练策略 |
| [mini-swe-agent 执行循环](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py) | 小型可检查执行循环、预算限制和运行记录 | 保留单模型动作循环；增加上下文统计、预算提示、模型失败请求的输入预算预留 | 用大型框架替换已有工具和比赛入口 |

此前提到的 Vibe Trading 没有确认唯一仓库地址，因此本报告没有把未经核对的具体实现归因给它。以上四个来源足以支撑本轮改造。

## 上下文：从按条数截断改为按预算装配

旧逻辑在消息超过 10 条后留下三个动作/观察对。一个观察即使有几十万字符，仍可能整体进入请求；少于 10 条时也会超出模型窗口。

现在每次请求依次组装：

1. 完整系统工具规则与任务指令，两者固定保留。
2. 控制器检查点：当前阶段、剩余调用/执行/时间预算、输出文件契约、当前源码哈希、待修复错误、已验证修复和相关规则。
3. 预算内最近的完整动作—观察组，避免把 JSON 或动作/反馈关系切断。

默认输入预算为 16,000 个**估算 token**；采用 UTF-8 字节启发式，不等同于模型 tokenizer。必须给模型输出另留窗口。基础指令与必要状态自身超过预算时，明确停止并记录原因，不静默裁剪任务。可恢复的历史和事实在必要时从检查点中移除。

工具观察超过默认 6,000 字符后，写入 `scratch/.agent/observations/<hash>.json`。提示中保留成功/失败状态、错误上下文、日志首尾、下一步建议和证据路径。长字符串拆为 `join-chunks` 数组，支持现有 `read_file` 按行分页；否则一个巨大 JSON 字符串会让“按需读取”难以使用。超出单文件存储上限时明确标记截断。

文件证据按既有工作区访问限制保存，已知 canary 标识会被清理。历史工具内容标注为证据数据，其优先级低于任务与系统规则。

## 记忆：保存状态与证据，而不只保存聊天

| 记忆层 | 内容与更新条件 | 存放位置 |
|---|---|---|
| 任务工作记忆 | 输出契约、阶段、剩余预算、源码版本、最近动作 | 每次模型请求的检查点 |
| 情景记忆 | 观察来源、内容哈希、失败类型、重复次数、相关源码版本 | `WorkingMemory`；`scratch/.agent/memory.json` |
| 修复记忆 | 只有同一执行/测试/文件验证门槛成功，才将对应失败改为 resolved；保留验证时源码哈希 | 同一任务记忆；提示中最多保留近期若干条 |
| 程序性知识 | schema、路径、数值、类型、超时等通用修复建议，按当前错误类型选择 | `memory.py` 中版本化 `REPAIR_RECIPES` |

例如第一次运行出现 `KeyError: close`，会留下 active 错误。改代码只改变源码版本，不会自行把错误判为已修复；同一脚本运行成功后才标为 resolved。另一个脚本成功不会消除这个错误，Python 成功也不会消除仍然失败的 pytest 或输出契约检查。

读取源码的历史事实带有观察时的源码版本；版本变化后标为可能过时，提醒重新核对。重复错误聚合计数，避免同一 traceback 一直挤占上下文。

这不是完整长期学习系统。任务实例不会自动加载上一题的记忆；程序性知识目前是人工维护的通用规则，没有自动抽取题目答案。`memory.json` 是可审计状态快照，**尚不支持进程断点续跑**。要实现可靠续跑，还需同时恢复工具预算、执行状态和输出版本，并验证输入内容身份。

## 执行与验收

原有领域路由和 builder → execute → audit/repair → finish 状态机继续使用。本轮加入：

- 自建 pytest 失败后，文件存在/格式正确不能放行 finish。
- 一个无关测试文件通过，不能清除先前失败的测试组。
- 修改源代码或其他模型写入内容后，已运行过的测试需要重新运行；使用保守失效策略，可能增加执行次数。
- 步数耗尽时也检查未通过的测试，修复过去“预算结束 + 文件有效 = completed”的漏洞。
- 模型请求异常时预留估算输入费用，避免把失败请求当作零消耗；实际输出消耗在服务未返回 usage 时仍不可知。

领域不变量仍来自已有 strategy 提示。新增门槛只跟踪实际运行的自建测试，不能替代官方 verifier，也不能保证测试覆盖充分或公式一定正确。

## 修正确定性求解器的评测口径

上一轮 `workflow-v5.9-cliquet-trusted-operator-coder14b` 的 17/17、约 1.1 秒、零模型调用，证明预写 Cliquet 求解器能够生成通过公开 checker 的结果。当时该路径绕过了 starter 关闭开关，`starter_used=false` 的历史记录具有误导性。

本轮已经改为：完整预写适配器统一服从 `QFA_DISABLE_STARTERS`；启用并使用时记录 `starter_used=true`。历史原始结果文件保留，但这条结果应解读为**确定性求解器回归**，不能作为 starter-disabled 的模型 pass@1 提升。

最新可引用的完整公开集基线仍为 `workflow-v5.6-all87-canonical-coder14b`：11/87，12.64%。其中七条无效运行经重跑替换，因此它是整理后的开发基线；本轮尚未重新运行完整同配置模型评测，也不代表官方榜单或未见任务表现。

## 本轮验证结果

- 全套自动测试：**110 passed**，包含大输出落盘、中文上下文预算、完整动作对保留、必要指令溢出、失败去重、同门槛修复确认、任务隔离、测试门槛和 starter 开关回归。
- 87 个公开任务初始上下文静态预检：**87/87 能装入默认预算**，最大估算 11,025 token。没有调用模型或执行题目 checker。
- 人工构造的大日志压力场景：旧上下文估算 **57,352 token**，新上下文 **2,147 token**，下降 **96.26%**；保留原任务指令，三次重复错误聚合为一条记忆。
- 压力场景不是公开金融任务，也不是模型实测账单；不能把这个降幅外推为平均成本下降或 pass@1 提升。

原始可复现结果：`reports/context-memory-benchmark.json`。

```bash
PYTHONPATH=src .venv-eval/bin/python -m pytest -ra
PYTHONPATH=src .venv-eval/bin/python scripts/benchmark_context_memory.py \
  --official-repo work/track1-coding-public-full \
  --output reports/context-memory-benchmark.json
```

## 配置与下一轮效果验证

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `QFA_CONTEXT_MEMORY` | `1` | 设 `0` 使用原有上下文装配与观察方式，用于组件消融 |
| `QFA_CONTEXT_MAX_TOKENS` | `16000` | 单次请求输入的估算 token 上限 |
| `QFA_OBSERVATION_MAX_CHARS` | `6000` | 长工具观察落盘的触发阈值 |
| `QFA_DISABLE_STARTERS` | 未设置 | 正式比较模型调用策略时设 `1`，本地公开 runner 已强制设置 |

每次请求的 `context` 轨迹事件记录估算长度、丢弃的历史条数、落盘观察次数。公开 runner 配置还记录上下文开关和框架源码摘要，便于追溯；运行期间仍应冻结源码，避免不同子任务加载到不同版本。

下一轮应冻结相同任务清单、模型、采样配置、调用上限、超时和源码，分别运行 `QFA_CONTEXT_MEMORY=0/1`。这个开关只消融上下文/记忆组件，其余验收修复保持相同。对比整体 pass@1、失败重复次数、超时率、真实 token 使用和每题耗时，之后再扩大到全部 87 题。公共集已用于调试，最终泛化能力还需要未用于改造的任务验证。
