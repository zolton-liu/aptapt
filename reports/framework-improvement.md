# QFA 框架改进记录

## 最新状态（2026-09-23）

- 整理后的 87 题全量开发基线：`workflow-v5.6-all87-canonical-coder14b`，11/87（12.64%）。
- 本轮增加预算化上下文、工具观察落盘、任务隔离工作记忆、按执行证据确认的修复状态，以及自建测试验收门槛。全套测试 110 passed。
- 完整改造依据、开源来源、配置、压力测试结果和限制见 [上下文与记忆改造说明](framework-context-memory.md)。
- v5.9 Cliquet 的 17/17 是零模型调用的预写求解器回归，不能记为 starter-disabled 模型能力提升；绕过 starter 开关的问题已修复。
- 新版本尚未完成完整 87 题模型重跑。以下记录保留为历史阶段结果。

## 历史结论（2026-09-21）

- 87 题 starter-disabled 本地公开基线：`3/87`，pass@1 `3.45%`。
- 最终统一策略的首批 4 题定向回归：`4/4`，公开 checker 合计 `54/54`。
- 第二批近通过任务：Asian option `25/25`、OHLC volatility `33/33`。
- 六个已修复任务合计公开 checker：`112/112`。
- 这两组任务范围不同；`4/4` 证明四个已知失败被修复，不代表全量 pass@1 已达到 100%。
- 本地 Docker daemon 不可用，因此以上是 `local-public` runner 的公开 checker 结果，不是官方容器榜单成绩。

## 同题前后对照

| 任务 | 87 题基线 | 最终定向回归 | 结果 |
|---|---:|---:|---|
| CIR bond pricing | 27 pass / 1 fail / 7 error | 35/35 | PASS |
| Corporate action adjustment | 6/7 | 7/7 | PASS |
| Interest-rate cap/floor | 6/7 | 7/7 | PASS |
| Option put-call parity audit | agent error，0/5 | 5/5 | PASS |
| Asian option Levy/Curran | 最新旧策略 21/25 | 25/25 | PASS |
| OHLC realized volatility | 20 pass / 6 fail / 7 error | 33/33 | PASS |

最终统一回归实验：`eval_runs/workflow-v4.9-final-policy-coder14b`。
第二批回归实验：`eval_runs/workflow-v5.0-near-pass-operators-coder14b`。

## 本轮框架变化

1. 按任务族路由：把官方 `pricing`、`derivatives` 类别稳定映射到衍生品定价策略，避免关键词并列时误入因子研究。
2. 确定性金融算子：加入 Black-Scholes/Greeks、cap/floor、CIR、公司行动、期权平价以及已有 VaR 数据清洗等可复用算子。
3. 高风险策略门控：CIR 与可执行期权平价若试图重新手写整套公式，控制器要求改为调用已测试的高层算子；模型仍负责识别输入路径、生成适配器、执行和验收。
4. 执行可靠性：上下文压缩、失败编辑防重复、最后脏脚本自动执行、模型端点三次有界重试、运行时窄修复。
5. 评测与可视化：支持 87 个公开任务的增量本地 runner；仪表盘明确显示每次实验的样本量，避免把定向回归与全量 pass@1 混为一谈。
6. 输出契约预检：从 instruction 的 Output 区段提取声明的文件名；缺少任一必需文件时禁止提前 finish，并把缺失清单反馈给模型。解析器已对全部 87 题静态审计，87/87 均识别到契约，且未把 `params.json`、`task_rules.json` 等输入文件误判为输出。
7. 第二批任务族算子：加入离散 Asian（几何闭式、Levy、Curran 条件积分、共享路径 MC）与五类 OHLC 波动率估计器。

框架测试：`90 passed`。

## 旧版 87 题暴露问题的系统修复

旧基线的失败构成为 56 个 `agent_incomplete`、19 个 `verifier_failure`、8 个
`agent_error`、1 个超时和 3 个通过。除任务专属金融公式外，已统一处理以下框架问题：

1. 修改—运行循环耗尽轮数：`solve.py` 写入或修补后由控制器自动执行并立即做输出契约检查。
2. 无修改重复执行：同一源码失败后禁止原样重跑；A→B→A 的源码振荡也会在执行前拦截。
3. 过浅的完成条件：输出文件契约覆盖 87/87，并检查 JSON 有限值、CSV/TSV 表头、重复列、列数、非有限标记及常见二进制签名。
4. 契约误判：输出章节按标题层级截断，只读取声明行；输入文件引用、`reward.json` 等 verifier 自有文件不再被当作提交物。
5. 二进制输入：Parquet 和 XLSX 在首轮 prompt 与 `read_file` 中返回 schema、行数、sheet、表头和样例，而不是抛出二进制读取错误。
6. 现成模板未复用：新增受限的 `copy_file`（仅 visible input → scratch），并自动把模板中的 `/app/data`、`/app/output` 正规化为运行时环境路径。
7. 模型服务错误：端点内部三次有界重试后，solve 循环还可在不改变工作区的情况下恢复一次，不再直接变成 `agent_error`。
8. 上下文摘要误用：模型若把 `content_record`/`old_record` 当源码提交，会自动转换成读取当前文件，而不是反复无效写入。
9. 路由不稳定：补齐 execution、factor-models、fixed-income-nlp、extreme-value-theory、cross-currency-rates 等公开 card 类别映射。

系统回归 `workflow-v5.2-parquet-cycle-fix-coder14b` 已将此前 agent error 的官方
Parquet 示例修复到 `14/14`。双曲线 bootstrap 与 ETF overlap 的框架故障已被拦截并给出
可修复反馈，但其任务专属金融计算仍未通过，不能记入 pass@1。

## 为什么暂未使用 RL

现阶段只有 87 个一次性、稀疏二元奖励样本，且缺少同一状态下的候选动作/反事实回报。直接做策略梯度或 LoRA 不仅信号不足，也不符合当前只允许组织方 house model 的比赛约束。本轮采用的是更可验证的“verifier 奖励驱动程序优化”：

1. 将失败分为路由、数值、schema、运行时与模型服务故障；
2. 把可重复的金融数值部分下沉为测试过的算子；
3. 用策略门控让小模型选择可靠算子，而不是每题重新发明公式；
4. 在固定任务回归集上重新采样验证。

后续若积累了足够轨迹，可做 contextual bandit：状态是任务类别、输入 schema 和历史失败类型，动作是 solver policy/operator，奖励是 verifier pass、token 成本与耗时的组合。它比当前直接训练模型权重更符合比赛限制。

## 下一步

- 用最终策略重新跑完整 87 题，得到可与 `3/87` 严格比较的新全量 pass@1。
- 优先处理剩余近通过任务和 `agent_incomplete` 占比最高的任务族；Asian option 与 OHLC 已从该队列移除。
- 正式 Docker/house endpoint 可用后，用官方 harness 重验，不把 `local-public` 当作最终成绩。
