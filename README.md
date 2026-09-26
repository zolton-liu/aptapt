# aptapt

Agenthon T1 量化金融编码 Agent 框架。

一个面向 Agenthon 2026 Track 1 的最小、可审计量化编码 Agent。它不是交易机器人，也不是
SWE-bench 仓库修复器：T1 要求 Agent 读取每个量化任务的说明和数据，生成/执行解法，并把该题
指定的 JSON、CSV、Parquet、Python 或 HTML 交付物写入输出目录；隐藏 `pytest` 与金融不变量
只会在 Agent 退出后由官方 verifier 执行。

当前实现遵循官方公开的 interface `2.0`：

```text
solve --task-dir /input --out /app/output
```

官方资料：

- [Track 1 公共仓库](https://github.com/Agenthon-2026/track1-coding-public)
- [Submission CLI 合约](https://github.com/Agenthon-2026/track1-coding-public/blob/main/SUBMISSION_CLI.md)
- [官网提交描述符](https://www.agenthon.net/guides/submission-format/)
- [正式规则](https://www.agenthon.net/rules/)

## 它做什么

```text
instruction.md + card.toml + environment/data/*
                    │
                    ▼
      读取任务并移除 canary/banner
                    │
                    ▼
       读取 card 类别 + instruction 特征
                    │
                    ▼
  十类金融路由（定价/固收/风险/FX/因子等）
                    │
                    ▼
 单 House model + 确定性角色状态图（不额外调用模型）
       构建 → 执行 → 审计 ↔ 修复 → 完成
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
  检查输入      写 scratch/     写 output/
                    │
                    ▼
         受限执行 Python / 自测
                    │
                    ▼
      通用格式检查 + 任务自定义检查
                    │
                    ▼
        题目指定的最终交付物
```

外层仍保留 mini-swe-agent / mini-coding-agent 的有限循环、结构化工具、独立子进程、错误反馈
和轨迹；内层会按官方十类任务切换求解架构、金融不变量、检查预算和修复策略。参考公开金融
Agent 后又加入显式状态图、确定性失败分类、重复失败升级、最终风险闸门和带 SHA-256/表形状的
交付物 provenance，并提供经过公开 checker 工程验证的金融计算算子。当前覆盖有序收益面板、
历史 VaR、Fama-French/Newey-West、隐含波动率近似、几何均值回复跳扩散、美式期权有限差分、
方差互换复制及 Asian option Levy/Curran。模型只生成短任务适配器，工具策略拒绝无证据探索、
重复执行和不合规的现场重写。运行前还生成有界 CSV/JSON 数据概况，减少模型把 House 请求浪费
在重复读取 header。这里没有照搬昂贵的多 Agent 对话，而是让一个 House model 依次承担构建者、
执行者、修复者和风控审计者角色；仍不加入 UI、模型权重训练、任意 Bash 或公网检索。

## 安全边界

- `input/` 永远只读；即使本地 public unit 包含 `checks/` 和 `reference_data/`，工具层也拒绝读取。
- 模型只能写 `scratch/` 和 `output/`，路径经真实路径解析，阻止 `..` 和符号链接逃逸。
- 不提供 shell；只允许运行 Agent 自己写出的 `.py` 文件和自建 pytest。
- 子进程有超时、进程组清理和有界日志。
- 禁止生成 `reward.json`、`reward.txt`、`pytest_report.json`。
- 输出前检查空文件、JSON 非有限值、CSV header、Python 语法、Parquet magic 和 canary 复制。
- 正式输出目录中不写调试 trace，避免额外文件破坏某题的输出约束。

这些检查只是框架护栏，不能替代每题隐藏的金融不变量。Agent 的最佳策略仍然是根据 instruction
写一个小型自检程序，例如验证权重约束、无未来函数、put-call parity 或现金/持仓恒等式。

## 本地快速演示

### 日常开发：本地模型 + Git 检查点

当前开发分支是 `dev/local-verifier`；`checkpoint/v6.2-submitted-20260924` 保留已提交比赛版本。
当前修复版本为 **v0.8.1**，基于 v0.8.0 checkpoint 增加 House endpoint 的 thinking
关闭与兼容清理；v0.8.0 对应固定十题本地公开集严格 pass@1 为 **8/10**，完整审计见
[v6.8 结果报告](reports/v6.8-random10-20260925-results.md)。
恢复的 v6.3–v6.5 检查点见 [版本恢复记录](reports/version-recovery-20260924.md)。
它们是从现存副本恢复的提交，不代表完整的历史编辑记录；没有可靠 v5.6 源码，不能精确回退。

本地配置在 `configs/local-dev.toml`：使用已安装、固定 digest 的 `qwen2.5-coder:14b`，
不自动下载模型，不回退到云端，也不改变官方 `solve` 的 House 配置。

```bash
.venv-eval/bin/python scripts/dev_local.py check
.venv-eval/bin/python scripts/dev_local.py smoke
```

`check` 不生成；`smoke` 只验证一次小型真实模型响应，不是金融题评测。
完整评测先完成代码测试、审查并提交 Git，再选定题单和一个全新的实验名：

```bash
PYTHONPATH=src .venv-eval/bin/python -m pytest
git status --short
# 审查后仅添加所需代码、测试、配置和摘要，再 git commit。
.venv-eval/bin/python scripts/dev_local.py eval \
  --experiment local-verifier-first-batch \
  --task-list configs/random10-20260923.txt
```

开发入口拒绝未提交改动和已有实验名；从 Git commit 导出源码，冻结题单，并让 runner 冻结
输入及 checker。配置、提交号、模型 digest、逐文件哈希存入 `work/local-runs/<实验名>/provenance.json`，
同时写进 `eval_runs/<实验名>/run_config.json`。后续编辑主目录不改变该批正在执行的源码。
本地基线默认采用兼容执行模式：模型写普通可执行脚本，在写出前做结构、计算与独立交叉断言，
框架执行脚本并检查实际产物；通过后直接结束。严格 `AuditReport` 四阶段协议仍保留为可选实验，
但历史 v6.3/v6.4 表明 14B 模型的协议采用率为零，因此不再作为本地基线的强制门槛。
只读的已验证经验仍启用；旧回放、其他模型适配器及遗留 QFA 配置不会混入本地实验。
这只是本地公开集口径，不是官方 House 模型或隐藏集成绩。原始轨迹和提交凭据不进入 Git。

### 无模型演示

演示使用固定模型响应，不需要网络或模型：

```bash
PYTHONPATH=src \
QFA_REPLAY_FILE=examples/demo_responses.json \
python -m qfa_agent.cli solve \
  --task-dir examples/demo_task \
  --out outputs/demo
```

查看结果：

```bash
python -m json.tool outputs/demo/results.json
```

每次运行要求输出目录为空：

```bash
rm -r outputs/demo
```

## 连接官方 house endpoint

正式环境由组织方注入：

```text
MODEL_ENDPOINT  官方 House origin，例如 http://model:8443（不带 path）
MODEL_NAME      固定模型 id
MODEL_TOKEN     每个 unit 注入的 Bearer token
QFBENCH_SEED    本次可重复性 seed
HTTP_PROXY      审计代理（urllib 自动读取）
HTTPS_PROXY     审计代理（urllib 自动读取）
NO_PROXY        必须绕过代理的 host
```

实现调用 `$MODEL_ENDPOINT/v1/chat/completions`，发送 `Authorization: Bearer $MODEL_TOKEN`，
并在每次请求中设置 `chat_template_kwargs: {"enable_thinking": false}`；若兼容服务仍返回
完整的 `<think>...</think>` 前缀，也会在解析工具 JSON 前将其移除。
不会添加 vendor tools、网页搜索、远程代码执行或检索。只有显式存在 `MODEL_ENDPOINT` 与
`MODEL_NAME` 时才会联网；正式 restricted runtime 缺少 token 会直接失败。开发时还可使用：

- `QFA_REPLAY_FILE=/path/responses.json`：确定性回放。
- `QFA_MODEL_COMMAND='local-adapter --json'`：messages JSON 从 stdin 输入，动作 JSON 从 stdout 输出。
- `QFA_JSON_MODE=1`：请求 OpenAI-compatible 服务返回严格 JSON；适合本地 Ollama 调试，
  正式 endpoint 不支持 `response_format` 时不要设置。

重要调节项：

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| `QFA_MAX_STEPS` | 20 | 最大模型轮数 |
| `QFA_MAX_PYTHON_RUNS` | 8 | 最大 Python 执行次数 |
| `QFA_MAX_PYTEST_RUNS` | 3 | 最大自建测试次数 |
| `QFA_MODEL_TIMEOUT_SEC` | 300 | 单次模型调用上限 |
| `QFA_MAX_RESPONSE_TOKENS` | 4000 | 单次响应 token 上限（官方硬上限） |
| `QFA_RESERVE_SEC` | 45 | 为最终检查/退出保留的时间 |

模型调用硬限制为每题最多 25 次；累计输入 1,000,000 tokens、累计输出 100,000 tokens。
总墙钟预算来自每题 `card.toml` 的 `[agent].timeout_sec`，不会把 verifier timeout 当作 Agent
预算。上述官方上限均已在配置中封顶，环境变量只能调低、不能突破。

## 构建比赛镜像

先取得官方 public repo 并构建它的金融基础镜像：

```bash
git clone https://github.com/Agenthon-2026/track1-coding-public.git
cd track1-coding-public
docker build -t finance-bench-sandbox:latest -f docker/sandbox.Dockerfile .
```

再回到本项目：

```bash
docker build \
  --build-arg AGENT_VERSION=0.8.1 \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -t agenthon-t1-minimal:0.8.1 .
```

`Dockerfile` 继承官方 Python 3.13 金融栈，并包含强制 label：

```text
qfbench2.interface_version=2.0
```

先用项目自带的 demo 测 Docker 入口（固定回放只验证框架，不代表真实解题能力）：

```bash
ROOT=/absolute/path/to/this/project
UNIT="$ROOT/examples/demo_task"
OUT=/tmp/qfa-output
mkdir -p "$OUT"

docker run --rm --network=none \
  -e QFA_REPLAY_FILE=/fixtures/demo_responses.json \
  -v "$UNIT:/input:ro" \
  -v "$ROOT/examples/demo_responses.json:/fixtures/demo_responses.json:ro" \
  -v "$OUT:/app/output" \
  -v "$OUT:/output" \
  agenthon-t1-minimal:0.8.1 \
  solve --task-dir /input --out /app/output
```

正式 harness 的实际调用由官方负责；不要在镜像中硬编码 proxy、模型名称或任务文件名。

## 测试

核心测试只依赖标准库：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

覆盖协议解析、canary 清理、只读/隐藏 checks、路径逃逸、事务式替换、有界执行、超时进程组、
输出格式检查、完整 Agent 回放循环和官方 descriptor digest fixture。

## 真实能力评测与训练曲线

领域 starter 的回归通过率不能代表模型能力。使用 `QFA_DISABLE_STARTERS=1` 可关闭所有 starter，
让评测只测模型、prompt、工具协议和执行循环。项目提供一个冻结的 10 题跨类别切片，以及会保存
逐题 verifier、sanitized trajectory、tokens、耗时与失败类型的批量运行器：

```bash
python scripts/evaluate_public.py \
  --official-repo /absolute/path/to/track1-coding-public \
  --experiment baseline-v0-qwen25-7b \
  --model-name qwen2.5:7b \
  --max-steps 12 \
  --json-mode
```

每个新 Agent 策略版本使用新的 `--experiment` 名称，但保持任务清单、seed、模型预算和
verifier 不变。生成跨版本训练效果仪表盘：

```bash
python scripts/render_eval_dashboard.py \
  --input-root eval_runs \
  --output reports/training-effect.html
```

重点比较未见任务的 pass@1，而不是已经加入 starter 的任务。公开 unit 带有 contamination canary，
只能用于评测与调试，公开评测集必须保持冻结。2026-09-18 规则已经取消 BYO 模型和 LoRA；
当前优化对象是 prompt、任务路由、工具循环、执行验证和修复策略。

## 生成 submission.json

复制 [`submission.example.json`](submission.example.json)，填写真实 team、镜像 digest 和组织方
公布的模型 pin，然后封印：

```bash
PYTHONPATH=src python -m qfa_agent.descriptor submission.json
PYTHONPATH=src python -m qfa_agent.descriptor submission.json --check
```

描述符工具会移除旧 `descriptor_digest`，对其余字段使用紧凑、排序 canonical JSON 计算 SHA-256，
并原子写回。浮动 Docker tag 不能替代 `image.digest`。

## 当前边界

这是可运行 MVP，不是有竞争力的最终方案：

1. 通用验证只能发现文件级错误，无法看到正式隐藏 pytest。
2. 模型必须从自然语言中准确推导每题输出 schema 和金融约定。
3. 已有十类任务路由、领域不变量和数据概况，但还没有为每类建立稳定的 few-shot 回归库。
4. 公共任务中的旧 `/app/...` 路径与新 `/input/...` 挂载并不总一致；工具会映射常见 legacy
   路径，但 prompt 仍要求先检查真实 inventory。
5. 正式提交前必须用 public repo 的完整任务集、官方 scorer 和真实 house endpoint 迭代。

下一步最有价值的改进是：为每类加入少量通用计算骨架；让模型生成任务专属 invariant tests；
在全部 public units 上按失败类别建立回归集，并在真实 House endpoint 可用后重新校准轮数与输出长度。

当前里程碑（2026-09-25）：固定十题、固定 `qwen2.5-coder:14b`、starter 关闭、
失败题不重跑的 v6.8 整体回归为 **8/10（80%）**，相比完整 v6.6 基线 2/10 净增 6 题。
10 个唯一 ID、原始 checker、退出码、输入/源码哈希和 summary 已完整核对；详见
[v6.8 结果报告](reports/v6.8-random10-20260925-results.md)。剩余失败集中在 local-vol 的修复协议
混淆，以及 DCC-GARCH 的 4,000-token 截断响应循环。该结果来自反复观察过的本地公开题，
不是官方 House/隐藏集成绩，不能直接外推到未见题。

上一里程碑（2026-09-23）：整理后的 87 题本地公开开发基线为
`workflow-v5.6-all87-canonical-coder14b`，11/87（12.64%）。新增预算化上下文、
长观察落盘、带源码版本和验证状态的任务记忆、通用修复规则检索，以及自建测试失败门槛。
完整测试 110 passed，87/87 初始任务上下文可装入默认预算。
新版本尚未重新测定完整 pass@1；零模型调用的 Cliquet 专用求解结果属于确定性回归。
配置与开源参考见 [上下文与记忆改造说明](reports/framework-context-memory.md)。

历史里程碑（2026-09-21）：starter 关闭的 87 题本地公开基线
`workflow-v4.4-all87-coder14b` 为 3/87（3.45%）。基于 verifier 失败归因加入类型路由、
确定性金融算子与高风险任务策略门控后，最终定向回归
`workflow-v4.9-final-policy-coder14b` 在 CIR、公司行动、利率 cap/floor、期权平价 4 题上取得
4/4。随后 `workflow-v5.0-near-pass-operators-coder14b` 将 Asian option 与 OHLC volatility
分别提升至 25/25、33/33。六个已修复任务的公开 checker 合计 112/112。定向回归的 100%
不能替代新的 87 题全量 pass@1。输出契约预检已覆盖 87/87 个公开任务；补齐自动执行、
防重复运行/源码振荡、Parquet/XLSX 概要、模板复制与路径正规化、模型错误恢复后，框架测试为
90 passed。官方 Parquet 示例的 starter-disabled 回归为 14/14；完整对照见
`reports/framework-improvement.md`。
