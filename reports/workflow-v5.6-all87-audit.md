# workflow-v5.6：87 题全量结果审计

审计日期：2026-09-23。以下是已有本地公开集数据的只读复核，没有重新执行题目或更改原始成绩。

## 成绩与口径

- 原始全量运行：87 个唯一 ID，11 题通过，记录的严格工作流 pass@1 为 **11/87（12.64%）**。
- 整理版：同样 87 个唯一 ID、11 题通过；题目集合与公开目录完全一致，87 条 `starter_used` 均为 false。
- 整理版全部 87 条 reward 均与 `agent_exit_code == 0 && verifier_exit_code == 0` 一致；checker-only 通过数也为 11。
- 分母是题目数，不是单项 checker 数。一次 rollout 可以包含多轮模型调用与内部修复。
- 本地公开 checker 不是官方隐藏集；题目已被开发过程使用，不能把成绩当作未见题泛化能力。

## 为什么同时保留原始版与整理版

原始版有 7 条 `agent_error`，退出码全部为 **-15（SIGTERM）**，且均早于每题 900 秒上限。这证明它们曾被信号终止，不能仅凭分类字段认定为模型或代码自行报错；现有退出码和空 stderr 不能证明是谁发出的信号。原始版因此也不是完全无干预的受控评测。

整理版用另外一个 7 题实验替换了这些记录。7 题原来全失败、替换后也全失败，所以通过数未增加。整理版是开发基线，不是一次独立无重跑的 87 题实验。已逐字段核对 merge_provenance：除实验名改为整理版名称外，每条记录都与指定来源完全一致。

| 版本 | 通过 | Agent 未完成 | 外层超时 | Agent 错误 | checker 失败 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始版 | 11 | 41 | 15 | 7 | 13 |
| 整理版 | 11 | 46 | 17 | 0 | 13 |

7 个替换 ID：
- `t1-american-option-fd-new`
- `t1-bs-greeks-pde`
- `t1-cliquet-ratchet-pricing`
- `t1-cme-hdd-option-pricing`
- `t1-compound-option-geske`
- `t1-copula-equity-fitting`
- `t1-copula-sampling-rank-correlation`

早停和超时均保留为失败，没有从分母中删除。后续若需要严格的统一框架效果估计，应采用冻结源码的新实验，不再将择优或替换记录混入原实验。

## 整理版任务族分布

优先按记录中的 routed_category 分组，缺失时使用题卡 category。分组描述任务覆盖，不证明类别之间难度可比。

| 路由任务族 | 题数 | 通过 | 未完成 | 超时 | checker 失败 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cross-domain | 21 | 1 | 14 | 3 | 3 |
| derivatives-pricing | 20 | 6 | 10 | 2 | 2 |
| risk-management | 11 | 2 | 6 | 1 | 2 |
| factor-research | 10 | 1 | 5 | 0 | 4 |
| fixed-income | 5 | 0 | 2 | 1 | 2 |
| backtesting | 4 | 1 | 3 | 0 | 0 |
| fx | 3 | 0 | 3 | 0 | 0 |
| execution | 2 | 0 | 0 | 2 | 0 |
| credit | 2 | 0 | 2 | 0 | 0 |
| dependence-modeling | 1 | 0 | 0 | 1 | 0 |
| credit-risk | 1 | 0 | 0 | 1 | 0 |
| crypto | 1 | 0 | 0 | 1 | 0 |
| risk-modeling | 1 | 0 | 0 | 1 | 0 |
| cross-asset-analysis | 1 | 0 | 0 | 1 | 0 |
| factor-models | 1 | 0 | 0 | 1 | 0 |
| fixed-income-nlp | 1 | 0 | 0 | 1 | 0 |
| debug-migration | 1 | 0 | 0 | 1 | 0 |
| nlp-on-finance | 1 | 0 | 1 | 0 | 0 |

## 耗时与用量

- 原始版累计记录的 Agent 耗时：12.70 小时，平均 525.71 秒/题；输入/输出 tokens：7256664 / 334806。
- 整理版累计记录的 Agent 耗时：13.38 小时，平均 553.54 秒/题；输入/输出 tokens：7581441 / 367072。
- 整理版累计时间由选入的记录相加，不是一次连续运行的墙钟时间，也不包含被替换记录的额外成本。未完成请求可能没有 usage，因此 token 汇总不是精确账单。

## 逐题结果（整理版）

星号表示该题来自 7 题替换实验。checker 列依次为通过/失败/错误的测试项数。

| 题目 | 结果 | checker 通过/失败/错误 | Agent 秒数 | 模型调用 |
| --- | --- | ---: | ---: | ---: |
| t1-13f-amendment-aware-crowding | 未完成 | 0/11/40 | 190.37 | 18 |
| t1-EXAMPLE-bs-greeks-pde | 通过 | 14/0/0 | 98.51 | 7 |
| t1-alpha-hedge-strategy | 未完成 | 0/1/0 | 502.28 | 18 |
| t1-american-option-fd-new * | 超时 | 0/54/0 | 900.01 | 8 |
| t1-asian-option-levy-curran | 通过 | 25/0/0 | 98.78 | 7 |
| t1-barone-adesi-whaley | 超时 | 3/0/33 | 900.01 | 17 |
| t1-barrier-garch-var | 未完成 | 0/1/0 | 497.29 | 18 |
| t1-binance-btc-participation-tca | 超时 | 0/5/0 | 900.01 | 17 |
| t1-bl-regime-hmm | 未完成 | 0/1/0 | 689.14 | 18 |
| t1-bollinger-backtest-aapl | 未完成 | 10/25/0 | 552.56 | 18 |
| t1-brinson-sector-attribution | 未完成 | 0/0/42 | 603.24 | 18 |
| t1-bs-greeks-pde * | 未完成 | 5/0/34 | 701.51 | 18 |
| t1-cir-bond-pricing | 未完成 | 0/2/33 | 462.17 | 18 |
| t1-cliquet-ratchet-pricing * | 未完成 | 2/15/0 | 506.89 | 18 |
| t1-cme-hdd-option-pricing * | 未完成 | 0/52/0 | 836.37 | 18 |
| t1-compound-option-geske * | 未完成 | 6/28/0 | 727.17 | 18 |
| t1-copula-equity-fitting * | 未完成 | 0/0/28 | 572.06 | 18 |
| t1-copula-sampling-rank-correlation * | 超时 | 0/0/51 | 900.02 | 15 |
| t1-corporate-action-adjustment | 通过 | 7/0/0 | 176.00 | 12 |
| t1-credit-migration-matrix | 未完成 | 0/8/99 | 551.88 | 18 |
| t1-credit-portfolio-var-cvar | 超时 | 0/12/118 | 900.01 | 10 |
| t1-credit-spread-decomposition | 未完成 | 0/0/39 | 751.79 | 18 |
| t1-creditmetrics-portfolio-var | 超时 | 0/0/26 | 900.01 | 12 |
| t1-cross-sectional-momentum | checker 失败 | 23/14/0 | 373.97 | 4 |
| t1-crypto-funding-rate-basis-carry | 超时 | 0/63/0 | 900.01 | 13 |
| t1-cta-basel-capital | 未完成 | 0/1/0 | 527.73 | 18 |
| t1-dcc-garch-portfolio-var | 超时 | 0/63/0 | 900.01 | 10 |
| t1-delta-hedging-pnl-simulation | 未完成 | 0/27/0 | 454.74 | 18 |
| t1-digital-barrier-options | checker 失败 | 21/14/0 | 415.83 | 9 |
| t1-double-sort | checker 失败 | 1/2/16 | 187.19 | 12 |
| t1-dupire-local-vol | 超时 | 11/13/44 | 900.01 | 14 |
| t1-earnings-surprise-calculator | 通过 | 8/0/0 | 100.78 | 5 |
| t1-etf-cross-asset-lead-lag | 超时 | 0/24/0 | 900.01 | 14 |
| t1-etf-overlap-redemption-pressure | 未完成 | 0/1/39 | 775.01 | 18 |
| t1-event-study-earnings | 未完成 | 0/34/0 | 493.44 | 18 |
| t1-evt-pot-var | 未完成 | 0/2/53 | 693.30 | 18 |
| t1-ewma-portfolio-risk-decomposition | checker 失败 | 2/25/0 | 193.38 | 12 |
| t1-fama-french-factor-model-new | 超时 | 0/101/0 | 900.01 | 16 |
| t1-fft-compound-poisson | checker 失败 | 41/22/0 | 131.71 | 3 |
| t1-first-passage-time | checker 失败 | 14/17/0 | 828.76 | 14 |
| t1-fomc-tone-event-study | 超时 | 0/20/0 | 900.01 | 13 |
| t1-form4-cross-sectional-sale-pressure | 未完成 | 0/7/47 | 173.66 | 18 |
| t1-fx-carry-forward-hedge | 未完成 | 1/28/16 | 685.49 | 16 |
| t1-fx-forward-cross-rate | 未完成 | 0/0/37 | 710.46 | 16 |
| t1-geometric-mean-reverting-jd | checker 失败 | 22/12/0 | 374.37 | 11 |
| t1-historical-var-data-prep | 通过 | 12/0/0 | 132.33 | 8 |
| t1-hull-white-swaption | 未完成 | 0/7/74 | 655.27 | 3 |
| t1-implied-vol-approximations | 未完成 | 8/2/0 | 863.09 | 18 |
| t1-interest-rate-cap-floor | checker 失败 | 5/2/0 | 69.68 | 4 |
| t1-intraday-volume-fitting-and-execution-scheduling | 超时 | 1/12/0 | 900.02 | 15 |
| t1-ipca-latent-factors | 未完成 | 0/1/0 | 490.76 | 18 |
| t1-kelly-var-sizing | 未完成 | 0/1/0 | 437.51 | 18 |
| t1-lob-pc-signal | 未完成 | 0/31/0 | 776.90 | 18 |
| t1-localvol-barrier | 未完成 | 0/7/32 | 782.53 | 7 |
| t1-lookback-options | 通过 | 25/0/0 | 596.55 | 6 |
| t1-mc-greek-surface-1 | 超时 | 0/8/41 | 900.02 | 8 |
| t1-merton-jump-diffusion | 未完成 | 0/0/28 | 779.67 | 18 |
| t1-momentum-backtest | 未完成 | 8/18/0 | 538.57 | 18 |
| t1-mtm-xccy-basis-desk | 未完成 | 4/141/0 | 497.45 | 16 |
| t1-multimodal-alpha-fusion-edgar-cot-gdelt | 未完成 | 0/0/26 | 244.61 | 18 |
| t1-ohlc-realized-vol-estimators | 通过 | 33/0/0 | 364.41 | 10 |
| t1-option-put-call-parity-forward-audit | 通过 | 5/0/0 | 133.59 | 8 |
| t1-ou-jump-commodity | 未完成 | 1/25/0 | 476.81 | 18 |
| t1-pca-factor-portfolio | 通过 | 8/0/0 | 109.56 | 4 |
| t1-polars-api-migration | 超时 | 0/57/0 | 900.01 | 12 |
| t1-prediction-markets-cross-venue-dislocation | 超时 | 10/15/0 | 900.02 | 13 |
| t1-realized-vol-estimators | checker 失败 | 68/73/0 | 179.74 | 8 |
| t1-regime-cta-vol-target | 未完成 | 0/1/0 | 741.31 | 18 |
| t1-regime-riskparity-cvar | 未完成 | 0/1/0 | 381.66 | 18 |
| t1-residual-momentum | 未完成 | 0/0/33 | 524.00 | 18 |
| t1-sec-10k-report-long | checker 失败 | 1/34/0 | 98.29 | 6 |
| t1-sec-8k-event-alpha | 未完成 | 0/1/0 | 720.81 | 18 |
| t1-sentiment-factor-alpha | 未完成 | 0/1/0 | 468.92 | 18 |
| t1-sma-crossover-spy | 未完成 | 8/2/10 | 419.40 | 18 |
| t1-smith-tail-index | 未完成 | 17/10/6 | 570.78 | 18 |
| t1-spread-option-kirk-margrabe | 未完成 | 25/2/8 | 629.09 | 18 |
| t1-stable-residual | checker 失败 | 8/11/0 | 212.29 | 7 |
| t1-standard-var-methods | 未完成 | 7/0/35 | 740.17 | 18 |
| t1-stochvol-implied-surface-new | 未完成 | 26/38/0 | 723.31 | 18 |
| t1-structured-note-risk | 未完成 | 0/1/0 | 683.77 | 18 |
| t1-swap-curve-bootstrap-ois | checker 失败 | 9/10/0 | 48.25 | 3 |
| t1-var-es-estimation | 未完成 | 6/9/32 | 667.58 | 18 |
| t1-variance-swap-replication | 通过 | 17/0/0 | 489.40 | 14 |
| t1-yield-curve-bond-immunization | 未完成 | 0/8/33 | 478.90 | 18 |
| t1-yield-curve-bootstrap-immunization | 超时 | 0/52/0 | 900.02 | 8 |
| t1-yield-curve-pca-dynamics | checker 失败 | 25/8/0 | 369.34 | 10 |
| t1-zero-coupon-bootstrapping | 通过 | 6/0/0 | 123.34 | 8 |

## 已识别问题与当前验证状态

1. 控制流和完成能力是主要瓶颈：整理版有 46 题未完成、17 题超时。但这些是终态分类，不能把每题根因都归为同一个框架错误。
2. 已结束但 checker 失败的 13 题说明结构性输出校验不足以证明金融计算、单位、schema 和边界处理正确。
3. 后续 v6 固定 10 题仅 1/10，通过率较旧版同题 4/10 退步；不能用单元测试通过替代模型效果验证。
4. 根据轨迹已修改时间预算传递、重复证据重放、上下文和失败记忆、语法编辑保护及执行完成门槛。见 [v6.1 改进与固定对照计划](recovery-v6.1-20260923.md)。
5. v6.1 固定 10 题对照已结束，严格 pass@1 为 2/10；高于 v6 的 1/10，低于 v5.6 同题 4/10。模型、题单、每题外层预算不变，但整体框架多组件一起修改，不是单组件因果实验。见 [最终效果与完成审计](recovery-v6.1-random10-results.md)，不把这 10 题成绩外推为新版 87 题成绩。

## 轨迹中的系统性失败模式

已读取整理版对应的全部 87 份来源轨迹，无缺失。下表只按工具事件自身的 summary 文本匹配，不读取或重复统计嵌套观察。一个题目可以命中多个模式，表格不能相加为失败题数。

| 观察到的模式 | 匹配事件数 | 涉及题数 | 其中最终失败题数 |
| --- | ---: | ---: | ---: |
| 失败脚本未修改就再次请求执行 | 200 | 60 | 54 |
| 编辑恢复为已执行过的源码版本 | 46 | 26 | 26 |
| 文本替换匹配不存在或不唯一 | 75 | 44 | 39 |
| 语法验证/语法错误摘要 | 55 | 22 | 20 |
| 输出契约验证失败 | 43 | 13 | 13 |

这些数量反映控制器可见的问题，不能直接推断金融公式错误的频率。旧轨迹没有保存完整工具输出；failure_kind 又来自状态机的最近错误，可能延续到后续事件，因此未把它当作逐事件独立根因统计。

另对原有 11 道通过题检查了按工具名和参数哈希计数的“无 mutation 的重复动作”：各题最大值均为 1，没有发现达到新阈值 3 次的旧成功轨迹。这个离线检查只能说明旧成功轨迹未直接触发该阈值，不能证明新模型行为不会发生退步。

## 可复现性限制

旧原始版和 7 题替换版 run_config 记录相同模型名称、18 步、4000 输出 tokens、360 秒模型请求限制和 900 秒外层任务限制，但没有保存框架源码哈希或模型 digest。因此不能从旧配置证明它们使用完全一致的源码和模型权重；成绩只描述留存运行记录。

新 v6.1 实验额外保存源码、runner 哈希、完整有序题单与预算传递标识，报告登记了模型 digest。运行期间源码保持不变，最终复核当前源码与记录哈希一致。

## 证据入口

- [原始 87 题结果](../eval_runs/workflow-v5.6-all87-systemic-coder14b/results.jsonl)
- [原始运行配置](../eval_runs/workflow-v5.6-all87-systemic-coder14b/run_config.json)
- [7 题替换实验](../eval_runs/workflow-v5.6-invalid7-rerun-coder14b/results.jsonl)
- [整理版结果](../eval_runs/workflow-v5.6-all87-canonical-coder14b/results.jsonl)
- [整理版来源](../eval_runs/workflow-v5.6-all87-canonical-coder14b/merge_provenance.json)
- [v6 固定随机 10 题报告](random10-20260923-results.md)

原始 eval_runs 目录仅在本地留存，未上传 GitHub；远程查看时部分相对链接可能不可用。
