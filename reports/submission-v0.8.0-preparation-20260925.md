# aptapt v0.8.0 上传准备（2026-09-25）

## 冻结源码

- 分支：`dev/local-verifier`
- 发布源码提交：`cd6e95b8ac7bde3b890848feadaa829f04911199`
- 检查点：`checkpoint/v6.8-upload-ready-20260925`
- Python 包版本：`0.8.0`
- 固定十题本地公开集严格 pass@1：8/10；完整审计见
  `reports/v6.8-random10-20260925-results.md`。

发布提交与检查点已经推送到 GitHub。旧的
`checkpoint/v6.2-submitted-20260924` 和已提交镜像均未修改。

## 验证

- 完整测试：280 passed，42.33 秒。
- `qfa_agent.__version__`、wheel METADATA 和 `pyproject.toml` 均为 0.8.0。
- wheel 构建成功：`agenthon_t1_minimal-0.8.0-py3-none-any.whl`。
- CLI `solve --help` 检查通过。
- Docker 禁网、只读输入、固定回放 smoke 通过；生成一个合法
  `results.json`，内容为预期收益 `[0.1, -0.05]`。

这些是发布工程检查，不是额外的 pass@1、官方 House 评测或隐藏集成绩。

## Docker 镜像

- 本地标签：`aptapt-agent:submission-v0.8.0-20260925`
- 远程标签：`zoltonton/aptapt:submission-v0.8.0-20260925`
- 上传时使用的不可变引用：
  `docker.io/zoltonton/aptapt@sha256:9473a6f5458a146c580484be4673a34785729e1bfdc3ba1017d619c8cb20e584`
- 平台：`linux/amd64`
- 镜像大小：476,705,105 bytes
- 基础镜像：
  `aptapt-finance-base:submission-20260924@sha256:c2e5e02c79d4740b7006f30f5dd22fdcbfcb875dd47f33400bf614bb3ce421e8`
- 强制接口标签：`qfbench2.interface_version=2.0`
- OCI 版本标签：`0.8.0`
- OCI revision：`cd6e95b8ac7bde3b890848feadaa829f04911199`

Docker Hub push 成功并返回上述 digest。随后从终端进行匿名 registry HEAD
复核时，`auth.docker.io` 连续连接超时；因此不把该次网络检查写成成功的匿名
远程拉取测试。此前同一公开仓库已验证为 public，正式上传前仍建议再按 digest
执行一次无凭据拉取。

## 描述符和打包入口

准备目录（受 `.gitignore` 保护，不进入公开仓库）：

`work/submission-v0.8.0-20260925/`

其中包含：

- `descriptor-draft.json`：指向新的不可变镜像 digest，并保留已公布的官方
  House model 声明；
- `pack.command`：调用官方 qfbench2 toolkit v2.4.3，交互读取数字 Team ID，
  并在隐藏提示中读取 Team Key；脚本不保存 Team Key；
- `README.txt`：记录源码提交、检查点、镜像和操作说明。

描述符已经用合成 team_id 完成官方 `SubmissionDescriptor` 结构校验，镜像引用、
赛道、阶段、类别和模型声明均可解析。没有用合成身份生成 submission ZIP。

运行 `pack.command` 后，应检查生成的 `submission.zip` 只包含
`submission.json` 与 `team-claim.json`，再上传到 CodaBench。不要直接上传
`descriptor-draft.json`，也不要将 Team Key、team claim 或 submission ZIP
提交到 Git。
