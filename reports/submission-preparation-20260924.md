# 当前版本提交准备（2026-09-24）

用户授权按当前主目录版本准备 T1 Development 提交，不做 v5.6 近似回退，不合并 work 中的独立 v6.3–v6.5 实验。

## 已完成

- 主目录测试：139 passed，27.50 秒。
- 本地基础镜像 aptapt-finance-base:submission-20260924：linux/amd64；Python 3.13.15；numpy、pandas、scipy、pyarrow、statsmodels、sklearn、pytest 导入通过。
- 从当前 Dockerfile 和源码构建本地镜像 `aptapt-agent:submission-v6.2-20260924`。
- 本地镜像索引 digest：`sha256:4306ab1b4cfa2bcb672c1f3fadf17df942c0f0c03e3cafe8e6cef87bfb2ceb99`。此时尚未推送，不是已验证的远程拉取地址。
- `solve --help` 容器启动检查通过。
- 禁网、只读输入、关闭 starter 的 demo 回放检查通过，6 步，1 个交付物，结果位于 `work/submission-smoke.1hRIvP/results.json`。

回放检查只证明容器和工具链可以运行，不是 House 模型效果测试、官方 checker 评测或官方成绩。

## 发布及打包准备进展

- Docker Hub 账户 zoltonton 已完成设备登录。
- 已成功发布公开镜像 `docker.io/zoltonton/aptapt@sha256:4306ab1b4cfa2bcb672c1f3fadf17df942c0f0c03e3cafe8e6cef87bfb2ceb99`，标签 submission-v6.2-20260924；首次传输中断后复用已有层重试成功，没有触发比赛提交。
- Docker Hub 未登录仓库查询返回 is_private=false；用新建的空白 Docker 配置按 digest 拉取 linux/amd64 成功（本机层已缓存，非全量重新下载）。
- 官方 toolkit v2.4.3（源提交 358656a32094b19eff6fa95fe44b6a6671dc5041）安装在 `work/submission-toolkit-py313-20260924`，不修改评测环境。
- `work/submission-v6.2-20260924/descriptor-draft.json` 已填写自己的镜像 digest 和官方 House 模型声明；用合成团队信息校验官方 schema 通过，未生成合成团队的 ZIP。
- `work/submission-v6.2-20260924/pack.command` 会询问真实数字 Team ID，再调用官方隐藏 Team Key 提示；不会保存密钥。团队别名与 descriptor_digest 由官方 toolkit 生成。
- 系统拒绝代理操作终端应用，因此须由用户自行运行该打包脚本并输入团队信息，不尝试绕过限制。

## 打包校验及平台回执

- 用户自行输入团队信息后生成 `work/submission-v6.2-20260924/submission.zip`，1194 字节。
- ZIP SHA-256：`1b26cd3179d2d3531349e3b02af275dfb6fdd5bd829ac822f7990434a8152056`。
- ZIP 仅含 submission.json 和 team-claim.json。官方 SubmissionDescriptor 校验通过；镜像、模型声明、比赛及阶段与草稿一致；团队编号与官网已注册团队一致。
- 描述符规范化 digest、证明绑定的描述符原始字节 SHA-256、团队别名和证明格式均校验通过。未在报告保存团队密钥或证明；证明认证仍须由主办方完成。
- 已通过已登录的 CodaBench 账户 zolton 上传一次；平台新建提交 **942603**，文件 submission.zip，页面时间 **2026-09-24 15:26（GMT+8）**。提交计数从 0/1、0/20 更新为 1/1、1/20。
- 首次回执状态为 **Submitting**；15:27（GMT+8）刷新后记录仍存在、状态相同，尚无分数。这不代表已启动评测或评测通过，未重复上传。

页面说明本比赛关闭自动运行，需主办方核验团队后派发；Submitted 等待不代表异常。

官方入口：https://www.codabench.org/competitions/17765/

官方流程：https://www.agenthon.net/guides/submission-format/
