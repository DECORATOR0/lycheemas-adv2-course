# 运行参数、百题对照与结果复查

安装和完整 CLI 命令见 [README](../README.md)，框架接入见 [架构说明](architecture.md)，合并后的报告见 [课程报告](report/expanded/课程报告.md)。本页同时保存初始四项测试和后续百题对照的参数、成本及复查入口。

## 实验组织与参数

MASPO 只优化角色提示词，模型权重保持不变。初始四项任务均从 8 道训练候选题中选 4 道优化，再测试 20 题。MATH 百题包含原 20 题及以 seed=43 从其余题目抽出的 80 题，冻结既有提示词，开启和关闭 ADv2 时分别重新推理全部 100 题。ARC-Challenge 先在 254 用 4 道题独立优化并测试 100 题，再将冻结提示词复制到 139，在同一环境顺序执行开启、关闭两组。

| 参数 | 设置 |
| --- | --- |
| 角色 / 轮数 / final | A–E 五 Agent，两轮角色执行及一个 final，共 11 个节点 |
| MASPO depth / beam / eval_batch | 1 / 2 / 4 |
| Local / Lookahead / Global | 0.4 / 0.4 / 0.2 |
| 生成与审核温度 / 提议温度 | 0 / 0.7 |
| 客户端题目并发 | 8 |
| 随机种子 | 运行选题与优化 seed=42；MATH 新增 80 题抽样 seed=43 |
| 角色输出预算 | 优化阶段 1024；MATH 正式测试 2048；其余测试 1024 |
| 提示词提议预算 | MATH 3072；其余 1536 |
| ADv2 语义重试 / 阈值 | max_retries=1；violation_threshold=0 |
| 客户端格式重试 | 最多追加 2 次，仅处理不完整输出或缺少必需字段的审核 JSON |
| 关闭 ADv2 | evaluation_gate=false；不改变题目、提示词、图、角色生成参数及评分规则 |

各配对的 config.json 除 evaluation_gate 外一致，题目、标准答案及 prompts.json 均逐项核对。关闭组仍执行 11 个节点，gate 为空，没有审核模型调用或 ADv2 反馈重写；输出完整性重试规则继续适用。

## 实际环境与版本

| 运行位置 | GPU | 主要软件 | 模型服务 |
| --- | --- | --- | --- |
| 254：初始测试、MATH 百题配对、ARC 原运行 | A100-SXM4-40GB | Python 3.10.19；vLLM 0.19.1；Torch 2.10；Transformers 5.6.0.dev0 | Qwen3-8B，BF16，非思考 |
| 139：ARC 百题配对 | RTX 4090，49140 MiB | Python 3.10.19；vLLM 0.19.1；Torch 2.10.0+cu129；Transformers 5.13.0 | Qwen3-8B，BF16，非思考 |

两台服务均设置上下文 16384、max_num_seqs=16、显存利用率 0.90；客户端使用 LangGraph 1.2.20。正式比较分别在同一服务器内完成。环境证据见 [254](../evidence/validation/environment.json) 和 [139](../evidence/validation/139/environment.json)。

| 代码版本 | 用途 |
| --- | --- |
| [8635db0](https://github.com/DECORATOR0/lycheemas-adv2-course/commit/8635db0c51a6d07a6d67707b0b5480d50ad87fc1) | 初始四项 20 题的最终运行代码 |
| [84f5389](https://github.com/DECORATOR0/lycheemas-adv2-course/commit/84f5389586800512402fd9999573f6539343c738) | 新增 ARC-Challenge 与 254 原百题运行 |
| [3d2a46e](https://github.com/DECORATOR0/lycheemas-adv2-course/commit/3d2a46e551ef60f151027c6fc7539ef92b344586) | 显式 ADv2 开关；MATH 关闭组和 139 两组运行 |
| [d2941ba](https://github.com/DECORATOR0/lycheemas-adv2-course/commit/d2941ba808a91c47b5aac08508f020cbda667609) | 将缺少审核必需字段纳入已有格式重试；仅恢复 MATH 题 454 |
| [396cf00](https://github.com/DECORATOR0/lycheemas-adv2-course/commit/396cf008903961cedddbb5f6e0724457e995d810) | 数学符号、显式输出进制、角度、根列表与年级序数的统一确定性等价判定 |

初始版本的 17 项工程测试在服务器通过；扩测服务器验证分别包括 18 项初始扩展测试、19 项开关测试、20 项 schema 恢复测试，139 上通过 19 项。最终数学评分版本在本地通过 29 项回归测试及 Ruff 检查。证据分别见 [扩展测试](../evidence/validation/extension/tests.txt)、[开关服务器测试](../evidence/validation/adv2_toggle/server_tests.txt)、[恢复测试](../evidence/validation/extension/schema_recovery_tests.txt)、[139 测试](../evidence/validation/139/tests.txt) 和 [最终本地评分回归](../evidence/validation/math_notation_v2/summary.json)。

## 初始四项结果

| 数据集 | 正确 / 测试 | 准确率 |
| --- | --- | --- |
| GSM8K | 20 完成 / 16 正确 | 80% |
| MATH500 | 20 完成 / 15 正确 | 75% |
| ARC-Easy | 20 完成 / 18 正确 | 90% |
| OpenBookQA | 20 完成 / 19 正确 | 95% |

原始四项结果与分母保持不变。MATH 初始 20 题为 15/20；后续冻结提示词重新推理该 20 题时，两组均为 16/20，属于不同运行。

## 百题同机配对结果

| 数据集 | 服务器 | 开启 | 关闭 | 差值 pp | 共同正确 | 仅开启正确 | 仅关闭正确 | 共同错误 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MATH500 | 254 | 79/100 | 75/100 | +4 | 75 | 4 | 0 | 21 |
| ARC-Challenge | 139 | 91/100 | 91/100 | +0 | 89 | 2 | 2 | 7 |

MATH500 本次百题对照由 75% 增至 79%，净增加 4 个百分点；ARC-Challenge 两组均为 91%。ARC 的两条收益与两条损失相抵，逐题分析见 [ARC 配对复核](../evidence/validation/extension/arc_challenge_final_review.md)，数学见 [MATH 最终复核](../evidence/validation/extension/math500_final_review.md)。

| MATH 子集 | 题数 | 开启正确 | 关闭正确 |
| --- | --- | --- | --- |
| 原 20 题重新推理 | 20 | 16 | 16 |
| 新增 80 题 | 80 | 63 | 59 |

MATH 仅开启组答对的题号为 234、294、473、152。254 原 ARC 运行同样为 91/100，完整结果保存在 [原运行](../results/arc_challenge_100/summary.json)；开关比较使用 139 的两组。

## 输出完整性恢复与统一评分

MATH 第 454 题初次遇到两条语法完整但缺少 violations 字段的审核 JSON，均作为错误处理。补丁将这种 schema 缺键纳入已有的有限格式重试，只恢复这道失败题，原 99 道已完成记录保持不变；失败响应和所有调用用量保留。恢复范围与版本见 [client_recovery.json](../results/math500_100/client_recovery.json)。

数学两组统一应用相同的确定性等价规则。开启组 100 题中有 96 条可评分最终输出及 4 条终端拒绝；关闭组有 100 条可评分输出。终端拒绝仍按错误计入 100 题分母。复评只更新评分，模型输出、参考答案、节点与调用用量不变。

| 运行 | 全分母 | 可评分输出 | 终端拒绝 | 旧正确数 | 最终正确数 | 评分变化条数 |
| --- | --- | --- | --- | --- | --- | --- |
| MATH 开启 | 100 | 96 | 4 | 72 | 79 | 7 |
| MATH 关闭 | 100 | 100 | 0 | 67 | 75 | 8 |

| 题号 | 模型输出示例 | 标准答案 | 等价规则 |
| --- | --- | --- | --- |
| 409 | 2√5 | 2 \sqrt{5} | normalized_exact |
| 235 | 5/9 | \frac 59 | symbolic_equivalence |
| 70 | 40 | 40_9 | radix_notation |
| 73 | 3√3⁄4 | \frac{3\sqrt{3}}{4} | symbolic_equivalence |
| 281 | 36 | 36^\circ | normalized_exact |
| 435 | (-∞, 2) ∪ (3, ∞) | (-\infty, 2) \cup (3, \infty) | normalized_exact |
| 284 | 3+2√2,3-2√2 | 3 \pm 2 \sqrt{2} | unordered_roots |
| 379 | 12th | 12 | normalized_exact |

完整逐题审计见 [开启组](../results/math500_100/rescore_audit.json)、[关闭组](../results/math500_100_no_adv2/rescore_audit.json)。`python scripts/rescore_math.py results/math500_100` 和相应关闭组命令可只读复算。第 379 题要求年级，关闭组输出 `12th` 与标准 `12` 的含义相同；此类规则按题面条件应用。

## 全部调用成本

以下成本取自全部 calls.jsonl，包含失败与恢复尝试。初始 MATH 复用优化另存 optimization_calls.jsonl，统计一次。服务连通性探针另有独立记录，不计入基准实验调用。

| 初始数据集或尝试 | 调用数 | 输入 token | 输出 token |
| --- | --- | --- | --- |
| GSM8K | 1234 | 764,829 | 114,851 |
| MATH500 | 1280 | 1,204,707 | 192,806 |
| ARC-Easy | 866 | 479,610 | 77,499 |
| OpenBookQA | 1028 | 586,089 | 97,393 |
| MATH 初次优化中止 | 314 | 227,594 | 48,991 |
| MATH 旧 1024 测试 | 459 | 594,539 | 73,537 |

初始四项流程及保留的 MATH 未完成尝试合计 5,181 次模型调用，输入 3,857,368 token，输出 605,077 token，达到输出长度上限 5 次。MATH 正式流程计入一次已完成优化的 790 次调用，以及 2,048 token 测试中包含失败和恢复的全部 490 次调用。另计初次中止优化 314 次和旧 1,024 token 测试 459 次，不重复计算复用的优化。

| 新增运行 | 调用数 | 输入 token | 输出 token |
| --- | --- | --- | --- |
| MATH500 开启 / 254 | 2352 | 2,869,815 | 382,767 |
| ARC 原运行 / 254 | 2978 | 1,883,340 | 249,558 |
| MATH500 关闭 / 254 | 1102 | 1,032,216 | 223,277 |
| ARC-Challenge 开启 / 139 | 2348 | 1,566,529 | 198,159 |
| ARC-Challenge 关闭 / 139 | 1100 | 489,228 | 98,522 |

扩测与开关对照新增 9,880 次调用，输入 7,841,128 token、输出 1,152,283 token，按五组运行的全部 calls.jsonl 计入失败与恢复尝试。MATH 原 790 次优化保留在初始账本；ARC 独立优化的 702 次调用仅在 254 原运行中计算一次，139 两组及 MATH 关闭组均无新增优化。

全部基准实验共 15,061 次模型调用，输入 11,698,496 token、输出 1,757,360 token。MATH 原优化 790 次仅计在初始账本；ARC 独立优化 702 次仅计在 254 原运行，139 两组均只统计新增推理。

## 记录入口

- [扩测数据与抽样清单](../evidence/data/extension/README.md)
- [历史数学主实验与复算脚本](../evidence/reproduction/README.md)
- [MATH 开启](../results/math500_100/) / [关闭](../results/math500_100_no_adv2/)
- [ARC 139 开启](../results/arc_challenge_100_139/) / [关闭](../results/arc_challenge_100_139_no_adv2/)
- [ARC 254 原运行](../results/arc_challenge_100/)
- [真实门控分支探针](../evidence/validation/gate_probe/summary.json)
- [139 非思考并发探针](../evidence/validation/139/service_probe.json)
