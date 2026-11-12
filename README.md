# LycheeMAS 与 AgentDropoutV2 课程实践

本仓库将 MASPO 提示词优化、五 Agent 两轮协作和 AgentDropoutV2 消息审核接入同一执行流程，并保存数学主实验的历史复算证据。

组员：熊思远（2023312323）、余思成（2023311605）。课程报告日期：2026-11-12。

- [合并课程报告](docs/report/expanded/课程报告.md)
- [Word 报告](docs/report/expanded/运行前优化-熊思远-余思成.docx) · [PDF 报告](docs/report/expanded/运行前优化-熊思远-余思成.pdf)
- [运行参数、百题对照、调用成本与复查](docs/extension_details.md)
- [框架适配、模块位置和两轮信息流](docs/architecture.md)
- [历史主实验的逐题证据和复算方法](evidence/reproduction/README.md)
- [数据来源、固定抽样和许可](evidence/data/SOURCES.md)
- [百题扩测数据与抽样记录](evidence/data/extension/README.md)

## 执行流程

1. 从官方训练划分选取少量问题，使用 MASPO 联合优化 A–E 五套角色提示词；模型权重保持冻结。
2. 保存 `prompts.json`，在两轮中使用相同的角色提示词。
3. 在固定测试子集上运行 A→B→C→D→E 两轮；每条消息由 ADv2 审核，失败时反馈重试，耗尽预算后阻断。
4. final 汇总第二轮通过审核的消息，再经审核、答案解析和评分。

同轮角色可见其前方全部已接受消息，第二轮还可见上一轮中自己及前方角色的已接受消息。被拒绝的消息只保留在审计记录中。具体矩阵和代码接入点见[架构说明](docs/architecture.md)。

本次适配使用三个固定指标：任务条件、推理有效性、答案一致性。数学题中的程序化角色使用受限 AST 算术解释器，选择题中改为选项证据核对。

## 安装与测试

需要 Python 3.10 或以上，以及一个可用的 OpenAI 兼容模型服务。请求端固定发送 `chat_template_kwargs={"enable_thinking": false}`；本次实验使用 Qwen3-8B 非思考模式。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[graph,data,dev]"
python -m unittest discover -s tests -v
python -m ruff check src scripts tests
```

Windows 激活命令为 `.venv\Scripts\Activate.ps1`。`graph` 依赖提供 LangGraph 执行载体；`data` 包含数据读取和符号评分需要的库。只运行基础确定性测试不需要 GPU 或模型服务。

## 运行一个 benchmark

仓库保留固定抽样的数据文件。每个数据集的训练候选池为 8 题，以下命令实际选用 4 题优化、20 题测试；选择结果写入 `selection.json`。

```bash
python scripts/run_course.py \
  --phase both --benchmark gsm8k \
  --train-file evidence/data/gsm8k_train.jsonl \
  --test-file evidence/data/gsm8k_test.jsonl \
  --output runs/gsm8k \
  --base-url http://127.0.0.1:8000/v1 --model Qwen3-8B \
  --train-n 4 --eval-n 20 --concurrency 8 \
  --max-tokens 1024 --seed 42 \
  --depth 1 --eval-batch 4 --beam-width 2 --max-retries 1 \
  --engine langgraph
```

将服务地址改为实际地址。需要认证时，通过 `MODEL_API_KEY` 环境变量提供令牌。

`seed` 控制题目与搜索抽样；候选提示提议的温度为 0.7，角色执行、比较与审核的温度为 0。候选提议默认最多 1536 tokens；本次 MATH 完成运行使用 `--proposal-max-tokens 3072`，优化阶段角色输出为 1024 token，最终测试阶段为 2048 token。

`--phase optimize` 只生成提示词产物；`--phase eval` 加载同一输出目录中的提示词并执行测试。重复相同命令可恢复运行，已完成题会跳过，失败题可重试；结果与全部调用记录均保留。修改会影响结果的参数时使用新输出目录。

支持 `gsm8k`、`math500`（内部名 `math`）、`arc_easy`、`arc_challenge`、`openbookqa`。切换 benchmark 时同时切换对应 train/test 文件和输出目录。各 benchmark 可以依次运行，题目之间通过 `--concurrency` 并发；一题内部维持固定两轮顺序。

提示词提议预算由 `--proposal-max-tokens` 单独控制，默认 1536。MATH 设为 3072，角色生成在优化阶段为 1024，最终测试阶段为 2048；其他 benchmark 的角色生成均为 1024。未完成的文本或 JSON 最多追加两次格式重试，每次响应及用量均记录，仍不完整则报错。

MATH 的两个阶段使用同一组选题和已完成的优化产物：

```bash
python scripts/run_course.py --phase optimize --benchmark math500 \
  --train-file evidence/data/math500_train.jsonl \
  --test-file evidence/data/math500_test.jsonl \
  --output runs/math-opt --base-url http://127.0.0.1:8000/v1 --model Qwen3-8B \
  --train-n 4 --eval-n 20 --concurrency 8 --seed 42 --depth 1 --eval-batch 4 \
  --beam-width 2 --max-retries 1 --engine langgraph \
  --max-tokens 1024 --proposal-max-tokens 3072
mkdir -p runs/math500
cp runs/math-opt/prompts.json runs/math500/
cp runs/math-opt/prompts_stats.json runs/math500/
python scripts/run_course.py --phase eval --benchmark math500 \
  --train-file evidence/data/math500_train.jsonl \
  --test-file evidence/data/math500_test.jsonl \
  --output runs/math500 --base-url http://127.0.0.1:8000/v1 --model Qwen3-8B \
  --train-n 4 --eval-n 20 --concurrency 8 --seed 42 --depth 1 --eval-batch 4 \
  --beam-width 2 --max-retries 1 --engine langgraph \
  --max-tokens 2048 --proposal-max-tokens 3072
```

### 复用提示词运行无 ADv2 对照

在新目录中复制开启组的提示词及搜索记录，使用 `--phase eval --disable-adv2`，不重复优化。对照组与开启组使用同一模型服务、相同题目和生成参数。以下 ARC-Challenge 命令使用保存的百题配置：

```bash
mkdir -p runs/arc-challenge-off
cp results/arc_challenge_100/prompts.json runs/arc-challenge-off/
cp results/arc_challenge_100/prompts_stats.json runs/arc-challenge-off/
python scripts/run_course.py --phase eval --benchmark arc_challenge \
  --train-file evidence/data/extension/arc_challenge_train.jsonl \
  --test-file evidence/data/extension/arc_challenge_test100.jsonl \
  --output runs/arc-challenge-off \
  --base-url http://127.0.0.1:8000/v1 --model Qwen3-8B \
  --train-n 4 --eval-n 100 --concurrency 8 --seed 42 \
  --depth 1 --eval-batch 4 --beam-width 2 --max-retries 1 \
  --engine langgraph --max-tokens 1024 --proposal-max-tokens 1536 \
  --disable-adv2
```

MATH 百题对照使用相应的训练文件、`evidence/data/extension/math500_test100.jsonl` 和 MATH 优化提示词，保持 `max_tokens=2048`、`proposal_max_tokens=3072`。关闭 ADv2 后仍执行相同的 11 个节点，省去审核与语义反馈重生成；客户端输出完整性重试继续生效。`config.json` 中开关记录为 `evaluation_gate`，配对结果保存在不同目录。

## 结果与证据

### 百题同机开关对照

每项任务在同一服务器、同一模型服务上比较开启与关闭 ADv2。题目、冻结角色提示词和生成参数相同，仅切换 `evaluation_gate`；两组都重新推理全部 100 题，不重复运行 MASPO。

| 数据集 | 服务器 | 开启 ADv2 | 关闭 ADv2 | 差值（百分点） |
|---|---|---:|---:|---:|
| MATH500 | 254 | [79/100（79%）](results/math500_100/summary.json) | [75/100（75%）](results/math500_100_no_adv2/summary.json) | +4 |
| ARC-Challenge | 139 | [91/100（91%）](results/arc_challenge_100_139/summary.json) | [91/100（91%）](results/arc_challenge_100_139_no_adv2/summary.json) | 0 |

MATH500 在本次百题测试中由 75% 增至 79%，净提高 4 个百分点；ARC-Challenge 两组均为 91%。数学仅开启组答对 4 题、仅关闭组答对 0 题；ARC 则分别有 2 题，收益与损失相抵。逐题复核见 [MATH](evidence/validation/extension/math500_final_review.md) 和 [ARC](evidence/validation/extension/arc_challenge_final_review.md)。

MATH 百题包含原 20 题与 seed=43 新抽取的 80 题，使用既有优化提示词；ARC-Challenge 先在 254 独立优化，再在 139 的 48GB RTX 4090 上运行两组同机对照。254 的原 ARC 运行 91/100 及其 702 次优化调用完整保留。所有任务使用 Qwen3-8B 非思考模式、LangGraph、五 Agent 两轮与 final，题目并发为 8。

数学两组按同一确定性等价规则复评，原模型输出和调用用量不变；审计见 [开启组](results/math500_100/rescore_audit.json) 和 [关闭组](results/math500_100_no_adv2/rescore_audit.json)。MATH 题 454 的缺字段审核恢复单独保存在 [client_recovery.json](results/math500_100/client_recovery.json)，原 99 道完成记录保持不变。

### 初始四项可用性测试

初始测试均使用 4 道训练题优化、20 道独立测试题，共 80/80 题完成。原结果保持如下：

| 数据集 | 正确 / 全部测试 | 准确率 | 重试次数 | 阻断消息 | 修订后通过 |
|---|---:|---:|---:|---:|---:|
| [GSM8K](results/gsm8k/summary.json) | 16 / 20 | 80% | 2 | 2 | 0 |
| [MATH500](results/math500/summary.json) | 15 / 20 | 75% | 11 | 5 | 6 |
| [ARC-Easy](results/arc_easy/summary.json) | 18 / 20 | 90% | 14 | 11 | 3 |
| [OpenBookQA](results/openbookqa/summary.json) | 19 / 20 | 95% | 35 | 27 | 8 |

MASPO 改变了 GSM8K 的 E、MATH 的 C、ARC-Easy 的 D 提示词；OpenBookQA 完成搜索后保留初始提示词。“修订后通过”表示审核器接受，具体答案可能改对、改错或保持原值。报告给出实际纠错及正确初稿被误判的案例，逐项记录保存在 `evidence/validation/*_live_review.*`。

### 功能验证与调用成本

初始 17 项工程测试在服务器通过；扩展和开关代码也经过各阶段服务器验证，139 上 19 项测试全部通过。最终数学评分版本另通过 29 项本地回归测试及 Ruff 检查，见 [最终评分验证](evidence/validation/math_notation_v2/summary.json)。真实模型的三条门控控制路径全部通过；预设草稿探针见 [记录](evidence/validation/gate_probe/summary.json)，139 的 8 并发非思考请求验证见 [服务探针](evidence/validation/139/service_probe.json)。

初始实验及其保留的中止尝试共 5,181 次模型调用；后续五项运行新增 9,880 次，合计 15,061 次，输入 11,698,496 token、输出 1,757,360 token。所有失败与恢复调用均计入；MATH 原 790 次优化仅计在初始账本，ARC 702 次优化仅计在 254 原运行。逐项成本、版本及复算方法见 [实验细节](docs/extension_details.md)。

### 历史主实验复算

历史复算使用同批完整数学主运行的原评分字段：

| 基准 | 正确 / 全集 | 准确率 | 推理日期 |
|---|---:|---:|---|
| GSM8K | 1209 / 1319 | 91.66% | 2026-04-23 |
| MATH500 | 398 / 500 | 79.60% | 2026-04-23 |
| GSM8K 独立重复 | 1213 / 1319 | 91.96% | 2026-07-11 |

论文 v1 Table 1 的相应结果为 91.74% 和 78.40%。归档主配置使用 Qwen3.5-9B 选择器、top-20 后选择 1–5 个指标、60% 通过阈值；v1 使用 GPT-4.1-mini 选择器、直接 top-5 和零容忍审核。版本与配置说明见 [历史证据](evidence/reproduction/README.md)。

```bash
python evidence/reproduction/verify_records.py
```

该命令重新检查完整分母、唯一题号和正确数。集成实验的正式结果保存在 `results/`；每题错误和终端拒绝均计入完整测试分母。

| 运行文件 | 内容 |
|---|---|
| `config.json`、`selection.json` | 配置、训练与测试题号、固定指标 |
| `prompts.json`、`prompts_stats.json`、`evaluation_prompts.json` | 优化提示词、搜索统计和测试实际提示词快照 |
| `results.jsonl` | 最终答案、评分、可见来源、每次门控与重试 |
| `summary.json` | 完成数、完整分母准确率和门控统计 |
| `calls.jsonl` | 全部调用、响应、用途、完整性检查和用量 |
| `prompts_source.json`、MATH 的 `optimization_calls.jsonl` | 冻结提示词和复用优化来源 |
| `rescore_audit.json`、`client_recovery.json` | 统一数学评分及指定错误题恢复记录 |

## 方法与来源

本仓库选取并适配 LycheeMAS 所需核心模块，新增课程执行层。MASPO 保留多粒度比较和提示词搜索机制，改用共享两轮执行器；ADv2 保留逐消息审核、反馈重生成和拒绝。本次课程集成使用固定人工指标池。

- [AgentDropoutV2 v1](https://arxiv.org/abs/2602.23258v1)
- [MASPO](https://arxiv.org/abs/2605.06623)
- [LycheeMAS](https://github.com/ecoli-hit/LycheeMAS)

数据、复用代码与新代码的来源和许可分别随对应目录保存，代码来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。各阶段运行与最终评分代码版本见 [版本对应表](docs/extension_details.md#实际环境与版本)。
