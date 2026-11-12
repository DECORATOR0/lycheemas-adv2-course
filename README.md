# LycheeMAS 与 AgentDropoutV2 课程实践

本仓库将 MASPO 提示词优化、五 Agent 两轮协作和 AgentDropoutV2 消息审核接入同一执行流程，并保存数学主实验的历史复算证据。

组员：熊思远（2023312323）、余思成（2023311605）。课程报告日期：2026-11-12。

- [课程报告与安装附录](docs/report/课程报告.md)
- [Word 报告](docs/report/运行前优化-熊思远-余思成.docx) · [PDF 报告](docs/report/运行前优化-熊思远-余思成.pdf)
- [框架适配、模块位置和两轮信息流](docs/architecture.md)
- [历史主实验的逐题证据和复算方法](evidence/reproduction/README.md)
- [数据来源、固定抽样和许可](evidence/data/SOURCES.md)

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

支持 `gsm8k`、`math500`（内部名 `math`）、`arc_easy`、`openbookqa`。切换 benchmark 时同时切换对应 train/test 文件和输出目录。各 benchmark 可以依次运行，题目之间通过 `--concurrency` 并发；一题内部维持固定两轮顺序。

提示词提议预算由 `--proposal-max-tokens` 单独控制，默认 1536。MATH 设为 3072，角色生成在优化阶段为 1024，最终测试阶段为 2048；其余三个 benchmark 的角色生成均为 1024。未完成的文本或 JSON 最多追加两次格式重试，每次响应及用量均记录，仍不完整则报错。

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

## 结果与证据

2026-11-12 的真实组合实验使用 Qwen3-8B、单张 40GB A100、非思考模式和 LangGraph 执行。每项使用 4 道训练题优化、20 道独立测试题，全部 80 题完成：

| 数据集 | 正确 / 全部测试 | 准确率 | 重试次数 | 阻断消息 | 修订后通过 |
|---|---:|---:|---:|---:|---:|
| [GSM8K](results/gsm8k/summary.json) | 16 / 20 | 80% | 2 | 2 | 0 |
| [MATH500](results/math500/summary.json) | 15 / 20 | 75% | 11 | 5 | 6 |
| [ARC-Easy](results/arc_easy/summary.json) | 18 / 20 | 90% | 14 | 11 | 3 |
| [OpenBookQA](results/openbookqa/summary.json) | 19 / 20 | 95% | 35 | 27 | 8 |

MASPO 改变了 GSM8K 的 E、MATH 的 C、ARC-Easy 的 D 提示词；OpenBookQA 完成搜索后保留初始提示词。运行时自然触发了反馈重生成与阻断；“修订后通过”表示审核器接受，实际答案既有改对，也有改错或保持原值的情况。逐项复核见 `evidence/validation/*_live_review.*`，报告举出 GSM8K 题 54 的正确初稿被误判案例。

MATH 完成优化后，将全部 20 道测试统一改为 2048 token 上限运行，再按原配置恢复两道执行错误题。最终评分对全部 20 题统一应用数学等价规则，修复 `7π` 与 `7\pi` 的格式误判；原输出与旧评分均保留在[重评分记录](results/math500/rescore_audit.json)中。重算命令为 `python scripts/rescore_math.py results/math500`。

17 项工程测试在实验服务器通过，另有真实模型三条门控路径探针全部通过；探针的草稿与修订文本为预设内容，详见[测试记录](evidence/validation/tests.txt)与[探针记录](evidence/validation/gate_probe/summary.json)。

全部完成流程及保留的中止尝试共 5,181 次模型调用，输入 3,857,368 token、输出 605,077 token。MATH 成本包含一次已完成优化的 790 次调用、最终测试及恢复的 490 次调用；另计首次失败优化 314 次和旧 1024-token 测试 459 次。原始记录分别保存在 `results/math500`、`results/math500_initial_failed` 和 `results/math500_eval1024_incomplete`，复用的优化不重复计费。

历史复算使用同批完整数学主运行的原评分字段：

| 基准 | 正确 / 全集 | 准确率 | 推理日期 |
|---|---:|---:|---|
| GSM8K | 1209 / 1319 | 91.66% | 2026-04-23 |
| MATH500 | 398 / 500 | 79.60% | 2026-04-23 |
| GSM8K 独立重复 | 1213 / 1319 | 91.96% | 2026-07-11 |

论文 v1 Table 1 的相应结果为 91.74% 和 78.40%。归档主配置使用 Qwen3.5-9B 选择器、top-20 后选择 1–5 个指标、60% 通过阈值；v1 使用 GPT-4.1-mini 选择器、直接 top-5 和零容忍审核。版本与配置说明见[历史证据](evidence/reproduction/README.md)。

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
| `calls.jsonl` | 逐次模型调用、响应、用途、完整性检查和全部尝试用量 |
| MATH 的 `optimization_calls.jsonl`、`prompts_source.json` | 复用优化的调用与两阶段预算来源 |

## 方法与来源

本仓库选取并适配 LycheeMAS 所需核心模块，新增课程执行层。MASPO 保留多粒度比较和提示词搜索机制，改用共享两轮执行器；ADv2 保留逐消息审核、反馈重生成和拒绝。本次课程集成使用固定人工指标池。

- [AgentDropoutV2 v1](https://arxiv.org/abs/2602.23258v1)
- [MASPO](https://arxiv.org/abs/2605.06623)
- [LycheeMAS](https://github.com/ecoli-hit/LycheeMAS)

数据、复用代码与新代码的来源和许可分别随对应目录保存，代码来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。报告对应代码版本为 `8635db0c51a6d07a6d67707b0b5480d50ad87fc1`。
