# 当前 MASRubric 配置的历史复现证据

本目录对照 MASRubric 2026-09-28 稿第 6 页 Table 1 的 Dynamic-MAS + MASRubric 行，保存论文采用组及另一组同配置历史重复的逐题记录。2026-11-12 重新读取服务器原归档并复算原评分；推理日期见下表。

| 数据集 | 运行 | 推理日期 | 正确 / 全集 | 准确率 |
|---|---|---|---:|---:|
| GSM8K | 当前论文采用组 | 2026-04-23 | 1209 / 1319 | 91.66% |
| MATH500 | 当前论文采用组 | 2026-04-23 | 398 / 500 | 79.60% |
| GSM8K | 七月独立重复 run 2 | 2026-07-11 | 1213 / 1319 | 91.96% |
| MATH500 | 七月同配置重复 run 2，含恢复 | 2026-07-12 | 387 / 500 | 77.40% |

GSM8K 与 MATH500 的差值分别为 +0.30、−2.20 个百分点。正式对照采用七月批次的同一重复编号 r2；全部四个重复编号的候选清单保存在 [candidate_runs.json](candidate_runs.json)。每个基准保留一个完整逻辑运行的最终文件，使用全部测试题和原有布尔评分。

GSM8K r2 未触发恢复，覆盖全部 1,319 个题目；它与论文采用组题面及参考答案一致，1,319 条原始最终输出均不同，74 题的正确性不同。MATH500 r2 共两次全量启动，分别有 493、496 题成功；最终文件保留末次 496 条结果，以及题号 100、108、239、481 的前次成功结果，共 500 条唯一题号。恢复时按成功完成情况更新结果，保留原答案和原评分；详情见 [恢复记录](math500_repeat2_recovery.json)。

## 配置

四个对照运行共用 Qwen3-8B 求解器与审核器、Qwen3.5-9B 动态发言者选择器、Qwen3-Embedding-8B 嵌入模型及数学 two-pool 指标池。指标重排由 Qwen3-8B 审核模型完成。拓扑为 AutoGen 动态 SelectorGroupChat，`math_team_variant=verifier`；`max_turns=7` 计入初始用户消息；先检索 20 个指标，再选 1–5 个；批量审核，`pass_rate=0.6`；修正预算 3；不锁定首次指标，不使用 cheap precheck。`select_q=5` 是指标数量上限，`exact_select_q=false` 时不补足为固定 5 项。四月题目并发 32，七月 r2 并发 50。详细配置与运行时间记录在 [summary.json](summary.json)。

## 与当前论文的对应

当前稿题名为 MASRubric: Auditing Information Flow in Multi-Agent Systems with Failure-Distilled Pitfall Rubrics，项目入口为 [MASRubric](https://github.com/TonySY2/MASRubric)。该稿第 6 页 Table 1 的 Dynamic-MAS + MASRubric 行报告 GSM8K 91.66%、MATH500 79.60%；同期发布配置的 `source_run` 指向 `math9_v2_top5_pass3of5_gpu234567_4lane_sel16k_2026-04-22-16-38-53`。本目录的 `*_main` 为该组，`*_repeat2` 为另一个七月重复批次。对应关系记录在 [current_paper_comparison.json](current_paper_comparison.json)。

## 复算

```bash
python verify_records.py
```

该命令重新统计保存的逐题布尔正确性、验证唯一题号及完整分母，并与 summary.json 对照。两侧均保留原运行评分，未改用课程集成实验的数学评分器。历史运行成本在原 summary 中累计，包含恢复尝试；与本次课程集成的调用成本分别记录。

文件说明：

- `gsm8k_main.jsonl`、`math500_main.jsonl`：当前论文采用组的逐题评分摘录。
- `gsm8k_repeat2.jsonl`、`math500_repeat2.jsonl`：七月同配置 r2 的最终评分摘录。
- `*_full.jsonl`：相应运行的题面、参考解答、预测答案及原始最终输出；MATH 的参考解答字段为 `answer`。
- `summary.json`：分子、分母、无缺题与重复题检查、配置及原运行汇总。
- `current_paper_comparison.json`：当前稿、论文采用运行与所选重复运行的对应。
- `candidate_runs.json`：四个重复编号的完整结果概况。
- `math500_repeat2_recovery.json`：两次启动、异常和四题保留来源。
