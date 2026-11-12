# 数据来源与抽样

| 文件前缀 | 优化样本 | 测试样本 | 原始划分 |
|---|---:|---:|---|
| `gsm8k` | 8 | 20 | GSM8K train 7473；test 1319 |
| `math500` | 8 | 20 | 原始 MATH train 7500；MATH-500 test 500 |
| `arc_easy` | 8 | 20 | ARC-Easy train 2251；test 2376 |
| `openbookqa` | 8 | 20 | OpenBookQA main train 4957；test 500 |

每个划分分别使用 Python `random.Random(42).sample(range(N), n)` 抽样。源题号、源行索引及完整种子设置见 `sampling_manifest.json`。样本选定不使用模型输出或正确率。两个基准的所选优化/测试题面交集均为 0；MATH 优化题的 `unique_id` 均以 `train/` 开头。

GSM8K 答案取原解 `####` 后的数值；MATH 答案取原解最后一个完整 `\\boxed{...}` 内的表达式。完整原解保存在 `reference` 字段，供离线核对；MASPO 接收题面，标准答案用于测试后的评分。

上表为固定候选池。本次正式实验从各训练候选池中用相同种子选取 4 题用于提示词优化，20 道测试题全部使用；实际题号见各结果目录的 `selection.json`。科学选择题的抽样记录在 `science_sampling.json`，保留原始选项标签。四个基准的训练与测试所选题面交集均为 0。

## 上游来源

- GSM8K：OpenAI，[数据与说明](https://github.com/openai/grade-school-math)，[MIT 许可证](https://github.com/openai/grade-school-math/blob/master/LICENSE)，Copyright (c) 2021 OpenAI。
- MATH：Hendrycks 等，[作者仓库](https://github.com/hendrycks/math)。作者链接的 [数据集卡](https://huggingface.co/datasets/qwedsacf/competition_math) 标注 MIT 许可证并指向 [原许可证](https://github.com/hendrycks/math/blob/main/LICENSE)。
- MATH-500：[HuggingFaceH4 数据集](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)，其[数据集卡](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/raw/main/README.md) 将来源指向 OpenAI [PRM800K 的 MATH 划分](https://github.com/openai/prm800k#math-splits)。本次使用原研究归档中的 500 题版本。
- ARC-Easy：[AllenAI 官方数据集](https://huggingface.co/datasets/allenai/ai2_arc)，CC BY-SA 4.0；所附文件为官方划分中的固定抽样，并转换为统一题面、选项和答案字段。
- OpenBookQA：[AllenAI 官方数据集](https://huggingface.co/datasets/allenai/openbookqa)，来源为 [allenai/OpenBookQA](https://github.com/allenai/OpenBookQA)，官方仓库采用 [Apache License 2.0](https://github.com/allenai/OpenBookQA/blob/main/LICENSE)；本次使用 main 配置。

数据许可沿用上游，项目代码许可证不改变数据的许可。相关许可证文本保留于 `licenses/`。
