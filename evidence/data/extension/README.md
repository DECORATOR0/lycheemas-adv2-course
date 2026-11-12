# 扩展测试样本

| 文件 | 题数 | 选样 |
|---|---:|---|
| math500_test100.jsonl | 100 | 原有 20 题全部字段与顺序保留；在完整 500 题的源顺序补集中以 seed 43 抽取另 80 题。 |
| arc_challenge_train.jsonl | 8 | ARC-Challenge 官方 train 的 1,119 题中，以 seed 42 抽取 8 道候选训练题。运行实际使用的 4 题由其 selection.json 记录。 |
| arc_challenge_test100.jsonl | 100 | ARC-Challenge 官方 test 的 1,172 题中，以 seed 42 抽取 100 题。 |

选题不读取模型结果或正确率。sampling_manifest.json 记录全部源题号、索引、种子、原 20/新增 80 划分及隔离检查。每份样本题号唯一，两个基准的训练测试题号和规范化题文交集均为 0。

MATH100 供冻结原优化提示后的独立新运行使用；原 20 题实验结果保留。ARC-Challenge 保留上游选项标签，标签包括字母和数字，提示和评分应使用实际标签。

MATH500 来源和 MIT 许可见上级 [SOURCES.md](../SOURCES.md)。本次全量源与原 20 题逐字段对齐后抽样，完整 500 题文件未包含在本目录。

ARC-Challenge 来源为 Allen Institute 的 [ai2_arc 数据集](https://huggingface.co/datasets/allenai/ai2_arc)，沿用 [CC-BY-SA-4.0](../licenses/ARC_CC_BY_SA_4.0.txt)。样本保存原题、选项和答案，新增来源索引及运行标识；许可不因项目代码许可证而改变。

复现命令（在仓库根目录执行，依赖 requests 和 pyarrow）：

    python scripts/prepare_extension_data.py --math-source /path/to/full_math500.jsonl

脚本默认读取 evidence/data/math500_test.jsonl 与 math500_train.jsonl，将新样本写入本目录。可通过 --base-math20、--math-train、--output、--math-extra-seed、--arc-seed、--arc-train-url、--arc-test-url 显式指定来源和种子；完整源须保持与原 20 题相同的题号及顺序。
