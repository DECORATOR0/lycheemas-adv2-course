# 139 的 ARC-Challenge 同机配对环境

ARC-Challenge 的开启与关闭 ADv2 两组均在同一张 48GB RTX 4090、同一模型服务中顺序执行。两组复用 254 已完成优化的冻结提示词；本机不重新运行 MASPO。

- `environment.json`：实际 Python、包版本、GPU、模型与服务参数。
- `tests.txt`：该机器上的 19 项工程测试全部通过，用时 1.043 秒。
- `service_probe.json`：8 个并发真实请求均返回正确整数答案，未返回思考内容；包含原始请求、响应与用量。

环境与探针记录的机器绝对路径已改为相对描述，模型名和软硬件配置保持不变。基准实验的调用与得分单独保存在 `results/arc_challenge_100_139` 和 `results/arc_challenge_100_139_no_adv2`。
