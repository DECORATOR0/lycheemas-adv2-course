# 代码审查记录

审查日期：2026-11-12。代码范围：交付仓库 repo/，项目版本 lycheemas-adv2-course 0.1.0。本记录对应完成数学判分、严格比较解析、恢复快照和两轮顺序修复后的代码。

## 覆盖范围与结果

| 范围 | 核查结果 |
|---|---|
| methods/prerun/maspo/optimizer.py | 实际生成候选提示，比较局部、后继和最终输出，并进行 beam 选择；五个角色使用真实优化器入口。 |
| course/workflow.py 与 MASPO executor.py | 两轮共享五个角色提示；修改 C 时 C、C_r2 及受影响后继、final 重算，其他缓存保留。 |
| course/workflow.py 与 runtime/agentdropout_v2.py | 本轮、跨轮和 final 只读取通过审核的最终版本。拒绝稿及被修正的旧稿仅保留于审计记录。 |
| course/workflow.py | 执行顺序为 A→B→C→D→E→A_r2→B_r2→C_r2→D_r2→E_r2→final，通信可见范围独立于执行顺序。 |
| course/data.py、course/cli.py | 训练和测试校验题号、题面及 split；优化器只接收训练题目，测试时模型与审核器输入不含参考答案。 |
| course/io.py、course/cli.py | 空输出、截断、HTTP/解析异常不会变成正常通过；题目记录为 error，整体状态 incomplete。 |
| course/data.py | 修复全局删逗号/删文字造成的假阳性；纯数采用 Decimal 精确比较；符号表达式处理 ASCII pi、LaTeX pi 和复数等价。Unicode π 的覆盖遗漏见下方真实运行发现。 |
| MASPO textops.py、course/io.py | 比较响应必须包含唯一合法 choose A/B，畸形响应不会默认候选胜出。 |
| course/cli.py | 恢复比较配置、所选题目全文及实际评测提示全文，拒绝同题号改内容或改提示后混用旧结果。 |
| evidence/reproduction/ | 全量题号、正确数和分母复算一致；文档区分 v1 参数、后续 Dynamic 主配置和历史独立重复。 |

## 已执行验证

python -m unittest discover -s tests -v：10 项通过。覆盖真实优化器调用、共享提示重算、拒绝与修正传播、训练测试隔离、严格比较解析、判分、算术工具边界及完整分母。

额外独立 mock：拒绝 A/A_r2、修正 B/B_r2 后，全部后继和 final 不含被拒绝或修正前版本；C 提示替换后两轮 C 均使用新提示。

额外判分反例：(1,23) 对 (12,3)、-21 对 -2,1、不同文字答案、大整数差一和非零小数对零均判错；7*pi 对 LaTeX 7\\pi、6-5*i 对 6-5i、千位分隔数值等价均通过。原记录误将该 ASCII/LaTeX pi 测试写成 Unicode π，现已纠正。

## 真实运行复核项

真实结果分别核对最终答案、逐消息输入、实际模型调用提示、训练测试隔离及调用账。MATH 的根列表按题意允许交换顺序；总调用/Token 从完整 calls.jsonl 聚合，并区分优化与测试阶段。

2026-11-12 的 MATH 部分结果复核发现题 279 的预测 7π 与标准 7\\pi 数学等价，但被判错。实际根因是 normalize_math 未转换 Unicode π，后续受限 ASCII 表达式检查将其拒绝；本机已安装 SymPy 也能复现。实现已补 Unicode π 归一；独立复测 7π 与标准相等、8π 与标准不等均通过。最终评分应保留原始分数与复评分记录，该修复不需要重新生成模型回答。

修复后再次运行同一 unittest 命令：发现 17 项，16 项通过，1 项因该独立检查的 Python 环境未安装可选 LangGraph 依赖而跳过。真实结果的 LangGraph 执行记录另按逐节点输入复核。新增测试涵盖 Unicode π、独立提议预算、输出完整性有界重试及恢复快照。

