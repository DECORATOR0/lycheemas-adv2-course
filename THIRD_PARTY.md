# 代码来源与许可记录

## LycheeMAS

本项目选取 [ecoli-hit/LycheeMAS](https://github.com/ecoli-hit/LycheeMAS) 的必要接口与算法实现，基于课程工作区已有的 AgentDropoutV2 分支继续适配。上游基线为 [2026-09-15 的提交 9189cc27](https://github.com/ecoli-hit/LycheeMAS/commit/9189cc27abc6f1440b45b36d90be1999aa9ab957)，提交作者为 Jay-cvc。课程分支在该基线上已有本地增量。

上游 [pyproject.toml](https://github.com/ecoli-hit/LycheeMAS/blob/9189cc27abc6f1440b45b36d90be1999aa9ab957/pyproject.toml) 的项目元数据声明 `license = { text = "MIT" }`，作者字段为 LycheeMAS。2026-11-12 核对时，仓库根目录没有单独的 LICENSE 文件。

| 本项目文件 | 来源与修改 |
| --- | --- |
| `src/lychee_mas/core/types.py` | 复用 LycheeMAS 公共数据结构 |
| `src/lychee_mas/core/registry.py` | 复用课程分支的注册表，包含已有 `runtime_controller` 扩展 |
| `src/lychee_mas/plugins/prerun/base.py` | 复用运行前优化协议和统一入口 |
| `src/lychee_mas/plugins/prerun/graphview.py` | 复用原生 StateGraph 与 GraphView 之间的契约 |
| `src/lychee_mas/methods/prerun/graphops.py` | 复用图拓扑工具 |
| `src/lychee_mas/methods/prerun/maspo/optimizer.py` | 复用 MASPO 实现，增加执行器注入和可优化节点选择 |
| `src/lychee_mas/methods/prerun/maspo/executor.py` | 复用推理缓存和候选局部重执行接口 |
| `src/lychee_mas/methods/prerun/maspo/textops.py` | 复用提示校验和答案抽取；比较结果改为严格格式解析 |
| `src/lychee_mas/methods/runtime/agentdropout_v2.py` | 复用课程分支先前实现的固定指标运行时控制器 |
| `src/lychee_mas/methods/prerun/maspo/prompts.py` | 本次重新编写的比较、反思和角色提示文本 |
| `src/lychee_mas/course/`、`scripts/run_course.py`、`tests/test_course.py` | 本次课程集成新增的编排、客户端、数据检查、评分及测试 |

## MASPO

提示优化算法来源于 [MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](https://arxiv.org/abs/2605.06623)，参考实现为 [wangzx1219/MASPO](https://github.com/wangzx1219/MASPO)。保留 LycheeMAS 文件中的算法来源说明。

2026-11-12 检查参考仓库根目录时未发现 LICENSE 文件。本项目中的 MASPO 算法实现经由上述 LycheeMAS 课程分支取得；比较和提示变异所用提示文本由本次集成重新编写。

## AgentDropoutV2

方法依据为课程指定的 [AgentDropoutV2 v1](https://arxiv.org/abs/2602.23258v1)。本项目运行时控制器使用显式固定指标，执行审核、反馈修正和消息阻断。论文复现实验来源与本次集成实验来源分别记录在实验文档中。

## 数据与依赖

基准数据分别来自 GSM8K、MATH/MATH-500、ARC-Easy 和 OpenBookQA，其数据源与样本划分见实验配置和数据来源记录。LangGraph、SymPy 通过包管理器安装，各自保留原项目许可。

