# 扩展运行独立复核

review_extension.py 只读运行资料并生成独立复核 JSON，不修改分数、模型输出或原始事件。

仓库根目录下使用：

    python evidence/validation/extension/review_extension.py --run results/extension/math500_100 --baseline results/math500
    python evidence/validation/extension/review_extension.py --run results/extension/arc_challenge_100
    python evidence/validation/extension/review_extension.py --run results/extension/math500_100_no_adv2 --baseline results/math500
    python evidence/validation/extension/compare_conditions.py --with-gate results/extension/math500_100 --without-gate results/extension/math500_100_no_adv2 --output evidence/validation/extension/math500_paired_review.json

若资料位于其他目录，通过 --run 指定；默认预期 100 题，可用 --expected 调整。--output 指定独立复核文件。

检查包括样本唯一性和分母、追加事件的最后状态、每题 11 节点顺序、共享提示、完整模型返回与审计来源、只接受通过消息的上下文、拒绝隔离、训练测试交集、原评分回放及全量调用成本。客户端格式重试以完整成功输出匹配，不按同 purpose 的第一次响应推断返回值。

MATH 附带原 20 题逐题对照、新 80 题单独统计，以及类型和难度覆盖。类型优先读取 type/subject；源样本未单列时使用 unique_id 中的官方题目目录类型。修订前后的标签匹配自动分类与人工核查分开；所有判错题及区间、元组、进制、角度等表示列为人工复核候选。

本脚本已用原 MATH20 的追加恢复资料校验，220 个节点和审计来源核对通过。完整扩测结果及数学等价人工复核完成后，再形成最终复核结论。

无 ADv2 条件要求所有节点 gate 为空、初稿与可传播输出一致，且没有审核调用或审核反馈生成。生成客户端自身的格式重试仍正常记录。配对脚本检查配置仅 evaluation_gate 不同、selection/prompts 全文一致，并逐题保存两边预测、正确性、共同正确、仅开启正确、仅关闭正确及共同错误的四格统计；终止拒绝计入已完成分母，API 错误单独标明。
