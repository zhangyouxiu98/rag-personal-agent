# Few-shot Examples for RAG QA

## Example 1: Simple factual question

**User**: 公司2024年第一季度营收是多少？
**Context**: [Chunk from 2024-financial-report.pdf: "公司2024年第一季度实现营业收入12.5亿元，同比增长18%，净利润2.3亿元。"]
**Assistant**: 根据公司2024年财务报告，第一季度营收为**12.5亿元**，同比增长18%。
[Source: 2024-financial-report.pdf]

## Example 2: When context is insufficient

**User**: 竞争对手B公司的市场份额是多少？
**Context**: [Only contains internal company data, no competitor information.]
**Assistant**: Based on the available knowledge base, I cannot answer this question. The current documents do not contain information about competitor B's market share. Please provide relevant documents if available.

## Example 3: Multi-part comparison

**User**: 请比较A产品和B产品的价格和性能。
**Context**: [Chunks from product-specs.pdf contain specs for both products.]
**Assistant**: Here is the comparison:

| 特性 | A产品 | B产品 |
|------|-------|-------|
| 价格 | ¥2,999 | ¥3,499 |
| 性能评分 | 92/100 | 95/100 |
| 保修期 | 2年 | 3年 |

[Source: product-specs.pdf]

## Example 4: Follow-up with conversation history

**User** (turn 1): 什么是RAG技术？
[Assistant explains RAG with citations.]

**User** (turn 2): 它有什么优点？
**Assistant**: Based on the previous context about RAG technology, its main advantages include:
1. **减少幻觉**: 回答基于检索到的文档，降低了模型编造信息的风险。
2. **知识可更新**: 无需重新训练模型，只需更新知识库文档。
3. **可溯源**: 每个回答都可以追溯到具体的文档来源。
[Source: ai-techniques-overview.md]
