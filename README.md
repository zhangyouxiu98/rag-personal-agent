# RAG Knowledge Agent

基于本地 LLM（Ollama）的 RAG 私有知识库问答 Agent。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 Ollama

```bash
ollama serve
```

确保已拉取所需模型：

```bash
ollama pull qwen2:7b
ollama pull nomic-embed-text
```

### 3. 摄入文档

```bash
# 单个文件
python run.py --ingest data/documents/report.pdf

# 整个目录
python run.py --ingest data/documents/
```

### 4. 开始问答

```bash
# 交互模式
python run.py

# 单次查询
python run.py --query "2024年第一季度营收是多少？"

# 流式输出
python run.py --query "总结一下这份报告" --stream
```

## 目录结构

```
my-agent/
├── run.py                    # CLI 入口
├── agent/
│   ├── core/                 # 核心：RAGAgent、State、Tools
│   ├── memory/               # 向量存储 + 会话记忆
│   ├── skills/               # 文档加载、分块、检索、重排序
│   ├── agents/               # 子 Agent：查询分析、检索、验证
│   ├── tools/                # 工具：Web搜索、DB查询、API调用
│   └── prompts/              # 系统提示词 + 示例
├── configs/
│   ├── config.yaml           # 主配置
│   └── models.yaml           # 模型元数据
├── data/
│   ├── documents/            # 待摄入文档
│   └── embeddings/           # ChromaDB 向量持久化
└── tests/
```

## 配置

编辑 `configs/config.yaml` 修改模型、检索参数等：

```yaml
ollama:
  base_url: "http://localhost:11434"
  chat_model: "qwen2:7b"
  embedding_model: "nomic-embed-text"

retrieval:
  top_k_initial: 10    # 初始检索数量
  top_k_final: 5       # 重排序后返回数量
```

## 支持的文档格式

- PDF (`.pdf`) - 通过 PyMuPDF
- Markdown (`.md`)
- 纯文本 (`.txt`)
- JSON (`.json`)

可通过 `DocumentLoader.register_handler()` 扩展更多格式。
