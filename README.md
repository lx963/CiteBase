# CiteBase

## 文档解析

在项目根目录运行以下命令，将 PDF 或 Word 文档转换为检索脚本可直接读取的 UTF-8 字符串数组。

```powershell
uv run python -m document_parser "论文.pdf"
uv run python -m document_parser "论文.docx" -o "论文.md_lines.json" --max-chars 800
```

默认写入当前目录的 `md_lines.json`，并替换同名文件。PDF 按页内空行分段，Word 按正文顺序读取段落和表格行。每块默认最多 500 个字符。扫描版 PDF 需先做 OCR，旧版 `.doc` 需先另存为 `.docx`。

Python 代码可用 `from document_parser import parse_document, write_md_lines`。调用 `parse_document(path)` 返回文本块，调用 `write_md_lines(path, output)` 写入 JSON。

## 文档问答

启动问答前，按 `.env.example` 在 `.env` 或环境变量中配置检索用的 `OPENROUTER_API_KEY`，以及回答用的 `LLM_BASE_URL`、`LLM_API_KEY` 和 `LLM_MODEL`。`LLM_BASE_URL` 填服务提供的 API 前缀，例如 `https://example.com/v1`，程序会拼接 `/chat/completions`。`--model` 可覆盖 `LLM_MODEL`。

默认读取项目根目录的 `md_lines.json`，也可用 `--source` 直接传入 PDF、DOCX 或文本块 JSON。直接运行可逐轮输入问题，也可在命令里提出单个问题。

```powershell
uv run python main.py
uv run python main.py "这篇论文的主要结论是什么？" --source "论文.pdf" --min-similarity 0.5
```

每轮独立计算当前问题与文档块的余弦相似度。达到阈值的块进入重排序，再将排名前三的片段交给回答模型。达标片段不足三条时使用全部达标片段，无达标片段时直接提示证据不足。默认阈值为 `0.4`，可用 `--min-similarity` 调整。

`client.py` 的每个 `User` 实例各自保存角色、背景和对话历史，并直接调用 `router.py`。可以用 `--role` 和 `--context` 配置命令行会话，也可以从 Python 创建用户。

```python
from client import User
from router import DocumentRouter

user = User(DocumentRouter("论文.pdf"), role="学术研究助理", context="我关注实证结果")
print(user.ask("论文的主要实证结论是什么？"))
```

## FastAPI 问答接口

沿用上面的 `.env` 配置，在项目根目录启动服务。接口默认读取 `md_lines.json`，可在 `http://127.0.0.1:8000/docs` 调试。

```powershell
uv run fastapi dev backend/main.py
```

| 接口 | 请求体 | 返回值 |
| --- | --- | --- |
| `POST /sessions` | `{}`，可选 `role` 和 `context` | `session_id` |
| `POST /sessions/{session_id}/questions` | `{"question":"论文的主要结论是什么？"}` | `question`、`answer` |
| `GET /sessions/{session_id}/history` | 无 | `session_id`、按顺序排列的 `history` |

连续提问时复用同一个 `session_id`。每个会话的历史独立保存在服务进程内存中，服务重启后清空。需要保留会话时使用单个 Uvicorn worker。
