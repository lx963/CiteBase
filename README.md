# CiteBase

## 文档解析

在项目根目录运行以下命令，将 PDF 或 Word 文档转换为检索脚本可直接读取的 UTF-8 字符串数组。

```powershell
uv run python -m document_parser "论文.pdf"
uv run python -m document_parser "论文.docx" -o "论文.md_lines.json" --max-chars 800
```

默认写入当前目录的 `md_lines.json`，并替换同名文件。PDF 按页内空行分段，Word 按正文顺序读取段落和表格行。每块默认最多 500 个字符。扫描版 PDF 需先做 OCR，旧版 `.doc` 需先另存为 `.docx`。

Python 代码可用 `from document_parser import parse_document, write_md_lines`。调用 `parse_document(path)` 返回文本块，调用 `write_md_lines(path, output)` 写入 JSON。
