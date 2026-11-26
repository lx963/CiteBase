"""从 PDF 和 Word 提取文本，生成检索入口使用的字符串数组。"""

import json
import re
from collections.abc import Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader


DEFAULT_CHUNK_SIZE = 500
PARAGRAPH_BREAK = re.compile(r"\n(?:[^\S\n]*\n)+")
SENTENCE_END = re.compile(r"[。！？；!?]+|[.;](?=\s|$)")


def _pdf_blocks(source: Path) -> Iterator[str]:
    """按页提取文本，用空行识别页内段落。"""
    with source.open("rb") as file:
        reader = PdfReader(file)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF 需要密码，请先解密再解析")
        for page in reader.pages:
            if "/Contents" not in page:
                continue
            text = page.extract_text(extraction_mode="layout") or page.extract_text()
            yield from PARAGRAPH_BREAK.split(text)


def _table_rows(table: Table) -> Iterator[str]:
    """逐行提取表格文本，合并单元格只提取一次。"""
    seen_cells = set()
    for row in table.rows:
        cells = []
        for cell in row.cells:
            if cell._tc in seen_cells:
                continue
            seen_cells.add(cell._tc)
            parts = []
            for block in cell.iter_inner_content():
                if isinstance(block, Paragraph):
                    parts.append(block.text)
                else:
                    parts.extend(_table_rows(block))
            text = " ".join(" ".join(parts).split())
            if text:
                cells.append(text)
        if cells:
            yield " | ".join(cells)


def _word_blocks(source: Path) -> Iterator[str]:
    """按正文顺序提取段落和表格行。"""
    for block in Document(str(source)).iter_inner_content():
        if isinstance(block, Paragraph):
            yield block.text
        else:
            yield from _table_rows(block)


def _split_block(block: str, max_chars: int) -> Iterator[str]:
    """优先在靠后的句末或空白处切分，必要时按字符数切分。"""
    text = " ".join(block.split())
    start = 0
    while len(text) - start > max_chars:
        window = text[start : start + max_chars + 1]
        midpoint = max_chars // 2
        sentence_end = max(
            (match.end() for match in SENTENCE_END.finditer(window) if match.end() <= max_chars),
            default=0,
        )
        space = window.rfind(" ")
        end = next(
            (position for position in (sentence_end, space) if position > midpoint),
            max_chars,
        )
        yield text[start : start + end].strip()
        start += end
        if text[start] == " ":
            start += 1
    if start < len(text):
        yield text[start:]


def parse_document(source: str | Path, *, max_chars: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """解析单个 .pdf 或 .docx，返回可写入 md_lines.json 的文本块。"""
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("每块最大字符数必须是正整数")
    path = Path(source)
    match path.suffix.lower():
        case ".pdf":
            blocks = _pdf_blocks(path)
        case ".docx":
            blocks = _word_blocks(path)
        case ".doc":
            raise ValueError("旧版 .doc 请先另存为 .docx")
        case _:
            raise ValueError("仅支持 .pdf 和 .docx")

    chunks = [chunk for block in blocks for chunk in _split_block(block, max_chars)]
    if not chunks:
        raise ValueError("未提取到文本；如果是扫描版 PDF，请先做 OCR")
    return chunks


def write_md_lines(
    source: str | Path,
    output: str | Path = "md_lines.json",
    *,
    max_chars: int = DEFAULT_CHUNK_SIZE,
) -> Path:
    """解析文档并将文本块写为 UTF-8 JSON 数组。"""
    destination = Path(output)
    if Path(source).resolve() == destination.resolve():
        raise ValueError("输入和输出不能是同一个文件")
    chunks = parse_document(source, max_chars=max_chars)

    # 先写入同目录临时文件，完整写入后再替换目标。
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(chunks, temporary, ensure_ascii=False, indent=4)
            temporary.write("\n")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination
