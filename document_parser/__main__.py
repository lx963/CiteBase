"""将单个 PDF 或 Word 文档转换为 md_lines.json。"""

import argparse
from zipfile import BadZipFile

from docx.opc.exceptions import PackageNotFoundError
from lxml.etree import XMLSyntaxError
from pypdf.errors import PyPdfError

from .parser import DEFAULT_CHUNK_SIZE, write_md_lines


def main() -> None:
    """读取命令行参数并写入解析结果。"""
    parser = argparse.ArgumentParser(prog="python -m document_parser", description=__doc__)
    parser.add_argument("source", help="PDF 或 DOCX 文件路径")
    parser.add_argument("-o", "--output", default="md_lines.json", help="输出路径（默认 md_lines.json）")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"每块最大字符数（默认 {DEFAULT_CHUNK_SIZE}）",
    )
    args = parser.parse_args()
    try:
        output = write_md_lines(args.source, args.output, max_chars=args.max_chars)
    except (OSError, ValueError, PyPdfError, PackageNotFoundError, BadZipFile, XMLSyntaxError) as error:
        parser.exit(1, f"解析失败：{error}\n")
    print(output)


if __name__ == "__main__":
    main()
