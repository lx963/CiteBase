"""文档问答的命令行入口。"""

import argparse
import sys
from collections.abc import Iterator

from requests import HTTPError, RequestException

from api._common import format_api_error
from client import User
from router import DEFAULT_MIN_SIMILARITY, DEFAULT_SOURCE, DocumentRouter


def read_questions() -> Iterator[str]:
    """逐轮读取用户问题，忽略空行。"""
    while True:
        try:
            question = input("问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if question:
            yield question


def main() -> None:
    """配置用户实例并把命令行问题交给它处理。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="用户问题，省略时进入多轮问答")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="文本块 JSON、PDF 或 DOCX 路径")
    parser.add_argument("--role", default="", help="回答角色或表达方式")
    parser.add_argument("--context", default="", help="辅助理解问题的背景信息")
    parser.add_argument("--model", help="回答模型，省略时读取 LLM_MODEL")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
        help=f"最低余弦相似度，范围 [-1, 1]（默认：{DEFAULT_MIN_SIMILARITY}）",
    )
    args = parser.parse_args()
    if args.question is not None and not args.question.strip():
        parser.error("问题不能为空")

    try:
        router = DocumentRouter(args.source, min_similarity=args.min_similarity, model=args.model)
        user = User(router, role=args.role, context=args.context)
    except (OSError, ValueError) as error:
        raise SystemExit(f"文档问答失败：{error}") from None

    questions = read_questions() if args.question is None else (args.question,)
    for question in questions:
        try:
            answer = user.ask(question)
        except HTTPError as error:
            message = format_api_error(error)
        except RequestException as error:
            message = f"API 请求失败（{type(error).__name__}）"
        except ValueError as error:
            message = f"文档问答失败：{error}"
        else:
            print(answer)
            continue

        if args.question is not None:
            raise SystemExit(message)
        print(message, file=sys.stderr)


if __name__ == "__main__":
    main()
