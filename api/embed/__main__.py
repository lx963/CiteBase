import json

from requests import HTTPError

from .._common import PROJECT_ROOT, format_openrouter_error
from .client import OpenRouterEmbeddings
from .embedding_parser import EmbeddingResponseParser


def main() -> None:
    """将示例词条转换为原文和向量的 JSON 数组。"""
    with (PROJECT_ROOT / "md_lines.json").open(encoding="utf-8") as file:
        texts = json.load(file)

    client = OpenRouterEmbeddings(model="nvidia/nemotron-3.5-embed-1b:free")
    try:
        result = client.embed(texts)
    except HTTPError as error:
        raise SystemExit(format_openrouter_error(error)) from None

    print(EmbeddingResponseParser.to_json(texts, result))


if __name__ == "__main__":
    main()
