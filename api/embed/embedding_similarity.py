import argparse
import json
import math
import sys
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import TypedDict

from .embedding_parser import TextEmbedding


class TextSimilarity(TypedDict):
    text_a: str
    text_b: str
    similarity: float


def _unit_vector(vector: Sequence[float]) -> list[float]:
    if not vector or any(
        type(value) not in (int, float)
        or (isinstance(value, float) and not math.isfinite(value))
        for value in vector
    ):
        raise ValueError("向量必须是非空的有限数值数组")

    scale = max(abs(value) for value in vector)
    if scale == 0:
        raise ValueError("零向量无法计算余弦相似度")

    scaled = [value / scale for value in vector]
    norm = math.hypot(*scaled)
    return [value / norm for value in scaled]


def _unit_dot(vector_a: list[float], vector_b: list[float]) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("两个向量的维度必须一致")
    score = math.fsum(a * b for a, b in zip(vector_a, vector_b))
    return max(-1.0, min(1.0, score))


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """计算两个向量的余弦相似度，结果范围为 [-1, 1]。"""
    return _unit_dot(_unit_vector(vector_a), _unit_vector(vector_b))


def pairwise_similarities(records: Sequence[TextEmbedding]) -> list[TextSimilarity]:
    """按输入顺序计算每对文本向量的相似度，每对只计算一次。"""
    normalized = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"第 {index + 1} 条记录必须是 JSON 对象")
        text = record.get("text")
        vector = record.get("embedding")
        if not isinstance(text, str) or not isinstance(vector, list):
            raise ValueError(f"第 {index + 1} 条记录需要文本 text 和向量数组 embedding")
        normalized.append((text, _unit_vector(vector)))

    return [
        {"text_a": text_a, "text_b": text_b, "similarity": _unit_dot(vector_a, vector_b)}
        for (text_a, vector_a), (text_b, vector_b) in combinations(normalized, 2)
    ]


def main() -> None:
    """读取向量 JSON 并将两两相似度写到标准输出。"""
    parser = argparse.ArgumentParser(description="计算文本向量的两两余弦相似度")
    parser.add_argument("input", nargs="?", type=Path, help="向量 JSON 文件，省略时从标准输入读取")
    args = parser.parse_args()

    try:
        source = args.input.read_bytes() if args.input is not None else sys.stdin.buffer.read()
        records = json.loads(source)
        if not isinstance(records, list):
            raise ValueError("输入必须是 JSON 数组")
        result = pairwise_similarities(records)
    except (OSError, ValueError) as error:
        parser.exit(1, f"计算相似度失败：{error}\n")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
