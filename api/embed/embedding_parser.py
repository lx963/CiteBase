"""配对 Embeddings 响应向量与原文。"""

import json
from collections.abc import Sequence
from typing import TypedDict


class TextEmbedding(TypedDict):
    text: str
    embedding: list[float]


class EmbeddingResponseParser:
    """根据响应索引为每条原文配对向量，并按原输入顺序输出。"""

    @staticmethod
    def parse(texts: str | Sequence[str], response: dict[str, object]) -> list[TextEmbedding]:
        """解析每条文本对应一个浮点向量的 Embeddings 响应。"""
        original_texts = [texts] if isinstance(texts, str) else texts
        items = response.get("data")
        if not isinstance(items, list):
            raise ValueError("响应中的 data 必须是数组")
        if len(items) != len(original_texts):
            raise ValueError("响应向量数量与原文本数量不一致")

        records: dict[int, TextEmbedding] = {}
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("响应中的每条向量记录必须是对象")

            index = item.get("index")
            if type(index) is not int or not 0 <= index < len(original_texts):
                raise ValueError("响应中的 index 超出原文范围或格式无效")
            if index in records:
                raise ValueError(f"响应中存在重复的 index {index}")

            vector = item.get("embedding")
            if not isinstance(vector, list) or not vector or any(
                type(value) not in (int, float) for value in vector
            ):
                raise ValueError(f"index {index} 的 embedding 必须是非空数值数组")

            records[index] = {"text": original_texts[index], "embedding": vector}

        return [records[index] for index in range(len(original_texts))]

    @classmethod
    def to_json(cls, texts: str | Sequence[str], response: dict[str, object]) -> str:
        """将原文和向量序列化为 JSON 数组。"""
        return json.dumps(cls.parse(texts, response), ensure_ascii=False, allow_nan=False)
