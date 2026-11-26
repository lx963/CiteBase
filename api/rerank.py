import math
from collections.abc import Sequence
from typing import TypedDict

import requests

from ._common import read_api_key


class RerankDocument(TypedDict, total=False):
    text: str
    image: str


class RerankResult(TypedDict):
    index: int
    relevance_score: float


class OpenRouterReranker:
    """按查询词和候选内容调用 OpenRouter 重排序接口。"""

    URL = "https://openrouter.ai/api/v1/rerank"

    def __init__(
        self,
        model: str = "nvidia/nemotron-3.5-rerank-8b:free",
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        key = api_key if api_key is not None else read_api_key("OPENROUTER_API_KEY")
        if not key.strip():
            raise ValueError("请传入 api_key，或在环境变量或 .env 中设置 OPENROUTER_API_KEY")

        self.model = model
        self.timeout = timeout
        self._authorization = f"Bearer {key.strip()}"

    def rerank(
        self,
        query: str,
        documents: Sequence[str | RerankDocument],
        *,
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """返回按相关性降序排列的候选索引和模型评分。"""
        if not query.strip() or not documents:
            raise ValueError("重排序需要非空查询和至少一个候选")
        if top_n is not None and not 1 <= top_n <= len(documents):
            raise ValueError("top_n 必须介于 1 和候选数量之间")

        payload: dict[str, object] = {
            "model": self.model,
            "query": query,
            "documents": list(documents),
        }
        if top_n is not None:
            payload["top_n"] = top_n

        headers = {"Authorization": self._authorization}
        response = requests.post(self.URL, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()

        body = response.json()
        items = body.get("results") if isinstance(body, dict) else None
        if not isinstance(items, list) or not items:
            raise ValueError("重排序响应缺少有效的 results 数组")
        if top_n is not None and len(items) != top_n:
            raise ValueError("重排序返回的结果数量与 top_n 不一致")

        results: list[RerankResult] = []
        seen: set[int] = set()
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("重排序结果必须是对象")
            index = item.get("index")
            score = item.get("relevance_score")
            if type(index) is not int or not 0 <= index < len(documents) or index in seen:
                raise ValueError("重排序响应包含无效或重复的候选索引")
            if type(score) not in (int, float) or not math.isfinite(score):
                raise ValueError("重排序响应包含无效的 relevance_score")
            results.append({"index": index, "relevance_score": score})
            seen.add(index)

        return sorted(results, key=lambda item: item["relevance_score"], reverse=True)
