from typing import Any, Literal, TypedDict

import requests

from .._common import read_api_key


class TextContent(TypedDict):
    text: str


class ImageContent(TypedDict):
    image: str


type EmbeddingInput = (
    str
    | list[str]
    | list[int]
    | list[list[int]]
    | TextContent
    | ImageContent
    | list[str | TextContent | ImageContent]
)


class SiliconFlowEmbeddings:
    """调用硅基流动 Embeddings API。"""

    URL = "https://api.siliconflow.cn/v1/embeddings"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        key = api_key if api_key is not None else read_api_key("SILICONFLOW_API_KEY")
        if not key.strip():
            raise ValueError("请传入 api_key，或在环境变量或 .env 中设置 SILICONFLOW_API_KEY")

        self.model = model
        self.timeout = timeout
        self._authorization = f"Bearer {key.strip()}"

    def embed(
        self,
        input: EmbeddingInput,
        *,
        encoding_format: Literal["float", "base64"] = "float",
        dimensions: int | None = None,
        user: str | None = None,
        truncate: Literal["left", "right"] | None = None,
        trace_id: str | None = None,
        traceparent: str | None = None,
    ) -> dict[str, Any]:
        """提交单条或批量输入，返回包含 data 和 usage 的完整响应。"""
        payload: dict[str, object] = {
            "model": self.model,
            "input": input,
            "encoding_format": encoding_format,
        }
        optional_fields = {"dimensions": dimensions, "user": user, "truncate": truncate}
        payload.update(
            (name, value) for name, value in optional_fields.items() if value is not None
        )

        headers = {"Authorization": self._authorization}
        if trace_id is not None:
            headers["X-Trace-Id"] = trace_id
        if traceparent is not None:
            headers["traceparent"] = traceparent

        response = requests.post(self.URL, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


type OpenRouterEmbeddingInput = (
    str | list[str] | list[int] | list[list[int]] | list[list[str]] | list[dict[str, Any]]
)


class OpenRouterEmbeddings:
    """调用 OpenRouter Embeddings API。"""

    URL = "https://openrouter.ai/api/v1/embeddings"

    def __init__(
        self,
        model: str,
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

    def embed(
        self,
        input: OpenRouterEmbeddingInput,
        *,
        encoding_format: Literal["float", "base64"] = "float",
        dimensions: int | None = None,
        input_type: str | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        """提交单条或批量输入，返回 OpenRouter 的完整响应。"""
        payload: dict[str, object] = {
            "model": self.model,
            "input": input,
            "encoding_format": encoding_format,
        }
        optional_fields = {"dimensions": dimensions, "input_type": input_type, "user": user}
        payload.update(
            (name, value) for name, value in optional_fields.items() if value is not None
        )

        headers = {"Authorization": self._authorization}
        response = requests.post(self.URL, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


