"""负责文档载入、向量召回、重排序和证据问答。"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict
from zipfile import BadZipFile

from docx.opc.exceptions import PackageNotFoundError
from lxml.etree import XMLSyntaxError
from pypdf.errors import PyPdfError

from api._common import PROJECT_ROOT
from api.embed import OpenRouterEmbeddings
from api.embed.embedding_parser import EmbeddingResponseParser
from api.embed.embedding_similarity import cosine_similarity
from api.llm import ChatCompletions, ConversationTurn
from api.rerank import OpenRouterReranker
from document_parser import parse_document


DEFAULT_SOURCE = PROJECT_ROOT / "md_lines.json"
ANSWER_EXCERPT_COUNT = 3
DEFAULT_MIN_SIMILARITY = 0.4
INSUFFICIENT_EVIDENCE = "没有找到达到相似度阈值的文档片段，证据不足，无法回答。"


class RankedChunk(TypedDict):
    text: str
    similarity: float
    relevance_score: float


def rank_query(
    query: str,
    texts: Sequence[str],
    client: OpenRouterEmbeddings,
    *,
    min_similarity: float = -1.0,
) -> list[RankedChunk]:
    """计算文本块相似度，仅对达到阈值的候选重排序。"""
    if not texts:
        return []

    inputs = [query, *texts]
    records = EmbeddingResponseParser.parse(inputs, client.embed(inputs))
    query_vector = records[0]["embedding"]
    candidates = []
    for record in records[1:]:
        similarity = cosine_similarity(query_vector, record["embedding"])
        if similarity >= min_similarity:
            candidates.append((record["text"], similarity))
    if not candidates:
        return []
    candidates.sort(key=lambda candidate: candidate[1], reverse=True)

    candidate_texts = [text for text, _ in candidates]
    reranked = OpenRouterReranker().rerank(query, candidate_texts, top_n=len(candidates))
    results: list[RankedChunk] = []
    for item in reranked:
        text, similarity = candidates[item["index"]]
        results.append({
            "text": text,
            "similarity": similarity,
            "relevance_score": item["relevance_score"],
        })
    return results


def load_chunks(source: str | Path) -> list[str]:
    """载入已分块的 JSON，或直接解析 PDF、DOCX。"""
    path = Path(source)
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as file:
            texts = json.load(file)
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise ValueError(f"{path.name} 必须是文本字符串数组")
        return texts

    try:
        return parse_document(path)
    except (PyPdfError, PackageNotFoundError, BadZipFile, XMLSyntaxError) as error:
        raise ValueError(f"无法解析 {path.name}：{error}") from error


class DocumentRouter:
    """载入文档一次，为用户的每轮提问执行检索和回答。"""

    def __init__(
        self,
        source: str | Path = DEFAULT_SOURCE,
        *,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        model: str | None = None,
    ) -> None:
        if not -1 <= min_similarity <= 1:
            raise ValueError("最低余弦相似度必须介于 -1 和 1 之间")
        self._llm = ChatCompletions(model=model)
        self._texts = load_chunks(source)
        self._min_similarity = min_similarity

    def answer(
        self,
        question: str,
        *,
        role: str = "",
        context: str = "",
        history: Sequence[ConversationTurn] = (),
    ) -> str:
        """独立检索当前问题，并用排名前三的达标片段生成回答。"""
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")
        if not self._texts:
            return INSUFFICIENT_EVIDENCE

        client = OpenRouterEmbeddings(model="nvidia/nemotron-3.5-embed-1b:free")
        ranked = rank_query(question, self._texts, client, min_similarity=self._min_similarity)
        if not ranked:
            return INSUFFICIENT_EVIDENCE

        excerpts = [item["text"] for item in ranked[:ANSWER_EXCERPT_COUNT]]
        return self._llm.answer(
            question, excerpts, role=role, context=context, history=history
        )
