import json
from pathlib import Path

from requests import HTTPError

from api._common import format_openrouter_error
from api.embed import OpenRouterEmbeddings
from api.embed.embedding_parser import EmbeddingResponseParser
from api.embed.embedding_similarity import cosine_similarity
from api.rerank import OpenRouterReranker


QUERY = "水果"


def rank_query(
    query: str, texts: list[str], client: OpenRouterEmbeddings
) -> list[dict[str, str | float]]:
    """将一个查询与所有候选匹配，再用重排序模型评分。"""
    if not texts:
        return []

    inputs = [query, *texts]
    records = EmbeddingResponseParser.parse(inputs, client.embed(inputs))
    query_vector = records[0]["embedding"]
    candidates = [
        (record["text"], cosine_similarity(query_vector, record["embedding"]))
        for record in records[1:]
    ]
    candidates.sort(key=lambda candidate: candidate[1], reverse=True)

    candidate_texts = [text for text, _ in candidates]
    reranked = OpenRouterReranker().rerank(query, candidate_texts, top_n=len(candidates))
    results = []
    for item in reranked:
        text, similarity = candidates[item["index"]]
        results.append({
            "text": text,
            "similarity": similarity,
            "relevance_score": item["relevance_score"],
        })
    return results


def main() -> None:
    """使用配置的查询文本匹配候选并输出重排序结果。"""
    query = QUERY.strip()
    if not query:
        raise SystemExit("QUERY 不能为空")

    with Path(__file__).with_name("md_lines.json").open(encoding="utf-8") as file:
        texts = json.load(file)

    client = OpenRouterEmbeddings(model="nvidia/nemotron-3.5-embed-1b:free")
    try:
        results = rank_query(query, texts, client)
    except HTTPError as error:
        raise SystemExit(format_openrouter_error(error)) from None

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
