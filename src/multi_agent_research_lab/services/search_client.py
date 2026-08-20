"""Search client abstraction for ResearcherAgent.

Implemented as an OFFLINE mock backed by `ai_agent_offline_research_corpus_v2/`
(see that folder's README). This keeps the lab runnable without a TAVILY_API_KEY and
makes ResearcherAgent's output reproducible for grading. Swap in a real provider
(Tavily, Bing, SerpAPI) by adding a branch in `search()` when TAVILY_API_KEY is set.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument

_Document = dict[str, Any]

_DEFAULT_CORPUS_DIR = Path("ai_agent_offline_research_corpus_v2/topics")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


@lru_cache(maxsize=4)
def _load_corpus_documents(corpus_dir: str) -> tuple[_Document, ...]:
    """Flatten every topic's source_documents into one tuple, tagged with topic name."""

    path = Path(corpus_dir)
    if not path.exists():
        return ()
    documents: list[_Document] = []
    for topic_file in sorted(path.glob("*.json")):
        data = json.loads(topic_file.read_text(encoding="utf-8"))
        topic_name = data.get("topic", {}).get("name", topic_file.stem)
        for doc in data.get("knowledge_base", {}).get("source_documents", []):
            documents.append({**doc, "topic": topic_name})
    return tuple(documents)


class SearchClient:
    """Provider-agnostic search client, currently backed by the offline corpus."""

    def __init__(self, corpus_dir: Path | str = _DEFAULT_CORPUS_DIR) -> None:
        self._corpus_dir = str(corpus_dir)

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        documents = _load_corpus_documents(self._corpus_dir)
        if not documents:
            raise AgentExecutionError(
                f"No offline corpus found at '{self._corpus_dir}'. Run this from the repo "
                "root (it's a relative path), or pass SearchClient(corpus_dir=...) pointing "
                "at ai_agent_offline_research_corpus_v2/topics."
            )

        query_tokens = _tokenize(query)
        scored: list[tuple[int, _Document]] = []
        for doc in documents:
            doc_tokens = _tokenize(doc.get("title", "") + " " + doc.get("full_text", "")[:2000])
            overlap = len(query_tokens & doc_tokens)
            if overlap:
                scored.append((overlap, doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        # Take the best-scoring documents first, then pad with whatever's left (in
        # corpus order) up to max_results. Padding matters: a query might overlap
        # with only 1-2 documents even when plenty more are available, and
        # ResearcherAgent works better with `max_results` sources than with 1.
        results: list[_Document] = []
        seen_ids: set[int] = set()
        for _, doc in scored:
            if len(results) >= max_results:
                break
            results.append(doc)
            seen_ids.add(id(doc))
        if len(results) < max_results:
            for doc in documents:
                if len(results) >= max_results:
                    break
                if id(doc) in seen_ids:
                    continue
                results.append(doc)
                seen_ids.add(id(doc))

        return [
            SourceDocument(
                title=doc["title"],
                url=doc.get("provenance_url"),
                snippet=doc.get("full_text", "")[:500],
                metadata={
                    "document_id": doc.get("document_id"),
                    "topic": doc.get("topic"),
                    "is_synthetic": doc.get("is_synthetic", False),
                    "year": doc.get("year"),
                },
            )
            for doc in results
        ]
