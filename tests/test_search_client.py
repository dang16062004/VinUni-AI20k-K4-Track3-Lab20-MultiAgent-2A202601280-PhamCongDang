"""Tests for the offline SearchClient (no network, no API key)."""

from multi_agent_research_lab.services.search_client import SearchClient


def test_search_returns_relevant_documents() -> None:
    client = SearchClient()
    results = client.search("multi-agent orchestration and role specialization", max_results=3)

    assert 1 <= len(results) <= 3
    for doc in results:
        assert doc.title
        assert doc.snippet
        assert doc.metadata.get("document_id")


def test_search_falls_back_to_some_documents_for_an_unrelated_query() -> None:
    client = SearchClient()
    results = client.search("zzzz totally unrelated nonsense zzzz", max_results=2)
    assert len(results) == 2
