from victor.framework.search.hybrid import HybridSearchEngine


def test_combine_results_filters_zero_score_noise_before_rrf():
    engine = HybridSearchEngine(semantic_weight=0.6, keyword_weight=0.4, rrf_k=60)

    semantic_results = [
        {"file_path": "real_semantic.py", "content": "def real():", "score": 0.85},
        {"file_path": "zero_semantic.py", "content": "def zero():", "score": 0.0},
    ]
    keyword_results = [
        {"file_path": "real_keyword.py", "content": "def keyword():", "score": 4.0},
        {"file_path": "blank.py", "content": "def blank():", "score": 0.0},
    ]

    results = engine.combine_results(semantic_results, keyword_results, max_results=10)

    assert [result.file_path for result in results] == ["real_semantic.py", "real_keyword.py"]


def test_keyword_search_returns_only_positive_matches():
    engine = HybridSearchEngine()
    documents = [
        {"file_path": "match.py", "content": "foo foo bar"},
        {"file_path": "miss.py", "content": "baz qux"},
    ]

    results = engine.keyword_search("foo", documents, max_results=10)

    assert len(results) == 1
    assert results[0]["file_path"] == "match.py"
    assert results[0]["score"] == 2.0
