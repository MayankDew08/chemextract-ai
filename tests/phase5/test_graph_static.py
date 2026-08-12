from pathlib import Path


def test_graph_visualization_defaults_prioritize_readability() -> None:
    """The graph view should not collapse labels into an unreadable center stack."""

    graph_script = Path("src/api/static/js/graph.js").read_text(encoding="utf-8")

    assert "labelHighlightBold: false" in graph_script
    assert "function visibleEdgeLabels(edgeIds)" in graph_script
    assert "edges.update(visibleEdgeLabels(" in graph_script
    assert "springLength: 210" in graph_script
    assert "network.fit" in graph_script
