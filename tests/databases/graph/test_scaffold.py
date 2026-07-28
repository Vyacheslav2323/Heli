"""Smoke test that the graph scaffold package imports."""

from databases.graph import __doc__ as graph_doc


def test_graph_package_imports():
    assert graph_doc is not None
