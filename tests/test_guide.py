"""Tests for the guide system."""

from mlguide import guide


def test_guide_overview(capsys):
    """guide() prints the overview without errors."""
    guide()
    out = capsys.readouterr().out
    assert "mlguide" in out
    assert "Welcome" in out


def test_guide_all_topics(capsys):
    """Every known topic is callable without exceptions."""
    topics = [
        "ml_basics", "load", "clean", "split", "encode", "scale",
        "train", "compare", "evaluate", "predict", "save",
        "workflow", "leakage", "cheatsheet", "text", "nlp",
    ]
    for t in topics:
        guide(t)
        out = capsys.readouterr().out
        assert "Guide" in out or "Cheatsheet" in out, f"Topic '{t}' output missing header"


def test_guide_unknown_topic(capsys):
    """Unknown topics print a helpful error."""
    guide("nonexistent_topic")
    out = capsys.readouterr().out
    assert "Unknown guide topic" in out
