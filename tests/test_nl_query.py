"""Tests for nl_query module."""

from nl_query.openai_handler import parse_user_text, map_keywords_to_segments


def test_parse_user_text_finds_trees():
    segments = parse_user_text("How much area is covered by trees?")
    assert "tree" in segments


def test_parse_user_text_finds_multiple():
    segments = parse_user_text("Show me water and buildings")
    assert "water" in segments
    assert "building" in segments


def test_parse_user_text_no_match():
    segments = parse_user_text("What is the weather today?")
    assert segments == []


def test_map_keywords_to_segments():
    result = map_keywords_to_segments(["river", "houses", "forest"])
    assert "water" in result
    assert "building" in result
    assert "tree" in result


def test_map_keywords_to_segments_dedup():
    result = map_keywords_to_segments(["river", "lake", "pond"])
    assert result.count("water") == 1


def test_parse_user_text_road_keywords():
    segments = parse_user_text("Find all roads and streets")
    assert "road" in segments
