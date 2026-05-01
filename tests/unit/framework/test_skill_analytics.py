"""Tests for skill analytics — selection tracking and hit rates.

Covers:
- Record skill selection
- Get stats per skill
- Get global stats (total matches, miss rate)
- Reset stats
"""

from __future__ import annotations

import pytest


class TestSkillAnalytics:
    """SkillAnalytics tracks selection events."""

    def test_record_selection(self):
        from victor.framework.skill_analytics import SkillAnalytics

        analytics = SkillAnalytics()
        analytics.record_selection("debug_test_failure", score=0.82)
        analytics.record_selection("debug_test_failure", score=0.75)
        analytics.record_selection("refactor_code", score=0.68)

        stats = analytics.get_skill_stats("debug_test_failure")
        assert stats["count"] == 2
        assert 0.78 < stats["avg_score"] < 0.79

    def test_record_miss(self):
        from victor.framework.skill_analytics import SkillAnalytics

        analytics = SkillAnalytics()
        analytics.record_miss()
        analytics.record_miss()
        analytics.record_selection("debug", score=0.80)

        global_stats = analytics.get_global_stats()
        assert global_stats["total_matches"] == 1
        assert global_stats["total_misses"] == 2
        assert global_stats["miss_rate"] == pytest.approx(2 / 3)

    def test_get_all_stats(self):
        from victor.framework.skill_analytics import SkillAnalytics

        analytics = SkillAnalytics()
        analytics.record_selection("debug", score=0.80)
        analytics.record_selection("refactor", score=0.65)
        analytics.record_selection("debug", score=0.90)

        all_stats = analytics.get_all_stats()
        assert len(all_stats) == 2
        assert all_stats[0]["name"] == "debug"  # sorted by count desc
        assert all_stats[0]["count"] == 2

    def test_empty_stats(self):
        from victor.framework.skill_analytics import SkillAnalytics

        analytics = SkillAnalytics()
        assert analytics.get_all_stats() == []
        global_stats = analytics.get_global_stats()
        assert global_stats["total_matches"] == 0
        assert global_stats["miss_rate"] == 0.0

    def test_reset(self):
        from victor.framework.skill_analytics import SkillAnalytics

        analytics = SkillAnalytics()
        analytics.record_selection("debug", score=0.80)
        analytics.reset()
        assert analytics.get_all_stats() == []

    def test_record_multi_skill(self):
        from victor.framework.skill_analytics import SkillAnalytics

        analytics = SkillAnalytics()
        analytics.record_multi_selection([("debug", 0.80), ("refactor", 0.70)])
        assert analytics.get_skill_stats("debug")["count"] == 1
        assert analytics.get_skill_stats("refactor")["count"] == 1
        assert analytics.get_global_stats()["multi_skill_count"] == 1
