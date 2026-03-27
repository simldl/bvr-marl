"""
Tests for episode_metrics.py - Episode metrics collection.
"""

import csv
import os
import tempfile

import pytest

from air_to_air_rl.rl.evaluation.episode_metrics import (
    EpisodeMetrics,
    MetricsCollector,
)


class TestEpisodeMetrics:
    """Tests for EpisodeMetrics dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        metrics = EpisodeMetrics()

        assert metrics.episode_id == 0
        assert metrics.scenario_name == ""
        assert metrics.winner == ""
        assert metrics.agent_missiles_fired == 0
        assert metrics.agent_kills == 0
        assert metrics.agent_time_to_first_detection_s == -1.0

    def test_custom_values(self):
        """Should accept custom values."""
        metrics = EpisodeMetrics(
            episode_id=5,
            scenario_name="test_scenario",
            agent_aircraft_type="F22",
            opponent_aircraft_type="Su57",
            agent_missiles_fired=3,
            agent_kills=2,
        )

        assert metrics.episode_id == 5
        assert metrics.scenario_name == "test_scenario"
        assert metrics.agent_aircraft_type == "F22"
        assert metrics.opponent_aircraft_type == "Su57"
        assert metrics.agent_missiles_fired == 3
        assert metrics.agent_kills == 2

    def test_compute_derived_metrics_shots_per_kill(self):
        """Should compute shots per kill correctly."""
        metrics = EpisodeMetrics(
            agent_missiles_fired=6,
            agent_kills=2,
            opponent_missiles_fired=4,
            opponent_kills=1,
        )
        metrics.compute_derived_metrics()

        assert metrics.agent_shots_per_kill == 3.0
        assert metrics.opponent_shots_per_kill == 4.0

    def test_compute_derived_metrics_accuracy(self):
        """Should compute missile accuracy correctly."""
        metrics = EpisodeMetrics(
            agent_missiles_fired=10,
            agent_missiles_hit=4,
            opponent_missiles_fired=5,
            opponent_missiles_hit=3,
        )
        metrics.compute_derived_metrics()

        assert metrics.agent_missile_accuracy == 0.4
        assert metrics.opponent_missile_accuracy == 0.6

    def test_compute_derived_metrics_zero_kills(self):
        """Should handle zero kills gracefully."""
        metrics = EpisodeMetrics(
            agent_missiles_fired=5,
            agent_kills=0,
        )
        metrics.compute_derived_metrics()

        assert metrics.agent_shots_per_kill == 0.0

    def test_compute_derived_metrics_zero_fired(self):
        """Should handle zero fired gracefully."""
        metrics = EpisodeMetrics(
            agent_missiles_fired=0,
            agent_missiles_hit=0,
        )
        metrics.compute_derived_metrics()

        assert metrics.agent_missile_accuracy == 0.0

    def test_to_dict(self):
        """Should convert to dictionary."""
        metrics = EpisodeMetrics(
            episode_id=1,
            scenario_name="test",
            agent_kills=2,
        )
        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert d["episode_id"] == 1
        assert d["scenario_name"] == "test"
        assert d["agent_kills"] == 2


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_init(self):
        """Should initialize with default output directory."""
        collector = MetricsCollector()
        assert collector.output_dir == "study_results"
        assert len(collector.episodes) == 0

    def test_init_custom_output_dir(self):
        """Should accept custom output directory."""
        collector = MetricsCollector(output_dir="custom_dir")
        assert collector.output_dir == "custom_dir"

    def test_start_episode(self):
        """Should create a new episode."""
        collector = MetricsCollector()
        ep = collector.start_episode(
            scenario_name="test_scenario",
            agent_aircraft_type="F22",
            num_agents=2,
        )

        assert ep.episode_id == 1
        assert ep.scenario_name == "test_scenario"
        assert ep.agent_aircraft_type == "F22"
        assert ep.num_agents == 2

    def test_episode_id_increments(self):
        """Episode IDs should increment."""
        collector = MetricsCollector()

        collector.start_episode()
        collector.end_episode("agent", "test", 100.0, 100, 2, 0)

        ep2 = collector.start_episode()
        assert ep2.episode_id == 2

    def test_record_detection(self):
        """Should record first detection time."""
        collector = MetricsCollector()
        ep = collector.start_episode()

        collector.record_detection("agent", 15.0)
        assert ep.agent_time_to_first_detection_s == 15.0

        # Second detection should not overwrite
        collector.record_detection("agent", 20.0)
        assert ep.agent_time_to_first_detection_s == 15.0

    def test_record_shot(self):
        """Should record missile shots."""
        collector = MetricsCollector()
        ep = collector.start_episode()

        collector.record_shot("agent", 10.0)
        collector.record_shot("agent", 15.0)
        collector.record_shot("opponent", 12.0)

        assert ep.agent_missiles_fired == 2
        assert ep.opponent_missiles_fired == 1
        assert ep.agent_time_to_first_shot_s == 10.0
        assert ep.opponent_time_to_first_shot_s == 12.0

    def test_record_hit(self):
        """Should record missile hits."""
        collector = MetricsCollector()
        ep = collector.start_episode()

        collector.record_hit("agent")
        collector.record_hit("agent")
        collector.record_hit("opponent")

        assert ep.agent_missiles_hit == 2
        assert ep.opponent_missiles_hit == 1

    def test_record_kill(self):
        """Should record kills."""
        collector = MetricsCollector()
        ep = collector.start_episode()

        collector.record_kill("agent", 1, 30.0)
        collector.record_kill("agent", 2, 45.0)
        collector.record_kill("opponent", 3, 50.0)

        assert ep.agent_kills == 2
        assert ep.opponent_kills == 1
        assert ep.agent_time_to_first_kill_s == 30.0
        assert ep.opponent_time_to_first_kill_s == 50.0

    def test_record_boundary_violation(self):
        """Should record boundary violations."""
        collector = MetricsCollector()
        ep = collector.start_episode()

        collector.record_boundary_violation("agent")
        collector.record_boundary_violation("agent")
        collector.record_boundary_violation("opponent")

        assert ep.agent_boundary_violations == 2
        assert ep.opponent_boundary_violations == 1

    def test_record_countermeasure(self):
        """Should record countermeasure usage."""
        collector = MetricsCollector()
        ep = collector.start_episode()

        collector.record_countermeasure("agent")
        collector.record_countermeasure("opponent")
        collector.record_countermeasure("opponent")

        assert ep.agent_countermeasures_used == 1
        assert ep.opponent_countermeasures_used == 2

    def test_record_reward(self):
        """Should accumulate rewards."""
        collector = MetricsCollector()
        ep = collector.start_episode()

        collector.record_reward("agent", 10.0)
        collector.record_reward("agent", 5.0)
        collector.record_reward("opponent", -3.0)

        assert ep.agent_total_reward == 15.0
        assert ep.opponent_total_reward == -3.0

    def test_end_episode(self):
        """Should finalize episode metrics."""
        collector = MetricsCollector()
        collector.start_episode()

        collector.record_shot("agent", 5.0)
        collector.record_shot("agent", 10.0)
        collector.record_kill("agent", 1, 20.0)

        collector.end_episode(
            winner="agent",
            termination_reason="all_enemies_dead",
            episode_duration_s=100.0,
            episode_steps=100,
            agent_survivors=2,
            opponent_survivors=0,
        )

        assert len(collector.episodes) == 1
        stored = collector.episodes[0]
        assert stored.winner == "agent"
        assert stored.termination_reason == "all_enemies_dead"
        assert stored.episode_duration_s == 100.0
        assert stored.agent_survivors == 2
        assert stored.opponent_survivors == 0
        # Derived metrics should be computed
        assert stored.agent_shots_per_kill == 2.0

    def test_get_current_episode(self):
        """Should return current episode during tracking."""
        collector = MetricsCollector()

        assert collector.get_current_episode() is None

        ep = collector.start_episode()
        assert collector.get_current_episode() == ep

        collector.end_episode("draw", "timeout", 100.0, 100, 1, 1)
        assert collector.get_current_episode() is None

    def test_get_episode_by_id(self):
        """Should retrieve episode by ID."""
        collector = MetricsCollector()

        collector.start_episode(scenario_name="ep1")
        collector.end_episode("agent", "test", 100.0, 100, 2, 0)

        collector.start_episode(scenario_name="ep2")
        collector.end_episode("opponent", "test", 100.0, 100, 0, 2)

        ep1 = collector.get_episode(1)
        ep2 = collector.get_episode(2)

        assert ep1.scenario_name == "ep1"
        assert ep2.scenario_name == "ep2"
        assert collector.get_episode(999) is None

    def test_get_statistics(self):
        """Should compute aggregate statistics."""
        collector = MetricsCollector()

        # Episode 1: Agent wins
        collector.start_episode()
        collector.record_kill("agent", 1, 30.0)
        collector.record_kill("agent", 2, 40.0)
        collector.record_shot("agent", 10.0)
        collector.record_shot("agent", 15.0)
        collector.end_episode("agent", "all_dead", 100.0, 100, 2, 0)

        # Episode 2: Opponent wins
        collector.start_episode()
        collector.record_kill("opponent", 1, 50.0)
        collector.record_shot("opponent", 20.0)
        collector.end_episode("opponent", "all_dead", 150.0, 150, 0, 2)

        # Episode 3: Draw
        collector.start_episode()
        collector.end_episode("draw", "timeout", 200.0, 200, 1, 1)

        stats = collector.get_statistics()

        assert stats["num_episodes"] == 3
        assert abs(stats["agent_win_rate"] - 1 / 3) < 0.01
        assert abs(stats["opponent_win_rate"] - 1 / 3) < 0.01
        assert abs(stats["draw_rate"] - 1 / 3) < 0.01
        assert stats["mean_duration_s"] == 150.0

    def test_get_statistics_empty(self):
        """Should return empty dict when no episodes."""
        collector = MetricsCollector()
        assert collector.get_statistics() == {}

    def test_export_to_csv(self):
        """Should export episodes to CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(output_dir=tmpdir)

            collector.start_episode(scenario_name="test1")
            collector.record_kill("agent", 1, 30.0)
            collector.end_episode("agent", "test", 100.0, 100, 2, 0)

            collector.start_episode(scenario_name="test2")
            collector.end_episode("draw", "timeout", 150.0, 150, 1, 1)

            collector.export_to_csv("test_results.csv", append=False)

            filepath = os.path.join(tmpdir, "test_results.csv")
            assert os.path.exists(filepath)

            with open(filepath, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert rows[0]["scenario_name"] == "test1"
            assert rows[1]["scenario_name"] == "test2"

    def test_export_to_csv_append(self):
        """Should append to existing CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(output_dir=tmpdir)

            collector.start_episode(scenario_name="first")
            collector.end_episode("agent", "test", 100.0, 100, 2, 0)
            collector.export_to_csv("test_results.csv", append=False)

            collector.clear()

            collector.start_episode(scenario_name="second")
            collector.end_episode("draw", "test", 100.0, 100, 1, 1)
            collector.export_to_csv("test_results.csv", append=True)

            filepath = os.path.join(tmpdir, "test_results.csv")
            with open(filepath, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2

    def test_clear(self):
        """Should clear all stored episodes."""
        collector = MetricsCollector()

        collector.start_episode()
        collector.end_episode("agent", "test", 100.0, 100, 2, 0)

        collector.start_episode()
        collector.end_episode("draw", "test", 100.0, 100, 1, 1)

        assert len(collector.episodes) == 2

        collector.clear()

        assert len(collector.episodes) == 0
        assert collector.get_current_episode() is None

    def test_no_recording_without_episode(self):
        """Recording methods should handle no active episode."""
        collector = MetricsCollector()

        # These should not raise
        collector.record_detection("agent", 10.0)
        collector.record_shot("agent", 10.0)
        collector.record_hit("agent")
        collector.record_kill("agent", 1, 10.0)
        collector.record_boundary_violation("agent")
        collector.record_countermeasure("agent")
        collector.record_reward("agent", 10.0)
