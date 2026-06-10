"""Smoke tests: verify bvr_marl_core packages import without errors."""

import pytest


def test_rl_imports():
    from bvr_marl_core import rl  # noqa: F401


def test_training_imports():
    from bvr_marl_core import training  # noqa: F401


def test_visualization_imports():
    from bvr_marl_core import visualization  # noqa: F401


def test_tacview_imports():
    from bvr_marl_core import tacview  # noqa: F401


def test_analysis_imports():
    from bvr_marl_core import analysis  # noqa: F401


def test_services_imports():
    from bvr_marl_core import services  # noqa: F401
