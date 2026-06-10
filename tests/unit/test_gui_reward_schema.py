import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

sys.modules.setdefault("streamlit", SimpleNamespace(warning=lambda *_args, **_kwargs: None))

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "bvr_marl_core"
    / "gui"
    / "components"
    / "config_builder.py"
)
_SPEC = importlib.util.spec_from_file_location("_config_builder_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CONFIG_BUILDER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CONFIG_BUILDER
_SPEC.loader.exec_module(_CONFIG_BUILDER)

create_default_config = _CONFIG_BUILDER.create_default_config
migrate_reward_schema = _CONFIG_BUILDER.migrate_reward_schema
reward_view_from_config = _CONFIG_BUILDER.reward_view_from_config
write_reward_view_to_config = _CONFIG_BUILDER.write_reward_view_to_config


def test_gui_migrates_legacy_reward_config_and_removes_old_key() -> None:
    cfg = {
        "env": {},
        "training": {
            "reward_config": {
                "enable_terminal": False,
                "enable_tactical": True,
                "kill_reward": 275.0,
            }
        },
    }

    migrate_reward_schema(cfg)

    assert "reward_config" not in cfg["training"]
    assert cfg["env"]["reward_categories"]["terminal"] is False
    assert cfg["env"]["reward_categories"]["tactical"] is True
    assert cfg["env"]["reward_magnitudes"]["kill_reward"] == 275.0


def test_gui_reward_view_writes_canonical_schema_only() -> None:
    cfg = create_default_config()
    view = reward_view_from_config(cfg)
    view["enable_line_objective"] = False
    view["kill_reward"] = 325.0

    write_reward_view_to_config(cfg, view)

    assert cfg["env"]["reward_categories"]["line_objective"] is False
    assert cfg["env"]["reward_magnitudes"]["kill_reward"] == 325.0
    assert "reward_config" not in cfg["training"]
