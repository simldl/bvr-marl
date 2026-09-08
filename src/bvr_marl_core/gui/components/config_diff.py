"""Nested configuration diff helpers shared by Streamlit config builders."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

ConfigDiff = dict[str, dict[str, dict[str, Any]]]
ConfigSection = tuple[str, Any, Any, str]

_MISSING = "NOT_SET"


def collect_config_differences(sections: Iterable[ConfigSection]) -> ConfigDiff:
    differences: ConfigDiff = {}

    def compare_section(section_name: str, section1: Any, section2: Any, path: str = "") -> None:
        if section_name not in differences:
            differences[section_name] = {}

        if isinstance(section1, dict) and isinstance(section2, dict):
            for key in set(section1.keys()) | set(section2.keys()):
                key_path = f"{path}.{key}" if path else key
                val1 = section1.get(key, _MISSING)
                val2 = section2.get(key, _MISSING)

                if isinstance(val1, dict) and isinstance(val2, dict):
                    compare_section(section_name, val1, val2, key_path)
                elif val1 != val2:
                    differences[section_name][key_path] = {
                        "config1": val1,
                        "config2": val2,
                    }
        elif section1 != section2:
            differences[section_name][path or section_name] = {
                "config1": section1,
                "config2": section2,
            }

    for section_name, section1, section2, path in sections:
        compare_section(section_name, section1, section2, path)

    return {key: value for key, value in differences.items() if value}
