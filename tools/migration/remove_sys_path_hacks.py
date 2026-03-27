"""
Removes sys.path.insert / sys.path.append hacks that were needed before
the package was properly installed. Also strips unused os/sys imports
left behind.

Targets files that follow the pattern:
    import os
    import sys
    from pathlib import Path
    project_root = Path(...) / ...   (or os.path.abspath(...))
    sys.path.insert(0, ...)
"""

import re
import sys
from pathlib import Path

root = Path(__file__).parent.parent.parent

# Files to strip — relative to project root
# Only strip if the hack is the ONLY reason for the os/sys import.
# For safety, we surgically remove only the specific pattern lines.

BLOCK_PATTERN = re.compile(
    r"""
    # Optional blank line before
    \n?
    # Optional comment line
    (?:\#[^\n]*\n)*
    # Optional: assignment of project_root / src_root / similar
    (?:(?:project_root|src_root|_root)\s*=\s*[^\n]+\n)?
    # The sys.path line (insert or append)
    (?:if\s+[^\n]+\n\s*)?        # optional: if ... not in sys.path
    \s*sys\.path\.(?:insert|append)\([^\n]+\)\n
    """,
    re.VERBOSE,
)


def strip_path_hacks(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    content = original

    # Remove the sys.path manipulation block
    # Strategy: find and remove lines matching the pattern
    lines = content.splitlines(keepends=True)
    new_lines = []
    i = 0
    changed = False
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect a "setup" comment like "# Add project root to path"
        if re.match(r"#\s*(add|setup|fix|ensure).*\b(path|sys\.path)\b", stripped, re.I):
            # Look ahead: if next non-empty lines are assignments + sys.path hack, skip them
            j = i + 1
            block = [line]
            # Consume optional assignment lines (project_root = ...)
            while j < len(lines) and re.match(
                r"\s*(?:project_root|src_root|_root|cwd)\s*=", lines[j]
            ):
                block.append(lines[j])
                j += 1
            # Consume optional "if ... not in sys.path" guard
            if j < len(lines) and re.match(r"\s*if\s+.*sys\.path", lines[j]):
                block.append(lines[j])
                j += 1
                # Consume the indented sys.path.insert inside the if
                if j < len(lines) and "sys.path" in lines[j]:
                    block.append(lines[j])
                    j += 1
            # Consume direct sys.path line
            elif j < len(lines) and re.match(r"\s*sys\.path\.", lines[j]):
                block.append(lines[j])
                j += 1
            if any("sys.path" in b for b in block):
                changed = True
                i = j
                continue  # skip all block lines
        # Bare sys.path line with no preceding comment
        elif re.match(r"\s*sys\.path\.(?:insert|append)\(", stripped):
            # Check if the previous line was the assignment (already consumed, or standalone)
            changed = True
            i += 1
            continue
        # Bare assignment for project_root that precedes sys.path (no comment)
        elif re.match(
            r"\s*(?:project_root|src_root)\s*=\s*(?:os\.path|Path|os\.path\.dirname)", stripped
        ):
            # Look ahead to see if next non-empty line is sys.path
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and re.match(r"\s*sys\.path\.", lines[j].strip()):
                changed = True
                i += 1
                continue

        new_lines.append(line)
        i += 1

    content = "".join(new_lines)

    # Now clean up unused imports: if 'os' or 'sys' is only imported for the hack
    # Only remove if the import is now completely unused
    for mod in ("sys", "os"):
        # Check if module is still used after stripping (excluding the import line itself)
        import_line_pattern = re.compile(rf"^import {mod}\n", re.MULTILINE)
        match = import_line_pattern.search(content)
        if match:
            rest = content[: match.start()] + content[match.end() :]
            # Usage: any reference to `mod.` or standalone `mod` in non-import lines
            if not re.search(rf"\b{mod}\b", rest):
                content = rest
                changed = True

    if content != original:
        path.write_text(content, encoding="utf-8")
        return True
    return changed


TARGET_FILES = [
    # graphics scripts
    "src/air_to_air_rl/aircrafts/graphics/nez_dlz_zones.py",
    "src/air_to_air_rl/aircrafts/graphics/sqi_aspect_angle.py",
    "src/air_to_air_rl/aircrafts/graphics/sqi_heatmap.py",
    "src/air_to_air_rl/aircrafts/graphics/sqi_range_closure.py",
    "src/air_to_air_rl/physics/graphics/air_density.py",
    "src/air_to_air_rl/physics/graphics/drag_aircraft.py",
    "src/air_to_air_rl/physics/graphics/drag_coefficient_aircraft.py",
    "src/air_to_air_rl/physics/graphics/drag_coefficient_missile.py",
    "src/air_to_air_rl/physics/graphics/drag_missile.py",
    "src/air_to_air_rl/physics/graphics/thrust_aircraft.py",
    "src/air_to_air_rl/physics/graphics/thrust_missile.py",
    "src/air_to_air_rl/physics/graphics/turn_rate_aircraft.py",
    "src/air_to_air_rl/physics/graphics/turn_rate_missile.py",
    "src/air_to_air_rl/radar/graphics/detection_probability_f22_variants.py",
    "src/air_to_air_rl/radar/graphics/detection_probability_heatmap.py",
    "src/air_to_air_rl/radar/graphics/rcs_altitude_angle.py",
    "src/air_to_air_rl/radar/graphics/rcs_horizontal_angle.py",
    # automation test scripts
    "src/air_to_air_rl/automation/scripted_control/test_scripted_control.py",
    "src/air_to_air_rl/automation/test_automation.py",
    # GUI
    "src/air_to_air_rl/gui/app.py",
    # RL configs test
    "src/air_to_air_rl/rl/configs/test_reward_config.py",
    # Tests
    "tests/aircrafts/test_auto_fire.py",
    "tests/aircrafts/test_auto_fire_controlled.py",
    "tests/aircrafts/test_auto_fire_scenarios.py",
    "tests/reinforcement_learning/environment/spaces/observation/test_nez_observation_integration.py",
    "tests/test_distance_engagement.py",
    "tests/test_tracker_error.py",
]


def main():
    changed = []
    for rel in TARGET_FILES:
        p = root / rel
        if not p.exists():
            print(f"  MISSING: {rel}")
            continue
        if strip_path_hacks(p):
            changed.append(rel)
            print(f"  Cleaned: {rel}")
        else:
            print(f"  Unchanged: {rel}")
    print(f"\n{len(changed)} files cleaned.")


if __name__ == "__main__":
    main()
