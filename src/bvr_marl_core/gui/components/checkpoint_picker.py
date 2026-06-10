"""
Hierarchical checkpoint picker for Streamlit GUIs.

Presents three cascading selectboxes:
  1. Model name  (subdirectory of ``models/``)
  2. Run         (date/time extracted from the Ray Tune run folder name)
  3. Checkpoint  (``checkpoint_NNNNNN`` subdirectory; defaults to latest)

Returns the absolute path to the selected checkpoint directory, or ``None``.
"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from bvr_marl_core.utils.paths import core_project_root, models_root, sibling_model_roots

# Ray Tune run dirs look like:
#   PPO_SimplifiedMultiAgentEnv_27910_00000_0_2026-03-20_14-31-51
_RUN_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$")
_CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)$")


def _default_models_root() -> Path:
    return core_project_root() / "models"


def candidate_campaign_model_roots() -> list[Path]:
    """Return model roots to scan for campaign checkpoints.

    The core repository stores models under ``models/``.  Campaign training can
    also be launched from sibling extension repositories, so include their
    model directories when present.
    """
    roots: list[Path] = [models_root()]
    resolved_roots = {root.resolve() for root in roots}
    for sibling in sibling_model_roots():
        try:
            if sibling.exists() and sibling.resolve() not in resolved_roots:
                roots.append(sibling)
                resolved_roots.add(sibling.resolve())
        except Exception:
            continue
    return [root for root in roots if root.exists()]


def find_campaign_checkpoints(models_root: Path | None = None) -> list[tuple[float, str, str]]:
    """Return ``(mtime, label, checkpoint_path)`` for campaign-trained models.

    Campaign directories are top-level directories whose children contain a
    ``run_manifest.json`` file, matching the layout produced by the automatic
    campaign training system.  For each campaign model, the newest checkpoint
    number is returned.
    """
    roots = [models_root] if models_root is not None else candidate_campaign_model_roots()
    results: list[tuple[float, str, str]] = []

    for root in roots:
        if root is None or not root.exists():
            continue
        for campaign_dir in root.iterdir():
            if not campaign_dir.is_dir() or (campaign_dir / "run_manifest.json").exists():
                continue
            for model_dir in campaign_dir.iterdir():
                manifest = model_dir / "run_manifest.json"
                if not model_dir.is_dir() or not manifest.exists():
                    continue
                ckpts: list[Path] = []
                for run_dir in model_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    for checkpoint_dir in run_dir.iterdir():
                        if checkpoint_dir.is_dir() and _CHECKPOINT_RE.match(checkpoint_dir.name):
                            ckpts.append(checkpoint_dir)
                if not ckpts:
                    continue
                ckpts.sort(key=lambda path: int(_CHECKPOINT_RE.match(path.name).group(1)))
                label = f"{campaign_dir.name} / {model_dir.name}"
                results.append((manifest.stat().st_mtime, label, str(ckpts[-1])))

    results.sort(reverse=True)
    return results


def list_model_names(models_root: Path | None = None) -> list[str]:
    """Return sorted model name directories found under *models_root*."""
    root = models_root or _default_models_root()
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def list_runs(model_name: str, models_root: Path | None = None) -> list[tuple[str, Path]]:
    """Return ``[(display_label, run_path), ...]`` for *model_name*, newest first.

    The display label is ``YYYY-MM-DD HH:MM:SS`` extracted from the Ray Tune
    run directory name.  Directories whose name does not contain a timestamp
    are shown with their full name as fallback.
    """
    root = (models_root or _default_models_root()) / model_name
    if not root.exists():
        return []
    runs: list[tuple[str, Path]] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        m = _RUN_DATE_RE.search(d.name)
        label = f"{m.group(1)} {m.group(2).replace('-', ':')}" if m else d.name
        runs.append((label, d))
    # Sort newest-first (ISO timestamps sort lexicographically on the folder name)
    runs.sort(key=lambda x: x[1].name, reverse=True)
    return runs


def list_checkpoints(run_path: Path) -> list[tuple[str, Path]]:
    """Return ``[(display_label, checkpoint_path), ...]`` sorted ascending by number."""
    ckpts = sorted(
        (d for d in run_path.iterdir() if d.is_dir() and _CHECKPOINT_RE.match(d.name)),
        key=lambda d: int(_CHECKPOINT_RE.match(d.name).group(1)),
    )
    return [(f"checkpoint #{int(_CHECKPOINT_RE.match(d.name).group(1))}", d) for d in ckpts]


def find_train_config_for_checkpoint(checkpoint_path: str | Path) -> str | None:
    """Return the path to the training config YAML for *checkpoint_path*, or ``None``.

    Resolution order:
    1. ``models/{model_name}/train_config.yaml`` — written by train.py /
       train_simple.py at the start of every training run (present for new runs).
    2. Heuristic from the Ray Tune run-directory name, which embeds the env class
       (e.g. ``PPO_SimplifiedMultiAgentEnv_hash_...``).  When "Simplified" is found
       in the run name the first matching simplified config in ``configs/training/``
       is returned so the correct environment is created.

    Note: Ray Tune's ``params.json`` is intentionally *not* used — its structure
    differs from a training-config YAML and cannot be passed to ``load_train_config``.
    """
    ckpt = Path(checkpoint_path)
    # Structure: models/{model_name}/{run_dir}/checkpoint_{N}
    run_dir = ckpt.parent  # models/{model_name}/{run_dir}
    model_dir = run_dir.parent  # models/{model_name}

    # Primary: explicitly saved train_config.yaml (new runs only).  Campaign
    # auto-training writes the same resolved trainer config as autotune_train_config.yaml.
    for train_cfg in (
        model_dir / "train_config.yaml",
        model_dir / "autotune_train_config.yaml",
    ):
        if train_cfg.exists():
            return str(train_cfg)

    # Heuristic: Ray Tune embeds the env class name in the run directory name.
    # PPO_SimplifiedMultiAgentEnv_... → look for a simplified YAML config.
    run_name = run_dir.name
    if "Simplified" in run_name:
        from bvr_marl_core.utils.paths import core_project_root as project_root

        training_dir = project_root() / "configs" / "training"
        candidates = sorted(training_dir.glob("simple*.yaml")) if training_dir.exists() else []
        if candidates:
            return str(candidates[0])

    return None


def checkpoint_picker(
    key_prefix: str = "ckpt",
    allow_none: bool = True,
    none_label: str = "Random Actions",
    models_root: Path | None = None,
    label: str = "Checkpoint",
) -> str | None:
    """Render a three-step checkpoint selection widget.

    Args:
        key_prefix:  Unique prefix for Streamlit widget keys (avoids key clashes
                     when the widget appears more than once on a page).
        allow_none:  Whether to offer a "no checkpoint" option at the top of the
                     model list.
        none_label:  Display text for the "no checkpoint" option.
        models_root: Override the ``models/`` search root (defaults to CWD/models).
        label:       Section label shown above the first selectbox.

    Returns:
        Absolute path string to the selected ``checkpoint_NNNNNN`` directory,
        or ``None`` when *allow_none* is ``True`` and the user selects it.
    """
    model_names = list_model_names(models_root)

    st.markdown(f"**{label}**")

    # ── Step 1: model name ────────────────────────────────────────────────────
    none_option = [none_label] if allow_none else []
    no_models_hint = ["(no models found)"] if not model_names else []
    model_options = none_option + model_names + no_models_hint

    selected_model = st.selectbox(
        "Model name:",
        options=model_options,
        key=f"{key_prefix}_model",
        help="Top-level model folder inside models/",
    )

    if selected_model in (none_label, "(no models found)"):
        return None

    # ── Step 2: run (date/time) ───────────────────────────────────────────────
    runs = list_runs(selected_model, models_root)
    if not runs:
        st.warning(f"No runs found for **{selected_model}**.")
        return None

    run_labels = [lbl for lbl, _ in runs]
    selected_run_label = st.selectbox(
        "Run (date / time):",
        options=run_labels,
        index=0,  # newest run first
        key=f"{key_prefix}_run",
        help="Each training run produces one folder. Newest run is selected by default.",
    )
    run_path = next(p for lbl, p in runs if lbl == selected_run_label)

    # ── Step 3: checkpoint ────────────────────────────────────────────────────
    checkpoints = list_checkpoints(run_path)
    if not checkpoints:
        st.warning("No checkpoints found in the selected run.")
        return None

    ckpt_labels = ["Latest (default)"] + [lbl for lbl, _ in checkpoints]
    selected_ckpt_label = st.selectbox(
        "Checkpoint:",
        options=ckpt_labels,
        index=0,  # default to latest
        key=f"{key_prefix}_checkpoint",
        help="Select a specific checkpoint or keep 'Latest' to use the most recent one.",
    )

    if selected_ckpt_label == "Latest (default)":
        _, ckpt_path = checkpoints[-1]
    else:
        ckpt_path = next(p for lbl, p in checkpoints if lbl == selected_ckpt_label)

    st.caption(f"`{ckpt_path}`")
    return str(ckpt_path)
