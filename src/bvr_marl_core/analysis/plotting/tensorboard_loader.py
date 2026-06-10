from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_scalars(logdir: str | Path) -> pd.DataFrame:
    """Recursively load all scalar summaries from TensorBoard event files.

    Returns a DataFrame with columns: run, tag, step, wall_time, value.
    The 'run' column is the path of the event directory relative to logdir.
    """
    logdir = Path(logdir)
    rows: list[dict] = []

    event_dirs = {event_file.parent for event_file in logdir.rglob("events.out.tfevents.*")}

    for event_dir in sorted(event_dirs):
        run_name = event_dir.relative_to(logdir).as_posix()

        accumulator = EventAccumulator(str(event_dir))
        accumulator.Reload()

        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                rows.append(
                    {
                        "run": run_name,
                        "tag": tag,
                        "step": event.step,
                        "wall_time": event.wall_time,
                        "value": event.value,
                    }
                )

    return pd.DataFrame(rows, columns=["run", "tag", "step", "wall_time", "value"])
