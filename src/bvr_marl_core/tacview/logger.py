import os
import threading
from datetime import UTC, datetime

from bvr_marl_core.tacview.properties import TacviewPropertyDict


class TacviewLogger:
    def __init__(self, filename=None):
        if filename is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            logs_dir = os.path.join(base_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            filename = os.path.join(logs_dir, "sim_realtime.acmi")
        else:
            logs_dir = os.path.dirname(os.path.abspath(filename))
            os.makedirs(logs_dir, exist_ok=True)
        self.filename = filename
        self._lock = threading.Lock()
        self._removed_ids = set()
        self._initialized_units = set()
        self._last_known_ids = set()
        self._last_frame = -1
        with open(self.filename, "w") as f:
            ref_time = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            f.write("FileType=text/acmi/tacview\nFileVersion=2.2\n")
            f.write(f"0,ReferenceTime={ref_time}\n")

    def log_tick(self, sim):
        with self._lock:
            sim_time = int(sim.elapsed_time_s)
            lines = []
            if sim_time != getattr(self, "_last_frame", -1):
                lines.append(f"#{sim_time}\n")
                self._last_frame = sim_time

            current_ids = set(unit.id for unit in sim.active_units.values())
            for removed_id in self._last_known_ids - current_ids:
                if removed_id not in self._removed_ids:
                    lines.append(f"-{removed_id}\n")
                    self._removed_ids.add(removed_id)
            self._last_known_ids = current_ids.copy()

            for unit in sim.active_units.values():
                pos = unit.position
                lon, lat, alt = pos.lon, pos.lat, pos.alt
                base = f"{unit.id},T={lon:.6f}|{lat:.6f}|{alt:.1f}"
                propdict = TacviewPropertyDict(unit)
                if unit.id not in self._initialized_units:
                    line = base + "," + ",".join(f"{k}={v}" for k, v in propdict.items()) + "\n"
                    self._initialized_units.add(unit.id)
                else:
                    line = (
                        base
                        + ","
                        + ",".join(
                            f"{k}={v}"
                            for k, v in propdict.items()
                            if k not in ["Name", "Type", "Color"]
                        )
                        + "\n"
                    )
                lines.append(line)
            with open(self.filename, "a") as f:
                f.writelines(lines)
