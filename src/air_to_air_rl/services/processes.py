"""
Process monitoring service — shared runtime process tracking for GUI components.

Provides ``ProcessRecord`` and ``ProcessMonitor`` so the visualization panel and
Tacview generator don't duplicate process-tracking logic.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil

from air_to_air_rl.core.paths import runtime_root


class ProcessRecord:
    """Serialisable record for a single managed background process."""

    def __init__(
        self,
        pid: int,
        label: str,
        command: str,
        log_file: str,
        start_time: datetime,
    ) -> None:
        self.pid = pid
        self.label = label
        self.command = command
        self.log_file = log_file
        self.start_time = start_time
        self.is_alive: bool = True
        self.failure_reason: str | None = None

    # Serialisation

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "label": self.label,
            "command": self.command,
            "log_file": self.log_file,
            "start_time": self.start_time.isoformat(),
            "is_alive": self.is_alive,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ProcessRecord:
        rec = cls(
            pid=data["pid"],
            label=data["label"],
            command=data["command"],
            log_file=data["log_file"],
            start_time=datetime.fromisoformat(data["start_time"]),
        )
        rec.is_alive = data.get("is_alive", True)
        rec.failure_reason = data.get("failure_reason")
        return rec

    # Log tail helper

    @staticmethod
    def read_log_tail(log_file: str, lines: int = 100) -> str | None:
        """Return the last *lines* non-empty lines of *log_file*, or None."""
        if not log_file:
            return None
        p = Path(log_file)
        if not p.exists():
            return None
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            segments = [s for s in re.split(r"[\r\n]", content) if s.strip()]
            tail = "\n".join(segments[-lines:]).strip()
            return tail or None
        except Exception:
            return None


class ProcessMonitor:
    """Persistent process monitor backed by a JSON file in ``runtime_root()``.

    Args:
        state_file: JSON file name (relative to ``runtime_root() / "gui"``).
                    E.g. ``"viz_processes.json"``.
    """

    def __init__(self, state_file: str) -> None:
        self._state_file = state_file

    @property
    def state_path(self) -> Path:
        return runtime_root() / "gui" / self._state_file

    # Persistence

    def load(self) -> dict[int, ProcessRecord]:
        if not self.state_path.exists():
            return {}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return {int(k): ProcessRecord.from_dict(v) for k, v in data.items()}
        except Exception:
            return {}

    def save(self, records: dict[int, ProcessRecord]) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {str(k): v.to_dict() for k, v in records.items()}
            self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass  # non-fatal; GUI will show stale data

    # CRUD

    def register(self, process: subprocess.Popen, label: str, log_file: str) -> ProcessRecord:
        """Add *process* to the tracked set and persist."""
        cmd = " ".join(process.args) if hasattr(process, "args") else "unknown"
        rec = ProcessRecord(
            pid=process.pid,
            label=label,
            command=cmd,
            log_file=log_file,
            start_time=datetime.now(),
        )
        records = self.load()
        records[process.pid] = rec
        self.save(records)
        return rec

    def update(self) -> dict[int, ProcessRecord]:
        """Refresh liveness of all tracked processes and persist."""
        records = self.load()
        for pid, rec in records.items():
            was_alive = rec.is_alive
            try:
                pp = psutil.Process(pid)
                rec.is_alive = pp.is_running()
            except psutil.NoSuchProcess:
                rec.is_alive = False
            if was_alive and not rec.is_alive and rec.failure_reason is None:
                rec.failure_reason = ProcessRecord.read_log_tail(rec.log_file)
        self.save(records)
        return records

    def terminate(self, pid: int) -> bool:
        """Kill process *pid* and all its children."""
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                )
            else:
                try:
                    parent = psutil.Process(pid)
                    for proc in [parent] + parent.children(recursive=True):
                        try:
                            proc.kill()
                        except psutil.NoSuchProcess:
                            pass
                except psutil.NoSuchProcess:
                    pass

            records = self.load()
            if pid in records:
                records[pid].is_alive = False
                if records[pid].failure_reason is None:
                    records[pid].failure_reason = ProcessRecord.read_log_tail(records[pid].log_file)
                self.save(records)
            return True
        except Exception:
            return False

    def cleanup_dead(self) -> None:
        """Remove stopped processes from the state file."""
        records = self.load()
        self.save({pid: rec for pid, rec in records.items() if rec.is_alive})
