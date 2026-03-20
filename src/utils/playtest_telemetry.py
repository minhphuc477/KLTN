"""
Playtest telemetry collection utilities.

This module records structured session/event logs that can be used for:
- validation of generated levels with real user traces,
- persona behavior analysis,
- difficulty model calibration against play data.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PlaytestEvent:
    """One playtest event captured during a session."""

    t_rel_sec: float
    event_type: str
    step_index: Optional[int] = None
    position: Optional[Tuple[int, int]] = None
    inventory: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlaytestSession:
    """Playtest session payload persisted to JSON/JSONL."""

    session_id: str
    started_at_utc: str
    context: Dict[str, Any] = field(default_factory=dict)
    events: List[PlaytestEvent] = field(default_factory=list)
    finished_at_utc: Optional[str] = None
    status: str = "running"
    summary: Dict[str, Any] = field(default_factory=dict)


class PlaytestTelemetryCollector:
    """
    Lightweight event logger for generation/playtest sessions.

    Typical flow:
        collector.start_session("session_001", context={...})
        collector.log_event("replay_step", step_index=1, position=(3, 7))
        collector.finish_session(summary={...})
    """

    def __init__(
        self,
        output_dir: str | Path = Path("results") / "playtest",
        *,
        append_jsonl: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.append_jsonl = bool(append_jsonl)
        self.current_session: Optional[PlaytestSession] = None
        self._start_time: float = 0.0

    def start_session(
        self,
        session_id: str,
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.current_session is not None and self.current_session.status == "running":
            logger.warning(
                "Closing active session '%s' before starting '%s'.",
                self.current_session.session_id,
                session_id,
            )
            self.finish_session(status="interrupted")

        self.current_session = PlaytestSession(
            session_id=str(session_id),
            started_at_utc=_utc_now_iso(),
            context=dict(context or {}),
        )
        self._start_time = time.time()

    def log_event(
        self,
        event_type: str,
        *,
        step_index: Optional[int] = None,
        position: Optional[Sequence[int]] = None,
        inventory: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.current_session is None:
            logger.warning("log_event('%s') ignored: no active session.", event_type)
            return
        pos_tuple: Optional[Tuple[int, int]] = None
        if position is not None:
            try:
                pos_tuple = (int(position[0]), int(position[1]))
            except Exception:
                pos_tuple = None

        evt = PlaytestEvent(
            t_rel_sec=float(max(0.0, time.time() - self._start_time)),
            event_type=str(event_type),
            step_index=int(step_index) if step_index is not None else None,
            position=pos_tuple,
            inventory=dict(inventory or {}),
            payload=dict(payload or {}),
        )
        self.current_session.events.append(evt)

    def finish_session(
        self,
        *,
        status: str = "completed",
        summary: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        if self.current_session is None:
            return None
        self.current_session.finished_at_utc = _utc_now_iso()
        self.current_session.status = str(status)
        self.current_session.summary = dict(summary or {})
        if "event_count" not in self.current_session.summary:
            self.current_session.summary["event_count"] = int(len(self.current_session.events))
        if "duration_sec" not in self.current_session.summary:
            self.current_session.summary["duration_sec"] = float(max(0.0, time.time() - self._start_time))

        path = self.export_session()
        if self.append_jsonl:
            self._append_jsonl(self.current_session)
        self.current_session = None
        self._start_time = 0.0
        return path

    def export_session(self, path: Optional[str | Path] = None) -> Path:
        if self.current_session is None:
            raise RuntimeError("No active playtest session to export.")
        if path is None:
            path = self.output_dir / f"{self.current_session.session_id}.json"
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = self._session_to_dict(self.current_session)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    def _append_jsonl(self, session: PlaytestSession) -> None:
        out = self.output_dir / "playtest_sessions.jsonl"
        payload = self._session_to_dict(session)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")))
            f.write("\n")

    @staticmethod
    def _session_to_dict(session: PlaytestSession) -> Dict[str, Any]:
        payload = asdict(session)
        # Normalize tuples for stable JSON.
        for evt in payload.get("events", []):
            if isinstance(evt.get("position"), tuple):
                evt["position"] = list(evt["position"])
        return payload

