"""
AudioStory — Job Manager

Thread-safe in-memory registry for tracking long-running pipeline jobs.
Allows the frontend to poll for real-time progress via /jobs/{job_id}.

Each job tracks:
  - Current status (pending → running → completed/failed)
  - Current stage (1-4)
  - Progress message for the current stage
  - Console-style log entries (timestamped)
  - Error details (if failed)
  - Final result dict (if completed)

The pipeline writes to this registry as it progresses through stages.
The frontend polls GET /jobs/{job_id} to display real-time updates.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class LogEntry:
    """A single console log line from the pipeline."""
    timestamp: str
    message: str
    msg_type: str  # "info", "success", "warn", "error"

    def to_dict(self) -> Dict[str, str]:
        return {"timestamp": self.timestamp, "message": self.message, "type": self.msg_type}


@dataclass
class JobState:
    """In-memory state for a single pipeline job."""
    job_id: str
    status: str = "pending"  # pending → running → completed | failed
    current_stage: int = 0   # 0 = not started, 1-4 = stage number
    stage_progress_msg: str = ""
    logs: List[LogEntry] = field(default_factory=list)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    cancelled: bool = False

    def to_dict(self, include_logs: bool = True) -> Dict[str, Any]:
        d = {
            "job_id": self.job_id,
            "status": self.status,
            "current_stage": self.current_stage,
            "stage_progress_msg": self.stage_progress_msg,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error": self.error,
            "cancelled": self.cancelled,
        }
        if include_logs:
            d["logs"] = [e.to_dict() for e in self.logs]
        if self.result is not None:
            d["result"] = self.result
        return d


# ── Thread-safe job registry ────────────────────────────────────────────────────

_jobs: Dict[str, JobState] = {}
_lock = threading.Lock()


def create_job(job_id: str) -> JobState:
    """Create a new job entry and return it."""
    job = JobState(job_id=job_id)
    with _lock:
        _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[JobState]:
    """Retrieve a job by ID. Returns None if not found."""
    with _lock:
        return _jobs.get(job_id)


def update_job(
    job_id: str,
    *,
    status: str = None,
    current_stage: int = None,
    stage_progress_msg: str = None,
    error: str = None,
    result: Dict[str, Any] = None,
    cancelled: bool = None,
    end_time: Optional[float] = None,
) -> JobState:
    """Update a job's state. Only non-None fields are changed."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        if status is not None:
            job.status = status
        if current_stage is not None:
            job.current_stage = current_stage
        if stage_progress_msg is not None:
            job.stage_progress_msg = stage_progress_msg
        if error is not None:
            job.error = error
        if result is not None:
            job.result = result
        if cancelled is not None:
            job.cancelled = cancelled
        if status == "running" and job.start_time is None:
            job.start_time = time.time()
        if status in ("completed", "failed", "cancelled") and job.end_time is None:
            job.end_time = time.time()
        if end_time is not None:
            job.end_time = end_time
        return job


def log_progress(job_id: Optional[str], msg: str, msg_type: str = "info") -> None:
    """
    Log a progress message for a job.

    Prints to stdout (so the server console shows output) AND appends
    to the job's logs list (so the frontend can display it via polling).

    If job_id is None, only prints to stdout.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{msg_type.upper()}] {msg}")

    if job_id:
        with _lock:
            job = _jobs.get(job_id)
            if job:
                job.logs.append(LogEntry(
                    timestamp=timestamp,
                    message=msg,
                    msg_type=msg_type,
                ))


def cancel_job(job_id: str) -> bool:
    """
    Request cancellation of a running job.

    Sets the cancelled flag on the job state. The pipeline checks this
    flag at stage boundaries and aborts gracefully if set.
    Returns True if the job was found and marked for cancellation.
    """
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return False
        job.cancelled = True
        job.status = "cancelled"
        return True


def is_cancelled(job_id: str) -> bool:
    """Check if a job has been cancelled. Call this at stage boundaries."""
    with _lock:
        job = _jobs.get(job_id)
        return job.cancelled if job else False
