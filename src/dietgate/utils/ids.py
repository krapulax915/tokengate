"""Request/task id generation."""
from __future__ import annotations

import uuid


def new_request_id() -> str:
    return "dg-" + uuid.uuid4().hex[:16]


def new_task_id() -> str:
    return "task-" + uuid.uuid4().hex[:16]


def new_eval_id() -> str:
    return "sev-" + uuid.uuid4().hex[:16]
