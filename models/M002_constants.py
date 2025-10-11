from __future__ import annotations

from typing import Dict, Sequence

STATE_NAMES: Sequence[str] = (
    "Accumulation",
    "EarlyUp",
    "Peak",
    "Distribution",
    "LateDown",
)

STATE_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(STATE_NAMES)}
ID_TO_STATE: Dict[int, str] = {idx: name for name, idx in STATE_TO_ID.items()}
