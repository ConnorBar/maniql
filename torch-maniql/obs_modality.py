"""Observation modality constants for the two pipeline modes.

Mode "wrist_state":
    wrist (raw image 224x224x3) + state (7-dim vector)

Mode "full":
    wrist (raw image 224x224x3) + tactile (raw image 224x224x3)
    + force (420-dim vector) + state (7-dim vector)
"""

from typing import Literal

WRIST_STATE_KEYS = ("wrist", "state")
FULL_KEYS = ("wrist", "tactile", "force", "state")

IMAGE_KEYS = frozenset({"wrist", "tactile"})

VALID_MODES = ("wrist_state", "full")

Modality = Literal["wrist", "tactile", "force", "state"]

def get_split_keys(mode: str):
    if mode == "wrist_state":
        return WRIST_STATE_KEYS
    if mode == "full":
        return FULL_KEYS
    raise ValueError(f"Unknown mode {mode!r}; expected one of {VALID_MODES}")
