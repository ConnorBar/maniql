"""Shared observation modality constants for split (nested dict) storage.

When ``metadata["modality_storage"] == "split"``, ``obs`` and ``next_obs`` are
dicts with keys ``pass`` (wrist+state), ``tact`` (tactile volume), ``forcefield``.
"""

SPLIT_KEYS = ("pass", "tact", "forcefield")
