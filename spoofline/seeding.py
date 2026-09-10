"""Seeding helpers.

Every stochastic component draws from a generator derived from a single run seed
plus a string label, so adding a component never shifts the draws of an existing
one and a clip can be regenerated in isolation.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import random

import numpy as np
import torch


def derive_seed(seed: int, label: str) -> int:
    """Derive a stable 31 bit sub-seed from a run seed and a label."""
    digest = hashlib.blake2b(f"{seed}:{label}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def rng(seed: int, label: str) -> np.random.Generator:
    """A numpy generator for a named component."""
    return np.random.default_rng(derive_seed(seed, label))


def seed_everything(seed: int, threads: int | None = None) -> None:
    """Seed python, numpy and torch, and pin CPU threading for reproducibility."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if threads is not None:
        torch.set_num_threads(int(threads))
        # Interop pool may already be started; the intra-op count above is what
        # determines reduction order for our workloads.
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(int(threads))
    torch.use_deterministic_algorithms(True, warn_only=True)
