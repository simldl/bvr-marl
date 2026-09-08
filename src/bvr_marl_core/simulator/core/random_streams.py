"""Deterministic hierarchical random streams for one simulation episode."""

from __future__ import annotations

import hashlib

import numpy as np


class EpisodeRandomStreams:
    """Create stable named NumPy generators from an episode root seed.

    Stream identity follows the stable entity ID, not spawn or dictionary order.
    Adding a new unrelated subsystem therefore does not perturb existing streams.
    """

    SCHEMA_VERSION = 1

    def __init__(self, root_seed: int | None):
        self.root_seed = int(root_seed) if root_seed is not None else 0
        self._streams: dict[tuple[str, str], np.random.Generator] = {}

    def generator(self, namespace: str, entity_id: object = "episode") -> np.random.Generator:
        key = (str(namespace), str(entity_id))
        generator = self._streams.get(key)
        if generator is None:
            digest = hashlib.sha256(f"{key[0]}\0{key[1]}".encode()).digest()
            namespace_words = np.frombuffer(digest[:16], dtype=np.uint32).tolist()
            seed_sequence = np.random.SeedSequence(
                [self.root_seed, self.SCHEMA_VERSION, *namespace_words]
            )
            generator = np.random.default_rng(seed_sequence)
            self._streams[key] = generator
        return generator

    def metadata(self) -> dict[str, int]:
        return {
            "root_seed": self.root_seed,
            "stream_schema_version": self.SCHEMA_VERSION,
        }
