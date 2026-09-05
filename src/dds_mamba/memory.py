from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    return float(a @ b / max(np.linalg.norm(a), eps) / max(np.linalg.norm(b), eps))


@dataclass
class Entry:
    key: np.ndarray
    weight: float
    age: float
    index: int

    def utility(self, decay: float) -> float:
        return self.weight * float(np.exp(-decay * self.age))


class RFMB:
    def __init__(self, capacity: int, decay: float, duplicate_threshold: float, refresh_delta: float) -> None:
        self.capacity, self.decay = capacity, decay
        self.duplicate_threshold, self.refresh_delta = duplicate_threshold, refresh_delta
        self.entries: list[Entry] = []
        self._next = 0

    def read(self, query: np.ndarray, threshold: float, topk: int = 5) -> np.ndarray | None:
        scored = [(cosine(query, e.key) * e.utility(self.decay), e) for e in self.entries]
        chosen = sorted((x for x in scored if x[0] >= threshold), key=lambda x: (-x[0], x[1].index))[:topk]
        if not chosen:
            return None
        scores = np.array([x[0] for x in chosen])
        weights = np.exp((scores - scores.max()) / .1); weights /= weights.sum()
        return sum(w * e.key for w, (_, e) in zip(weights, chosen))

    def age_all(self, amount: float) -> None:
        for entry in self.entries:
            entry.age += amount

    def write(self, key: np.ndarray, quality: float) -> None:
        if self.capacity <= 0:
            return
        key = key / max(np.linalg.norm(key), 1e-6)
        # A sequence may contain several entries above the duplicate threshold.
        # The method is intentionally unambiguous: refresh the *most similar*
        # one and use the oldest insertion index only for an exact tie.
        duplicates = [(cosine(key, entry.key), entry) for entry in self.entries]
        duplicates = [(score, entry) for score, entry in duplicates if score >= self.duplicate_threshold]
        duplicate = max(duplicates, key=lambda item: (item[0], -item[1].index))[1] if duplicates else None
        if duplicate is not None:
            if quality > duplicate.utility(self.decay) + self.refresh_delta:
                duplicate.key, duplicate.weight, duplicate.age = key, quality, 0.0
            return
        item = Entry(key, quality, 0.0, self._next); self._next += 1
        if len(self.entries) < self.capacity:
            self.entries.append(item); return
        victim = min(self.entries, key=lambda e: (e.utility(self.decay), e.index))
        self.entries[self.entries.index(victim)] = item
