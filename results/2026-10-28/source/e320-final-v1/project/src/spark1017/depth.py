"""Acceptance-calibrated joint speculation with a bounded row-budget solver.

Both TP ranks see the same confidence logits and target-verification outcomes.
Updates use integer counts, never local clocks, so their next plan is identical.
The hardware cost curve is frozen before evaluation. Only verified target tokens
are emitted; this module never substitutes draft output for target verification.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

def probability(logit: float) -> float:
    if not math.isfinite(logit):
        raise ValueError("non-finite draft confidence")
    return 1 / (1 + math.exp(-max(-80.0, min(80.0, logit))))

def best_depths(curves: list[list[float]], live: int, max_rows: int,
                costs: list[float], draft_cost: float) -> list[int]:
    """Exact dynamic program, O(streams * row_budget * max_depth).

    curve[k] is 1 + the expected accepted prefix length at depth k. Nondrafting
    streams contribute one target token and consume one row like drafting ones.
    Stable iteration order provides deterministic ties across TP ranks.
    """
    if live < len(curves) or live < 1 or max_rows < live or len(costs) < max_rows:
        raise ValueError("invalid live streams, row budget or cost curve")
    if not math.isfinite(draft_cost) or draft_cost < 0 or any(not math.isfinite(x) or x <= 0 for x in costs):
        raise ValueError("invalid hardware costs")
    capacity = max_rows - live
    states = {0: (float(live - len(curves)), ())}
    for curve in curves:
        if not curve or curve[0] != 1.0 or any(not math.isfinite(x) for x in curve):
            raise ValueError("invalid survival curve")
        nxt = {}
        for used, (expected, path) in sorted(states.items()):
            for k, gain in enumerate(curve[:capacity - used + 1]):
                rows = used + k
                candidate = (expected + gain, path + (k,))
                old = nxt.get(rows)
                if old is None or candidate[0] > old[0]:
                    nxt[rows] = candidate
        states = nxt
    best, value = None, -1.0
    for used, (expected, path) in sorted(states.items()):
        score = expected / (costs[live + used - 1] + draft_cost)
        if score > value:
            best, value = path, score
    return list(best)

@dataclass
class Cell:
    seen: int = 0
    kept: int = 0

class AdaptiveDepth:
    def __init__(self, costs, draft_cost=3.5, prior=32, depth=5, bins=10):
        if prior <= 0 or depth <= 0 or bins <= 1:
            raise ValueError("positive prior, depth, and at least two bins required")
        self.costs = [float(x) for x in costs]
        self.draft_cost, self.prior, self.bins = float(draft_cost), int(prior), int(bins)
        self.cells = [[Cell() for _ in range(bins)] for _ in range(depth)]
        self.observations = 0

    def _cell(self, depth, raw):
        return self.cells[depth][min(self.bins - 1, int(raw * self.bins))]

    def curve(self, confs, limit):
        result, survival = [1.0], 1.0
        for j, logit in enumerate(confs[:min(limit, len(self.cells))]):
            raw = probability(float(logit))
            cell = self._cell(j, raw)
            p = (self.prior * raw + cell.kept) / (self.prior + cell.seen)
            survival *= p
            result.append(result[-1] + survival)
        return result

    def choose(self, confs, limits, live, max_rows):
        if len(confs) != len(limits):
            raise ValueError("one depth limit per drafting stream")
        curves = [self.curve(cf, max(0, limit)) for cf, limit in zip(confs, limits)]
        return best_depths(curves, live, max_rows, self.costs, self.draft_cost)

    def observe(self, confs, verified, accepted):
        if not 0 <= accepted <= verified <= len(confs):
            raise ValueError("accepted prefix must be within the verified draft")
        # Rejected suffixes were never reached. Treating them as failures biases
        # the estimator and suppresses useful speculation at deeper positions.
        reached = min(verified, accepted + 1, len(self.cells))
        for j in range(reached):
            cell = self._cell(j, probability(float(confs[j])))
            cell.seen += 1
            cell.kept += int(j < accepted)
            self.observations += 1
            # Bounded memory of old workloads; integer updates match on ranks.
            if cell.seen >= 1024:
                cell.seen //= 2
                cell.kept //= 2

    def snapshot(self):
        return {"observations": self.observations,
                "cells": [[[c.seen, c.kept] for c in row] for row in self.cells]}
