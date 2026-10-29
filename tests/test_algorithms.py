import itertools
import math
import random
import unittest
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from spark1017.depth import AdaptiveDepth, best_depths
from spark1017.planning import GIB, plan_storage
from spark1017.lookup import SuffixIndex, Arbiter
from spark1017.configuration import verify_peers

class Algorithms(unittest.TestCase):
    def test_peer_configuration_mismatch_fails_on_both_ranks(self):
        class Store:
            def __init__(self): self.values={}; self.condition=threading.Condition()
            def set(self,key,value):
                with self.condition:
                    self.values[key]=value; self.condition.notify_all()
            def get(self,key):
                with self.condition:
                    if not self.condition.wait_for(lambda:key in self.values,timeout=2):
                        raise TimeoutError('missing peer')
                    return self.values[key]
        for digests in [('same','same'),('one','different')]:
            store=Store()
            with ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(verify_peers,SimpleNamespace(rank=i,world=2,nccl=SimpleNamespace(store=store)),digest)
                         for i,digest in enumerate(digests)]
                for future in futures:
                    if digests[0]==digests[1]: future.result(timeout=3)
                    else:
                        with self.assertRaisesRegex(RuntimeError,'differ across ranks'):
                            future.result(timeout=3)

    def test_lookup_committed_continuation_and_ring_bounds(self):
        index = SuffixIndex(capacity=128)
        pattern = list(range(32))
        index.append(pattern + [999] * 30 + pattern)
        self.assertEqual(index.propose(), ([999] * 15, 32))
        rng = random.Random(1017); history = []
        bank = SuffixIndex(capacity=128)
        for step in range(3000):
            chunk = ([rng.randrange(100) for _ in range(rng.randrange(1, 6))]
                     if step % 10 or len(history) < 80 else history[-80:-48])
            bank.append(chunk); history += chunk
            end = len(history); floor = max(0, end - 128)
            key = history[-8:]
            candidates = [p for p in range(floor + 8, end + 1) if history[p-8:p] == key][-4:]
            best = None
            for pos in candidates:
                if pos == end: continue
                n = 8
                while n < 32 and pos-n-1 >= floor and history[pos-n-1] == history[end-n-1]:
                    n += 1
                count = min(15, end-n-pos)
                if n >= 16 and count > 0 and (best is None or (n,pos) > best[:2]):
                    best = n,pos,count
            expected = ([],0) if best is None else (history[best[1]:best[1]+best[2]],best[0])
            self.assertEqual(bank.propose(), expected)
            self.assertLessEqual(len(bank.index), 128)
            self.assertLessEqual(len(bank.order), 128)

    def test_lookup_rejects_unproductive_proposals_deterministically(self):
        a, b = Arbiter(), Arbiter()
        costs = [30 + 3.6 * i for i in range(24)]
        for _ in range(100):
            self.assertEqual(a.choose([1]*15, 24, costs), b.choose([1]*15, 24, costs))
            a.observe(24,15,0); b.observe(24,15,0)
        self.assertEqual(a.choose([1]*15,24,costs), 0)
        self.assertEqual(a.cells[1][1], [0,0])

    def test_no_memory_overcommit(self):
        for available in range(40):
            for scales in range(8):
                p = plan_storage(available * GIB, scales * GIB)
                used = p.resident_scale_bytes + p.hot_row_budget
                self.assertLessEqual(used, min(4 * GIB, max(0, (available - 12) * GIB)))
                self.assertIn(p.resident_scale_bytes, (0, scales * GIB))

    def test_dp_matches_exhaustive_global_optimum(self):
        rng = random.Random(1017)
        for _ in range(1000):
            n = rng.randint(1, 5); live = n + rng.randint(0, 2); budget = live + rng.randint(0, 10)
            curves = []
            for _ in range(n):
                curve, survival = [1.0], 1.0
                for _ in range(rng.randint(0, 5)):
                    survival *= rng.random(); curve.append(curve[-1] + survival)
                curves.append(curve)
            costs = [20 + 3.1 * x for x in range(budget)]
            chosen = best_depths(curves, live, budget, costs, 3.5)
            def score(ks):
                return (live - n + sum(c[k] for c, k in zip(curves, ks))) / (costs[live + sum(ks) - 1] + 3.5)
            optimum = max(score(ks) for ks in itertools.product(*(range(len(c)) for c in curves)) if live + sum(ks) <= budget)
            self.assertAlmostEqual(score(chosen), optimum, places=12)

    def test_censored_suffix_is_not_a_failure(self):
        a = AdaptiveDepth([30 + 4 * i for i in range(16)])
        a.observe([0.] * 5, 5, 1)
        self.assertEqual(a.snapshot()['cells'][0][5], [1, 1])
        self.assertEqual(a.snapshot()['cells'][1][5], [1, 0])
        self.assertEqual(sum(c.seen for row in a.cells[2:] for c in row), 0)

    def test_tp_ranks_plan_identically_after_deterministic_replay(self):
        a, b = [AdaptiveDepth([30 + 4 * i for i in range(24)]) for _ in range(2)]
        rng = random.Random(31017)
        for _ in range(5000):
            cf = [[rng.uniform(-3, 6) for _ in range(5)] for _ in range(2)]
            ka = a.choose(cf, [5, 5], 2, 12)
            self.assertEqual(ka, b.choose(cf, [5, 5], 2, 12))
            for logits, k in zip(cf, ka):
                kept = rng.randint(0, k)
                a.observe(logits, k, kept); b.observe(logits, k, kept)
        self.assertEqual(a.snapshot(), b.snapshot())

if __name__ == '__main__': unittest.main()
