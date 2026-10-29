import random
import types
import unittest

from spark1017.memory_budget import allocator_limit, refresh_limit

GIB = 2**30


class MemoryBudget(unittest.TestCase):
    def test_stale_boot_cap_recovers_without_consuming_host_floor(self):
        # Recreate a 102.04 GiB stale cap with a later host observation that
        # permits growth. CUDA's reported free bytes are not used as host RAM.
        reserved = int(101.84 * GIB)
        limit = allocator_limit(4 * GIB, reserved, 3 * GIB, int(121.69 * GIB))
        self.assertGreaterEqual(limit - reserved, 200 * 2**20)
        self.assertLessEqual(limit - reserved, GIB)

    def test_host_pressure_reduces_cap_and_private_slack_gets_no_credit(self):
        reserved = 101 * GIB
        self.assertEqual(allocator_limit(2 * GIB, reserved, 3 * GIB, 121 * GIB), reserved)
        self.assertLess(allocator_limit(4 * GIB, reserved, 3 * GIB, 121 * GIB),
                        allocator_limit(5 * GIB, reserved, 3 * GIB, 121 * GIB))

    def test_physical_bounds_for_varied_observations(self):
        rng = random.Random(1017)
        for _ in range(10000):
            total = rng.randrange(1, 130 * GIB)
            reserved = rng.randrange(total + 1)
            available, floor = rng.randrange(130 * GIB), rng.randrange(10 * GIB)
            limit = allocator_limit(available, reserved, floor, total)
            self.assertGreaterEqual(limit, reserved)
            self.assertLessEqual(limit, total)
            self.assertLessEqual(limit - reserved, max(0, available - floor))

    def test_runtime_updates_both_directions_and_coalesces_small_changes(self):
        calls = []
        cuda = types.SimpleNamespace(mem_get_info=lambda: (10 * GIB, 120 * GIB),
                                     memory_reserved=lambda: 100 * GIB,
                                     set_per_process_memory_fraction=calls.append)
        engine = types.SimpleNamespace(rank=0, memory_available_floor_bytes=3 * GIB,
                                       memory_ceiling_bytes=101 * GIB)
        self.assertEqual(refresh_limit(engine, cuda, 5 * GIB), 102 * GIB)
        refresh_limit(engine, cuda, 5 * GIB + 1024)
        self.assertEqual(len(calls), 1)
        self.assertEqual(refresh_limit(engine, cuda, 4 * GIB), 101 * GIB)
        self.assertEqual(len(calls), 2)

    def test_invalid_observations_fail_closed(self):
        for args in ((-1, 1, 1, 2), (1, 3, 1, 2), (1, 0, 1, 0)):
            with self.assertRaises(ValueError):
                allocator_limit(*args)
