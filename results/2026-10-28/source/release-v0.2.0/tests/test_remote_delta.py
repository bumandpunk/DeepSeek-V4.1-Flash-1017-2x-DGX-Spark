import random
import unittest
from spark1017.remote_draft import delta_start


class RingDeltaTest(unittest.TestCase):
    def test_exact_ring_after_reuse_rewind_gap_and_wrap(self):
        rng=random.Random(101728)
        size=144; local=[None]*size; remote=[None]*size
        previous=None; owner='a'; end=0
        for i in range(10000):
            old_end=end
            if i%101==0:
                owner=str(i);end=rng.randrange(1,524000)
                local=[(owner,rng.getrandbits(32)) for _ in range(size)]
            elif i%67==0:
                end=max(0,end-rng.randrange(1,300))
                local=[(owner,rng.getrandbits(32)) for _ in range(size)]
            else:
                end+=rng.randrange(1,200 if i%17==0 else 17)
                for j in range(old_end,end):local[j%size]=(owner,rng.getrandbits(32))
            start=delta_start(owner,end,previous,size)
            for j in range(start,end):remote[j%size]=local[j%size]
            self.assertEqual(local,remote)
            previous=(owner,end)


if __name__=='__main__':unittest.main()
