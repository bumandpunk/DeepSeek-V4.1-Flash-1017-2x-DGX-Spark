from typing import Any, List


class Free:
    def __repr__(self):
        return "Free"


# singleton
FREE = Free()


class MemoryAllocation:
    def __init__(self, size, address, buf):
        self.size = size
        self.address = address
        self.buffer = buf

    def __repr__(self):
        return f"MemoryAllocation(size={self.size}, address={self.address})"

    def write(self, data: List[Any]):
        for ex in data:
            if self.address >= len(self.buffer):
                break
            self.buffer[self.address] = ex
            self.address += 1


class MemoryAllocator:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer: List[Any] = [FREE] * max_size
        self.current = 0

    def allocate(self, size):
        if self.current + size > self.max_size:
            return None
        else:
            self.current += size
            return MemoryAllocation(size, self.current - size, self.buffer)

### START TESTS ###
if True: # pragma: no cover
    assert FREE.__repr__() == "Free"
    m1 = MemoryAllocator(100)
    a1 = m1.allocate(10)
    assert a1.__repr__() == "MemoryAllocation(size=10, address=0)"
    assert a1 is not None
    a1.write([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert a1.buffer == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [FREE] * 90
    a2 = m1.allocate(20)
    assert a2 is not None
    a2.write([11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30])
    assert a2.buffer == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30] + [FREE] * 70
    assert m1.buffer == a2.buffer
    a3 = m1.allocate(5)
    assert a3 is not None
    a3.write([31, 32, 33, 34, 35, 36, 37, 38, 39, 40])
    assert a3.buffer == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] + [FREE] * 65
    a4 = m1.allocate(65)
    assert a4 is not None
    a4.write([123] * 65)
    assert a4.buffer == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] + [123] * 65
    a5 = m1.allocate(1)
    assert a5 is None
print('SPARK1017_TESTS_PASSED_SENTINEL')
