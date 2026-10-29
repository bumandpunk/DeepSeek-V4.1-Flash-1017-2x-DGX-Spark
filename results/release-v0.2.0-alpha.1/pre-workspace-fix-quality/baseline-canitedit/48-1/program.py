from typing import List, Tuple

def max_sublstay_sum(lst: List[int]) -> Tuple[int, int, int]:
    max_so_far = lst[0]
    curr_max = lst[0]
    start_idx = 0
    end_idx = 0
    temp_start = 0
    for i in range(1, len(lst)):
        if lst[i] > curr_max + lst[i]:
            curr_max = lst[i]
            temp_start = i
        else:
            curr_max = curr_max + lst[i]
        if curr_max > max_so_far:
            max_so_far = curr_max
            start_idx = temp_start
            end_idx = i
    return (max_so_far, start_idx, end_idx)

### START TESTS ###
if True:  # pragma: no cover
    assert max_sublstay_sum([-2, -3, 4, -1, -2, 1, 5, -3]) == (7, 2, 6)
    assert max_sublstay_sum([-2, -3, -4, -1, -2, -1, -5, -3]) == (-1, 3, 3)
    assert max_sublstay_sum([1, 2, 3, 4, 5]) == (15, 0, 4)
    assert max_sublstay_sum([4]) == (4, 0, 0)
    assert max_sublstay_sum([1, -2, 3, 10, -4, 7, 2, -5]) == (18, 2, 6)
print('SPARK1017_TESTS_PASSED_SENTINEL')
