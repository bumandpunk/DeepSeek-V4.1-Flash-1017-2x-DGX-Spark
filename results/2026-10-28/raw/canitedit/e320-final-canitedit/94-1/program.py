from typing import List
from math import sqrt


class Label:
    def __init__(self, name: str) -> None:
        self.name = name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Label) and self.name == __value.name


class Point:
    def __init__(self, x: int, y: int, label: Label | None) -> None:
        self.x = x
        self.y = y
        self.label = label

    def distance(self, other: "Point") -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def knn(self, others: List["Point"], k: int) -> Label:
        assert k > 0
        assert others
        assert not self.label
        assert len(others) >= k
        distances = map(lambda point: (
            point.label, self.distance(point)), others)
        votes = {}
        for label, _ in sorted(distances, key=lambda tup: tup[1])[:k]:
            if label not in votes.keys():
                votes[label] = 1
            else:
                votes[label] += 1
        return max(votes.items(), key=lambda item: item[1])[0]

### START TESTS ###
if True: # pragma: no cover
    origin = Point(0, 0, None)
    one_one = Point(1, 1, Label("one"))
    two_two = Point(2, 2, Label("two"))
    two_two_neg = Point(-2, -2, Label("one"))
    three_three = Point(3, 3, Label("two"))
    three_three_2 = Point(3, 3, Label("two"))

    assert origin == origin
    assert origin != "bla"
    assert Label("one") == Label("one")
    assert Label("one") != Label("two")
    assert Label("one") != "bla"

    try:
        origin.knn([one_one], -1)
        assert False
    except AssertionError:
        assert True


    try:
        origin.knn([], 1)
        assert False
    except AssertionError:
        assert True

    try:
        one_one.knn([two_two], 1)
        assert False
    except AssertionError:
        assert True

    try:
        origin.knn([two_two], 3)
        assert False
    except AssertionError:
        assert True

    assert (
        origin.knn([one_one, two_two, two_two_neg, three_three, three_three_2], 1).name
        == "one"
    )
    assert (
        origin.knn([one_one, two_two, two_two_neg, three_three, three_three_2], 3).name
        == "one"
    )
    assert (
        origin.knn([one_one, two_two, two_two_neg, three_three, three_three_2], 5).name
        == "two"
    )
print('SPARK1017_TESTS_PASSED_SENTINEL')
