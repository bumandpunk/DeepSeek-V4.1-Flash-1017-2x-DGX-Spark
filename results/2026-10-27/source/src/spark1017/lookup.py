"""Bounded suffix proposals with deterministic arbitration against DSpark.

Prompt lookup is established prior art (see docs/design.md). This implementation
combines a bounded incremental index, censored acceptance estimates, and the
target engine's verification budget. It never emits an unverified token.
"""
from collections import deque


class SuffixIndex:
    def __init__(self, capacity=8192, key_size=8, candidates=4):
        if capacity < 64 or key_size < 1 or candidates < 1:
            raise ValueError('invalid suffix-index bounds')
        self.capacity, self.key_size, self.candidates = capacity, key_size, candidates
        self.ring = [0] * capacity
        self.end = 0
        self.index = {}
        self.order = deque()

    def at(self, pos):
        if not max(0, self.end - self.capacity) <= pos < self.end:
            raise IndexError('expired or future token')
        return self.ring[pos % self.capacity]

    def append(self, tokens):
        for token in tokens:
            self.ring[self.end % self.capacity] = int(token)
            self.end += 1
            if self.end >= self.key_size:
                key = tuple(self.at(i) for i in range(self.end - self.key_size, self.end))
                queue = self.index.setdefault(key, deque(maxlen=self.candidates))
                queue.append(self.end); self.order.append((self.end, key))
            floor = max(0, self.end - self.capacity) + self.key_size
            while self.order and self.order[0][0] < floor:
                pos, key = self.order.popleft()
                queue = self.index.get(key)
                if queue and queue[0] == pos:
                    queue.popleft()
                    if not queue:
                        del self.index[key]

    def propose(self, max_tokens=15, min_match=16, max_match=32):
        if self.end < min_match * 2 or max_tokens < 1:
            return [], 0
        key = tuple(self.at(i) for i in range(self.end - self.key_size, self.end))
        best = None
        floor = max(0, self.end - self.capacity)
        for pos in self.index.get(key, ()):
            if pos == self.end:
                continue
            n = self.key_size
            while n < max_match and pos - n - 1 >= floor and self.end - n - 1 >= floor:
                if self.at(pos - n - 1) != self.at(self.end - n - 1):
                    break
                n += 1
            # The continuation comes entirely from earlier committed history,
            # excluding the query suffix. No recursive extrapolation is used.
            count = min(max_tokens, self.end - n - pos)
            if n >= min_match and count > 0 and (best is None or (n, pos) > best[:2]):
                best = n, pos, count
        if best is None:
            return [], 0
        n, pos, count = best
        return [self.at(i) for i in range(pos, pos + count)], n


class Arbiter:
    def __init__(self):
        self.cells = [[[0, 0] for _ in range(15)] for _ in range(3)]
        self.neural_gain, self.neural_cost = 4.5, 50.0
        self.rounds = self.accepted = self.proposed = 0

    def choose(self, proposal, matched, costs):
        bucket = min(2, (matched - 16) // 8)
        survival, gain, best, value = 1.0, 1.0, 0, 0.0
        for j in range(min(len(proposal), len(costs) - 1, 15)):
            seen, kept = self.cells[bucket][j]
            probability = (4 * .95 + kept) / (4 + seen)
            survival *= probability; gain += survival
            rate = gain / costs[j + 1]
            if rate > value:
                best, value = j + 1, rate
        return best if value > 1.03 * self.neural_gain / self.neural_cost else 0

    def observe(self, matched, verified, accepted):
        if matched < 16 or not 0 <= accepted <= verified <= 15:
            raise ValueError('invalid lookup verification outcome')
        bucket = min(2, (matched - 16) // 8)
        for j in range(min(verified, accepted + 1, 15)):
            cell = self.cells[bucket][j]
            cell[0] += 1; cell[1] += int(j < accepted)
            if cell[0] >= 256:
                cell[0] //= 2; cell[1] //= 2
        self.rounds += 1; self.accepted += accepted; self.proposed += verified

    def observe_neural(self, verified, accepted, costs, draft_cost):
        cost = costs[min(verified, len(costs) - 1)] + draft_cost
        self.neural_gain = .95 * self.neural_gain + .05 * (accepted + 1)
        self.neural_cost = .95 * self.neural_cost + .05 * cost


def propose(decoder, live, costs):
    decoder._spark1017_lookup_meta = {}
    decoder._spark1017_lookup_live = len(live)
    # A single active stream can use the full 16-row target window. Multiple
    # streams keep the existing joint DSpark scheduler in this first version.
    if len(live) != 1 or not live[0].draft or live[0].done:
        return {}
    stream = live[0]
    remaining = stream.count - len(stream.out)
    if remaining < 3:
        return {}
    state = getattr(stream, '_spark1017_suffix', None)
    sc = stream.st.sc; pos = sc.length
    if state is None or state[1] > pos or pos - state[1] > state[0].capacity:
        bank = SuffixIndex()
        start = max(0, pos - bank.capacity + 1)
        bank.append(sc.host[start:pos].tolist())
    else:
        bank, start = state
        bank.append(sc.host[start:pos].tolist())
    bank.append([stream.pending])
    stream._spark1017_suffix = bank, pos + 1
    if not hasattr(decoder, '_spark1017_lookup_arbiter'):
        decoder._spark1017_lookup_arbiter = Arbiter()
    draft, matched = bank.propose(min(15, decoder.max_rows - 1, remaining))
    if not draft:
        return {}
    k = decoder._spark1017_lookup_arbiter.choose(draft, matched, costs)
    if not k:
        return {}
    decoder._spark1017_lookup_meta[stream.sid] = matched
    return {stream.sid: draft[:k]}


def observe(decoder, stream, verified, accepted, costs, draft_cost):
    arbiter = getattr(decoder, '_spark1017_lookup_arbiter', None)
    if arbiter is None:
        return
    matched = decoder._spark1017_lookup_meta.get(stream.sid)
    if matched is None:
        if decoder._spark1017_lookup_live == 1:
            arbiter.observe_neural(verified, accepted, costs, draft_cost)
    else:
        arbiter.observe(matched, verified, accepted)
        if decoder.e.rank == 0 and arbiter.rounds % 20 == 0:
            print('[spark1017] lookup rounds=%d accepted=%d proposed=%d' %
                  (arbiter.rounds, arbiter.accepted, arbiter.proposed), flush=True)
