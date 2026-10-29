"""DeepSeek-V4.1's concurrent decoding (``--parallel N``): up to N requests on the lane at once, every live stream's
verify window in one forward a round, each reply exactly its solo run.

- **Pool.** One cache plane a layer holds every slot (``Model.new_pool``). A stream's prompt fills its slot through
  the single-stream path on the slot's views, so its prompt bits are the solo prompt's. (v1: the whole prompt fills at
  admission, between rounds.)
- **Rounds.** Every live stream's pending token and drafts go through one forward (``rounds.RoundRunner``): each row
  at its own position and slot. Every kernel is row-invariant, the cache kernels read each row's own stream and
  selection is a total order, so a row's logits are the solo run's at that position, and each stream samples its own
  rows by its own rule: its reply is its solo reply, whatever else runs beside it. A round holds 16 rows at most
  (the decode windows' kernels): each stream's depth comes from ``DEPTH_BY`` by how many streams decode.
- **Drafts.** Each stream's DSpark drafter runs on its slot's own drafter cache (a captured graph a slot). Drafts
  only propose: they change speed, never a reply.
- **Ranks.** Rank 0 schedules (``tensorfold.cuda.scheduler``) and sends each step to the followers over a TCP link
  (``Link``) before running it; every rank checks the step's digest with one small all-gather before its model
  collectives (``OutOfStep``: the step fails on every rank at the same point). ``Watchdog``: a lost rank breaks the
  lane on every rank instead of hanging it, and a broken lane ends its process so a supervisor restarts the ranks
  (``TF_DS_EXIT_ON_BREAK``).
- **Faults.** A stream's own host work (its grammar, its samples, its reply's callback) runs no collective: when it
  raises, only that stream ends with the error and the lane goes on (``_stream_failed``). A device fault breaks the
  lane (``_device_fault``: out of memory, a CUDA or NCCL error): a sample syncs with the device, so a CUDA error of
  the forward can first show there. If one rank ends a stream alone, the next step fails on every rank: a follower
  that has no live stream, or another draft depth, still takes part in the digest check (``_differs``). Any other
  failed step breaks the lane. One rank has no ``Watchdog`` and no collective, so a host error of a step keeps
  the lane up there; but a device fault poisons the CUDA context, so it ends that process too (``_exit_soon``).
- **Kept prompts** (``TF_DS_KEEP``). A finished stream's prompt stays in the window (``Kept``): its extent's rows up
  to its last kept chunk boundary, and its slot's rings at its kept boundaries (the window keys and compressor inputs
  are a slot's, not an extent's). A later prompt that starts the same continues from the longest boundary they share:
  its rows are copied into its own extent (or it grows the kept extent in place), the rings come back, and its chunks
  from there are a fresh prefill's own chunks, so its reply is the fresh reply; usage's ``cached_tokens`` is that
  boundary. Only prompt rows are kept (a verify round's rows are not a prompt chunk's bits); with replay prefill the
  boundary stays a window below the prompt's end (the decoder layers' rows there are the prompt's own). Kept entries
  give their room back, oldest first, whenever an admission needs it.

Ported from our GLM-5.3 fork's ``glm_moe_dsa/cuda/multi.py`` (the same design, pipeline/CONCURRENCY-DESIGN.md).
Structured output: a stream's grammar goes with its admission to every rank, and each round cuts its drafts, masks its
verify rows and follows its kept tokens on every rank (``constrain``). Logprobs are not served with ``--parallel``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time

import torch

from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.streams import Stream
from tensorfold.engine.grammar import GrammarError
from tensorfold.engine.grammar import pack as pack_grammar

from ..ops import HostIds
from . import constrain
from .model import RAW

# the most drafts a stream verifies a round by how many streams decode ("5,5,3,3": up to the drafter's block of five
# alone or beside one other, three at three or four streams: 16 rows at four), capped by the engine's --mtp-drafts
DEPTH_BY = [int(x) for x in (os.environ.get("TF_DS_PARALLEL_DEPTH") or "5,5,3,3").split(",") if x.strip()]
# while prompts fill, decoding keeps this share of the time (a prompt chunk runs once the rounds since the last one
# took share / (1 - share) of its time); a prompt whose rest fits in QUICK_ROWS tokens fills before the next round
DECODE_SHARE = float(os.environ.get("TF_DS_DECODE_SHARE") or 0.5)
QUICK_ROWS = int(os.environ.get("TF_DS_QUICK_ROWS") or 1024)
# TF_DS_ROUND_STATS=1: each round's stages synchronized and timed (drafts, Engram rows + forward, sampling, absorb),
# their averages printed every 100 rounds by streams decoding (profiling only: the syncs cost a little)
ROUND_STATS = os.environ.get("TF_DS_ROUND_STATS", "0") == "1"
# TF_DS_HOST_TRACE=1: host timestamps through each round (no syncs added): every 100 rounds, per rank, the median ms of
# each step (the follower's wait for rank 0's op included), to find host gaps between and inside rounds
HOST_TRACE = os.environ.get("TF_DS_HOST_TRACE", "0") == "1"
# TF_DS_LINK_SPIN_MS (default 3): a follower polls its step link this long before a blocking receive, so the next round's
# step is picked up without a thread wake-up (~0.1-0.6 ms on GB10 after a short sleep); 0: block at once as before
LINK_SPIN = float(os.environ.get("TF_DS_LINK_SPIN_MS") or 3.0) / 1000.0
_HT: dict = {}


def _ht(tag: str) -> None:
    if HOST_TRACE:
        _HT.setdefault("_marks", []).append((tag, time.perf_counter()))


# TF_DS_DEPTH_POLICY=conf (default; "fixed": the table depth): each stream verifies the k (<= its table depth, up to the drafter's block) with the most
# expected tokens a millisecond: expected tokens from the confidence head's prefix survivals, the round's cost from a
# fixed table of forward ms by rows (TF_DS_ROUND_MS, a pure function of state every rank holds: no live timers)
# "even": while the drafting streams started together (their rounds within TF_DS_EVEN_COHORT), the stream with the
# most rounds left (its remaining tokens over its tokens a round lately) takes the k with the most expected tokens a
# millisecond, up to the drafter's block in the rows left, and every other stream the fewest drafts (<= its table
# depth) whose expected tokens still end it before that one (by TF_DS_EVEN_MARGIN): a burst's streams end together
# and its rounds stay short. Streams that did not start together get "conf".
DEPTH_POLICY = os.environ.get("TF_DS_DEPTH_POLICY", "conf")
EVEN_MARGIN = float(os.environ.get("TF_DS_EVEN_MARGIN") or 1.1)
EVEN_COHORT = int(os.environ.get("TF_DS_EVEN_COHORT") or 4)
ROUND_MS = [float(x) for x in (os.environ.get("TF_DS_ROUND_MS") or
                               "30.0,35.2,39.9,44.9,48.6,52.1,55.6,59.1,62.7,66.3,69.6,72.9,75.5,78.2,81.3,84.4,"
                               "87.5,90.6,93.7,96.8,99.9,103.0,106.1,109.2").split(",")]
# the drafter pass in the depth choice (ms): 3.5 = its measured cost on the pair since the vocabulary-split Markov loop
# (5.4 before; set-b code at 1 stream over test-server starts: 5.4 -> 99.6, 3.5 -> 100.1-100.7, 7.5 -> 98.9)
DRAFT_MS = float(os.environ.get("TF_DS_DRAFT_MS") or 3.5)
# TF_DS_EAGER_ABSORB=1 (default): every drafting stream's window rows go into its drafter rings right after the forward
# is launched, on a side stream behind the forward (its host launches overlap the forward, not the gap between rounds),
# instead of the kept rows after sampling. Rows past the kept ones sit at positions the drafter does not read (it reads
# up to the stream's length) until a later absorb writes them, so no draft changes; the main stream waits for the side
# stream at the round's end. 0: the kept rows after sampling, as before
EAGER_ABSORB = os.environ.get("TF_DS_EAGER_ABSORB", "1") == "1"
# TF_DS_CONF_LOG=1: per draft depth, drafts verified and kept, and kept by confidence (sigmoid tenths), printed every
# 200 rounds (the DSpark acceptance report)
CONF_LOG = os.environ.get("TF_DS_CONF_LOG", "0") == "1"
STEP_TIMEOUT = float(os.environ.get("TF_DS_STEP_TIMEOUT") or 900.0)
# TF_DS_EXIT_ON_BREAK=1 (default): a broken lane ends its process with EXIT_CODE TF_DS_EXIT_GRACE seconds after the
# break (time for the live requests to get their error), so a supervisor (a container restart policy, systemd)
# restarts the ranks. A broken lane cannot serve again: NCCL is aborted. A device fault ends a one-rank process
# the same way: its CUDA context is poisoned. 0: the process stays up and every step fails, as before
EXIT_ON_BREAK = os.environ.get("TF_DS_EXIT_ON_BREAK", "1") == "1"
EXIT_GRACE = float(os.environ.get("TF_DS_EXIT_GRACE") or 5.0)
EXIT_CODE = 75                   # EX_TEMPFAIL: a restart can serve again
# TF_DS_KEEP=1 (default): finished prompts stay in the window for later prompts that start the same (``Kept``), at
# most KEEP_ENTRIES of them, each with its rings at KEEP_MARKS chunk boundaries at most (the doubling ones from the first
# chunk, the prompt's last two, the ones it continued from)
KEEP = os.environ.get("TF_DS_KEEP", "1") == "1"
KEEP_ENTRIES = int(os.environ.get("TF_DS_KEEP_ENTRIES") or 8)
KEEP_MARKS = int(os.environ.get("TF_DS_KEEP_MARKS") or 10)
# TF_DS_KEEP_SHRINK=1 (default): a request that needs room cuts the oldest kept prompt back to its largest kept boundary
# that makes the room (_shrink) before it forgets one whole; prompts also keep boundaries at 3/4 and 7/8 of their
# length (and every kept boundary in their upper half), so a long kept prompt shrinks a little, not to half. A 1M kept
# prompt left 10,240 free positions and any chat beside it (the lane's 32,768 default reply) forgot all of it.
# 0: forget whole kept prompts, doubling boundaries only, as before
KEEP_SHRINK = os.environ.get("TF_DS_KEEP_SHRINK", "1") == "1"
# prompts of at least this many tokens get the 3/4 and 7/8 boundaries (one ring snapshot each, ~8 MB): the ones whose
# fill is expensive to repeat; shorter prompts keep the boundaries they had
KEEP_SHRINK_MIN = int(os.environ.get("TF_DS_KEEP_SHRINK_MIN") or 131072)
# TF_DS_SPILL_GIB=N (default 0: off; with TF_DS_KEEP): every kept prompt of TF_DS_SPILL_MIN tokens or more also goes to
# this rank's disk (spill.py: TF_DS_SPILL_DIR, N GiB at most, unused ones gone after TF_DS_SPILL_MAX_AGE_H), so a
# prompt that continues one the window no longer holds (a restart, an eviction) reads its rows back instead of filling
# them: rank 0 offers the spilled boundary when it is past every kept prompt's, and every rank must hold it


ALIGN = 2048                     # extents start and end on multiples of this many positions


class Extents:
    """First-fit extents of a pool of ``total`` positions in ALIGN steps; a freed extent merges with its neighbours."""

    def __init__(self, total: int, align: int = ALIGN) -> None:
        self.align = int(align)
        self.total = int(total) // self.align * self.align
        self.gaps: list[tuple[int, int]] = [(0, self.total)] if self.total else []

    def size(self, rows: int) -> int:
        return -(-int(rows) // self.align) * self.align

    def take(self, rows: int) -> int | None:
        """The first gap that holds ``rows`` (aligned up): its start, else None."""

        n = self.size(rows)
        for i, (a, b) in enumerate(self.gaps):
            if b - a >= n:
                self.gaps[i:i + 1] = [(a + n, b)] if b - a > n else []
                return a
        return None

    def take_at(self, start: int, rows: int) -> bool:
        """``rows`` (aligned up) from ``start`` if they are free."""

        n = self.size(rows)
        for i, (a, b) in enumerate(self.gaps):
            if a <= start and start + n <= b:
                self.gaps[i:i + 1] = [g for g in ((a, start), (start + n, b)) if g[1] > g[0]]
                return True
        return False

    def give(self, start: int, rows: int) -> None:
        merged: list[tuple[int, int]] = []
        for a, b in sorted(self.gaps + [(start, start + self.size(rows))]):
            if merged and merged[-1][1] == a:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        self.gaps = merged


class OutOfStep(RuntimeError):
    """The ranks planned a different step: it fails on every rank before its collectives."""


# what a RuntimeError says when the device or NCCL failed (and not the stream's own host work): see ``_device_fault``
DEVICE_MARKS = ("CUDA error", "CUBLAS_STATUS", "cuDNN", "NCCL", "illegal memory access", "device-side assert")


def _device_fault(exc: BaseException) -> bool:
    """Whether ``exc`` is a fault of the device (out of memory, a CUDA or NCCL error) and not of one stream's host
    work. A CUDA error is asynchronous: an error of the forward can first show at a later sync, such as a sample's
    ``.cpu()``, with a message that names no CUDA call. So a synchronize that fails counts as a device fault too."""

    if isinstance(exc, (torch.cuda.OutOfMemoryError, getattr(torch, "AcceleratorError", ()))):
        return True
    if isinstance(exc, RuntimeError) and any(mark in str(exc) for mark in DEVICE_MARKS):
        return True
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:                                # noqa: BLE001  (a sync that raises: the device is in error)
            return True
    return False


_exit_lock = threading.Lock()
_exit_armed = False


def _exit_soon(rank: int, why: str) -> None:
    """With EXIT_ON_BREAK, end this process with EXIT_CODE after EXIT_GRACE seconds (time for the live requests to
    get their error). The watchdog (a broken lane) and the decoder (a device fault on one rank) both call it. The
    first call starts the timer; a later call does nothing, so one process has one exit."""

    global _exit_armed
    if not EXIT_ON_BREAK:
        return
    with _exit_lock:
        if _exit_armed:
            return
        _exit_armed = True
    timer = threading.Timer(EXIT_GRACE, _exit_now, args=(rank, why))
    timer.daemon = True
    timer.start()


def _exit_now(rank: int, why: str) -> None:
    print(f"[tensorfold] rank {rank}: the process exits ({why}) so that its supervisor restarts the ranks "
          "(TF_DS_EXIT_ON_BREAK=0 keeps it up)", flush=True)
    os._exit(EXIT_CODE)


class Link:
    """Rank 0's steps to every follower, in order, over one TCP connection each (port through the rendezvous store);
    a follower waits on its socket between steps, never inside a collective."""

    KEY = "tensorfold/dsv41/multi/port"

    def __init__(self, store, *, rank: int, world: int, host: str) -> None:
        import socket
        from datetime import timedelta

        self.rank, self.world, self.socks = rank, world, []
        if rank == 0:
            self.server = socket.create_server(("0.0.0.0", 0))
            store.set(self.KEY, str(self.server.getsockname()[1]))
        else:
            self.server = None
            store.wait([self.KEY], timedelta(hours=24))
            sock = socket.create_connection((host or "127.0.0.1", int(store.get(self.KEY).decode())))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.sendall(rank.to_bytes(4, "big"))
            self.socks = [sock]

    def _accept(self) -> None:
        import socket

        got = {}
        while len(got) < self.world - 1:
            sock, _ = self.server.accept()
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            got[int.from_bytes(self._read(sock, 4), "big")] = sock
        self.socks = [got[r] for r in sorted(got)]

    @staticmethod
    def _read(sock, n: int) -> bytes | None:
        out = b""
        while len(out) < n:
            chunk = sock.recv(n - len(out))
            if not chunk:
                return None
            out += chunk
        return out

    def send(self, op: list) -> None:
        if not self.socks:
            self._accept()
        data = json.dumps(op).encode()
        for sock in self.socks:
            sock.sendall(len(data).to_bytes(4, "big") + data)

    def receive(self) -> list | None:
        if LINK_SPIN > 0:                            # a follower: poll for rank 0's next step a while before blocking
            import select

            sock, until = self.socks[0], time.perf_counter() + LINK_SPIN
            while not select.select([sock], [], [], 0)[0] and time.perf_counter() < until:
                pass
        head = self._read(self.socks[0], 4)
        body = None if head is None else self._read(self.socks[0], int.from_bytes(head, "big"))
        return None if body is None else json.loads(body)


class Watchdog:
    """Liveness between rank 0 and each follower (heartbeats on their own TCP connections), NCCL's asynchronous error
    and a step that never ends: any of them breaks the lane, the RDMA gathers stop waiting, NCCL aborts, and the step
    in flight fails on every rank instead of hanging."""

    KEY = "tensorfold/dsv41/multi/watch"

    def __init__(self, comm, store, *, rank: int, world: int, host: str, timeout: float = 30.0,
                 every: float = 1.0) -> None:
        import socket
        from datetime import timedelta

        self.comm, self.rank, self.world, self.timeout, self.every = comm, rank, world, timeout, every
        self.broken: str | None = None
        self.busy: float | None = None
        self.lock = threading.Lock()
        self.socks: list = []
        if rank == 0:
            server = socket.create_server(("0.0.0.0", 0))
            store.set(self.KEY, str(server.getsockname()[1]))

            def accept() -> None:
                for _ in range(world - 1):
                    sock, _ = server.accept()
                    peer = int.from_bytes(Link._read(sock, 4) or b"\xff\xff\xff\xff", "big")
                    with self.lock:
                        self.socks.append(sock)
                    threading.Thread(target=self._listen, args=(sock, peer), daemon=True).start()

            threading.Thread(target=accept, daemon=True).start()
        else:
            store.wait([self.KEY], timedelta(hours=24))
            sock = socket.create_connection((host or "127.0.0.1", int(store.get(self.KEY).decode())))
            sock.sendall(rank.to_bytes(4, "big"))
            self.socks = [sock]
            threading.Thread(target=self._listen, args=(sock, 0), daemon=True).start()
        threading.Thread(target=self._beat, daemon=True).start()
        threading.Thread(target=self._nccl, daemon=True).start()

    def _beat(self) -> None:
        while self.broken is None:
            with self.lock:
                socks = list(self.socks)
            for sock in socks:
                try:
                    sock.sendall(b"H")
                except OSError:
                    pass
            time.sleep(self.every)

    def _listen(self, sock, peer: int) -> None:
        sock.settimeout(self.timeout)
        while self.broken is None:
            try:
                got = sock.recv(64)
            except OSError:
                got = b""
            if not got:
                self.abort(f"rank {peer} stopped answering" if self.rank == 0 else "rank 0 stopped answering")
                return
            if b"A" in got:
                self.abort("rank 0 lost a rank")
                return

    def _nccl(self) -> None:
        nccl = getattr(self.comm, "nccl", self.comm)
        check = getattr(nccl, "async_error", None)
        while self.broken is None:
            time.sleep(self.every)
            code = check() if check is not None else 0
            if code > 0:
                self.abort(f"NCCL error {code}")
                return
            busy = self.busy
            if busy is not None and time.monotonic() - busy > STEP_TIMEOUT:
                self.abort(f"a step ran past {STEP_TIMEOUT:.0f} s on rank {self.rank}")
                return

    def abort(self, why: str) -> None:
        with self.lock:
            if self.broken is not None:
                return
            self.broken = why
            socks = list(self.socks)
        print(f"[tensorfold] rank {self.rank}: the lane is broken ({why}); every step fails from now on", flush=True)
        rdma = getattr(self.comm, "rdma", None)
        if rdma is not None:
            rdma.abort(why)
        nccl = getattr(self.comm, "nccl", self.comm)
        if hasattr(nccl, "abort"):
            nccl.abort()
        if self.rank == 0:
            for sock in socks:
                try:
                    sock.sendall(b"A")
                except OSError:
                    pass
        _exit_soon(self.rank, why)


def _pack(sampling) -> list | None:
    if sampling is None:
        return None
    return [int(sampling.seed), float(sampling.temperature), int(sampling.top_k), float(sampling.top_p),
            float(sampling.min_p)]


def _unpack(values):
    from tensorfold.engine.exact_sampling import Sampling

    return None if values is None else Sampling(values[0], values[1], values[2], values[3], values[4])


class Slot:
    """A stream's slot: its index (its window ring, compressor inputs and drafter rings) and, while a stream holds it,
    that stream's SeqCache over its extent of the pool."""

    def __init__(self, index: int, dc) -> None:
        self.index, self.sc, self.dc = index, None, dc


class Kept:
    """A finished stream's prompt kept in the window: positions [base, base + size) hold the compressed and indexer
    rows (and token ids) of its first ``top`` prompt tokens; ``snaps`` its slot's rings and compressor inputs at each
    kept chunk boundary; ``host`` the prompt's host ids to ``top`` (Engram's n-grams); ``keys`` (rank 0 only) the
    prompt's ids to ``top`` with each image span's positions keyed by its picture; ``tick`` when it was last used."""

    def __init__(self, eid: int, base: int, size: int, top: int, host: list, snaps: dict, replay: bool, keys,
                 tick: int) -> None:
        self.eid, self.base, self.size, self.top, self.host = eid, base, size, top, host
        self.snaps, self.replay, self.keys, self.tick, self.hits = snaps, replay, keys, tick, 0
        self.disk: dict | None = None                    # its spilled manifest (TF_DS_SPILL_GIB), when written


def _picture_key(pic) -> int:
    """A picture's 62-bit key (its patches and grids), kept on the picture."""

    key = getattr(pic, "_tf_key", None)
    if key is None:
        h = hashlib.sha256(repr((pic.n_vit_h, pic.n_vit_w, pic.n_llm_h, pic.n_llm_w)).encode())
        h.update(pic.patches.contiguous().view(torch.int16).numpy().tobytes())
        key = int.from_bytes(h.digest()[:8], "big") >> 2
        pic._tf_key = key
    return key


def image_workspace(vision, tower) -> tuple[int, int]:
    """Additional live bytes before admission, separately for the two ranks.

    Images are encoded sequentially on rank 0. Bound the largest picture's
    fp32 attention/rotary and MLP live tensors, plus all retained span rows,
    concatenation and RDMA copies. Encoding and prefill run sequentially on
    the scheduler's CUDA stream: reserve their maximum, not their sum. Add
    128 MiB for cache restoration and 128 MiB for vision library workspace.
    Keep the existing 768 MiB prefill allowance. No resize or dtype changes.
    """
    base = 768 * 2**20
    spans = getattr(vision, "spans", ()) if vision is not None else ()
    if not spans:
        return base, base
    cfg = tower.cfg
    patches = max(pic.n_vit_h * pic.n_vit_w for _, pic in spans)
    tokens = sum(pic.tokens for _, pic in spans)
    # Generous live-tensor bound for Tower._block (including fp32 SDPA inputs).
    temporary = patches * (64 * cfg.dim + 16 * cfg.inter)
    if tower.attention == "math":
        temporary += 8 * cfg.heads * patches * patches
    rows = tokens * cfg.model_dim * 2
    persistent = 128 * 2**20 + 4 * rows
    head = persistent + max(base, 128 * 2**20 + temporary)
    worker = persistent + base
    return head, worker


class MultiDecoder:
    """Rounds over the live streams; ``slots`` streams at most, each with ``cap`` cache rows."""

    def __init__(self, engine, *, slots: int, cap: int) -> None:
        from .rounds import MAX_ROWS, RoundRunner

        self.e = engine
        m = engine.model
        self.m, self.cap = m, int(cap)
        self.pool = m.new_pool(slots, self.cap)
        d = engine.drafter
        self.dpool = None
        if d is not None:
            from .dspark import DraftPool

            self.dpool = DraftPool(d, slots)             # every slot's drafter rings in one plane a stage
        self.slots = [Slot(i, self.dpool.views[i] if d is not None else None) for i in range(slots)]
        self.extents = Extents(self.cap)              # every stream takes an extent of the one window
        self._table_view = m.pool_view(self.pool, 0, 0, self.extents.total)   # RoPE tables for any position
        self.drafters: dict[int, object] = {}            # batched drafter graphs by drafting streams
        self.free = list(range(slots))
        self.max_rows = MAX_ROWS
        if not engine.drafts + 1 <= MAX_ROWS < 64:     # a solo verify window must take the decode kernels, a round too
            raise ValueError(f"TF_DS_DECODE_ROWS={MAX_ROWS} must hold a verify window ({engine.drafts + 1} rows) "
                             "and stay under the 64-row prompt kernels")
        self.depth_most = engine.drafts
        graphs = engine.runner is not None
        self.runner = RoundRunner(m, self.pool, graphs=graphs,
                                  graph_pool=engine.runner.pool if graphs and engine.runner.pool is not None else None)
        self.eos = tuple(engine.eos)
        self.streams: dict[int, Stream] = {}           # decoding (and just ended) streams
        self.filling: list[Stream] = []                 # admitted, their prompts filling their slots (oldest first)
        self.chunk_s, self.since_fill = 0.0, 0.0        # the last prompt chunk's time; decoding since then
        self.next_id = 0
        self.link: Link | None = None
        self.watch: Watchdog | None = None
        self.rounds = 0
        self.round_log: list[tuple[int, int, float, int]] = []
        self.stage = {}                                 # ROUND_STATS: streams -> [rounds, rows, tokens, seconds a stage]
        self.conf_depth = [[0, 0] for _ in range(8)]     # CONF_LOG: (drafts reached, kept) by depth
        self.conf_bins = [[0, 0] for _ in range(10)]     # CONF_LOG: (drafts, kept) by sigmoid(confidence) tenths
        self.conf_rounds = 0
        self.k_hist: dict[int, list[int]] = {}           # CONF_LOG: drafts verified a stream a round, by streams live
        self.kept: dict[int, Kept] = {}                  # kept prompts by id
        self.next_kept, self.ticks = 0, 0
        self.keep_stats: dict[str, int] = {}             # admissions that continued a kept prompt, by placement
        self.spill = None                                # TF_DS_SPILL_GIB: kept prompts on this rank's disk
        self.spill_nonce = os.urandom(4).hex() if engine.rank == 0 else None   # names this run's spilled prompts
        if KEEP:
            from . import spill as sp

            if sp.SPILL_GIB > 0:
                try:
                    ratios = {i: m.cfg.compress_ratios[i] for i in set(self.pool.comp) | set(self.pool.index_k)}
                    ident = sp.engine_id(getattr(engine, "model_dir", "."), world=engine.world, pool=self.pool,
                                         dpool=self.dpool)
                    self.spill = sp.Spill(sp.SPILL_DIR, engine=ident, rank=engine.rank, pool=self.pool, ratios=ratios,
                                          budget=int(sp.SPILL_GIB * 2**30), max_age=sp.SPILL_MAX_AGE_H * 3600.0)
                    self.spill.recycle()
                    if engine.rank == 0:
                        st = self.spill.stats()
                        print(f"[tensorfold] spilled kept prompts: {self.spill.dir} ({st['manifests']} on disk, "
                              f"{st['bytes'] / 2**30:.2f} of {sp.SPILL_GIB:g} GiB)", flush=True)
                except Exception as exc:                 # noqa: BLE001  (no disk: kept prompts stay in the window only)
                    print(f"[tensorfold] rank {engine.rank}: spilled kept prompts off ({type(exc).__name__}: {exc})",
                          flush=True)
                    self.spill = None

    def warm(self, buckets=(1024, 2048, 4096, 8192)) -> None:
        """Before serving, on every rank in the same order: round graphs for 1 .. 16 rows at the small context
        buckets (synthetic rows on slot 0, which a request rewrites before it reads them) and each slot's drafter
        graph."""

        t0 = time.perf_counter()
        n = 0
        for b in buckets:
            if b > self.cap:
                break
            host = [1000 + (i * 7919) % 60000 for i in range(b)]
            for R in range(self.max_rows, 0, -1):
                if (R, self.runner.bucket(b)) in (self.runner.graphs or {}):
                    continue
                self.runner.forward([(0, 0, self.extents.total, b - R, host[b - R:b], host)], replay=False)
                n += 1
        if self.e.drafter is not None:
            for k in range(1, len(self.slots) + 1):
                self._drafts([0] * k, [1] * k, list(range(k)))
        torch.cuda.synchronize()
        if self.e.rank == 0:
            print(f"[tensorfold] --parallel warm-up: {n} round graphs, {len(self.drafters)} drafter graphs, "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)

    def _drafts(self, tokens: list[int], q0: list[int], slots: list[int]) -> list[list[int]]:
        """Every drafting stream's drafts from one batched drafter pass (a graph a stream count)."""

        if os.environ.get("SPARK1017_DRAFT_HELPER") and self.streams:
            from spark1017.remote_draft import propose as remote_propose
            remote = remote_propose(self, tokens, q0, slots)
            if remote is not None:
                return remote
        from .dspark import BatchDraftGraph

        N = len(tokens)
        g = self.drafters.get(N)
        if g is None:
            # as deep as any round with N drafting streams may verify ("even": the whole block)
            steps = None if DEPTH_POLICY == "even" else max(self._depth(L) for L in range(N, len(self.slots) + 1))
            g = BatchDraftGraph(self.e.drafter, self.m.pool_view(self.pool, 0, 0, self.extents.total), self.dpool, N,
                                steps=steps)
            g.tokens.copy_(torch.tensor(tokens, dtype=torch.long))
            g.q0.copy_(torch.tensor(q0, dtype=torch.long))
            g.slots.copy_(torch.tensor(slots, dtype=torch.long))
            if self.e.runner is not None:
                if self.e.runner.pool is None:
                    self.e.runner.pool = torch.cuda.graph_pool_handle()
                g.capture(self.e.runner.pool)
            self.drafters[N] = g
        _ht("drafter run")
        return g.run(tokens, q0, slots)

    # -- the scheduler's interface --------------------------------------------------------------------------------
    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def _send(self, op: list) -> None:
        self._alive()
        if self.link is not None:
            self.link.send(op)

    def _step(self, busy: bool) -> None:
        if self.watch is not None:
            self.watch.busy = time.monotonic() if busy else None

    def _broken(self, exc: BaseException) -> None:
        """A step that raised past its digest check (out of memory under the allocator ceiling, a bug) may have left
        another rank inside a collective: the lane breaks on every rank (and each process exits, TF_DS_EXIT_ON_BREAK)
        instead of the next step's collectives pairing with the wrong ones. OutOfStep and NoRoom happen on every rank
        at the same point. A stream's own host work does not get here (``_stream_failed``), unless it is a device
        fault (``_device_fault``).

        One rank has no watchdog and no collective to pair wrongly, so a host error of a step leaves the lane up.
        A device fault poisons the CUDA context: the process ends as on a broken lane (``_exit_soon``), once, and
        not at all with TF_DS_EXIT_ON_BREAK=0."""

        if isinstance(exc, (OutOfStep, NoRoom)):
            return
        why = f"a step failed on rank {self.e.rank} ({type(exc).__name__}: {str(exc)[:160]})"
        if self.watch is not None:
            self.watch.abort(why)
        elif _device_fault(exc):
            print(f"[tensorfold] rank {self.e.rank}: a device fault poisons this lane ({why})", flush=True)
            _exit_soon(self.e.rank, why)

    def _alive(self) -> None:
        if self.watch is not None and self.watch.broken is not None:
            raise RuntimeError(f"this lane lost a rank ({self.watch.broken}); restart it on every rank")

    def _agree(self, what: str, plan: list, guard_bytes=None):
        """One small all-gather checks the step on every rank before any of its model collectives."""

        e = self.e
        if guard_bytes is not None and os.environ.get("SPARK1017_CONTROL_FUSION") == "1":
            from spark1017.control import agree_and_budget
            return agree_and_budget(self, what, plan, guard_bytes, OutOfStep)
        if e.world < 2:
            return
        dgst = int.from_bytes(hashlib.sha256(json.dumps(plan).encode()).digest()[:8], "big")
        # 16-bit pieces as fp32 (exact): the model's gather, so a round's check rides the RDMA path (~0.1 ms, not ~0.9)
        mine = torch.tensor([(dgst >> b) & 0xFFFF for b in (0, 16, 32, 48)], dtype=torch.float32, device="cuda")
        every = self.m.comm.gather(mine).tolist()
        if any(row != every[0] for row in every):
            raise OutOfStep(f"the ranks planned different {what}s; its requests fail, serving goes on")

    def _all(self, ok: bool) -> bool:
        """Whether ``ok`` holds on every rank (one small gather: every rank calls it at the same point of a step)."""

        if self.e.world < 2:
            return bool(ok)
        mine = torch.tensor([1.0 if ok else 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        return all(row[0] == 1.0 for row in self.m.comm.gather(mine).tolist())

    def _memory_guard(self, needed: int, *, admission: bool = False) -> bool:
        """A collective boundary check; never catch and continue after a CUDA fault.

        Both ranks choose the same branch before model collectives. NoRoom at
        admission queues the caller; OutOfStep during execution safely cancels
        live work via the existing synchronized drop path, leaving the lane up.
        Only recomputable prefix caches are evicted, never request token history.
        Returns True when cached-prefix reuse plans need to be discarded.
        """
        if not hasattr(self.e, "memory_ceiling_bytes"):
            return False  # startup warm-up, before serving

        def fits():
            try:
                with open("/proc/meminfo") as f:
                    available = next(int(line.split()[1]) * 1024 for line in f if line.startswith("MemAvailable:"))
                allocated = torch.cuda.memory_allocated()
                # Graph-private reserved blocks cannot necessarily back a new
                # eager allocation. Do not count all reserved-minus-allocated as
                # reusable host memory. On shortage, release the ordinary cache
                # below and remeasure physical headroom before admitting work.
                self._memory_snapshot = dict(available=available, allocated=allocated,
                    reserved=torch.cuda.memory_reserved(), needed=needed,
                    ceiling=self.e.memory_ceiling_bytes,
                    host_floor=self.e.memory_available_floor_bytes)
                return (self.e.memory_ceiling_bytes - allocated >= needed and
                        available - self.e.memory_available_floor_bytes >= needed)
            except (OSError, StopIteration, ValueError):
                return False

        if self._all(fits()):
            return False
        # Same eviction order on all ranks, preserving active request extents.
        for key in list(self.kept):
            self._forget(self.kept[key])
        # Retain only the newest active prefix checkpoint; older checkpoints
        # improve reuse speed, but have no effect on this stream's KV/history.
        for stream in list(self.streams.values()) + self.filling:
            snaps = getattr(stream, "snaps", None)
            if snaps:
                newest = max(snaps)
                for boundary in list(snaps):
                    if boundary != newest:
                        del snaps[boundary]
        import gc
        from .weights import _trim
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _trim()
        if self._all(fits()):
            return True
        message = ("memory_pressure: insufficient safe workspace; request deferred or stopped "
                   "before CUDA allocation; model remains loaded")
        stamp = time.monotonic()
        if stamp - getattr(self, "_memory_log_at", 0) >= 5:
            self._memory_log_at = stamp
            detail = {k: round(v / 2**30, 3) for k, v in getattr(self, "_memory_snapshot", {}).items()}
            print(f"[tensorfold] rank {self.e.rank}: memory guard GiB {detail}; admission={admission}", flush=True)
        if admission:
            raise NoRoom(message)
        raise OutOfStep(message)

    def _shape(self) -> list:
        return [self.next_id, list(self.free), list(self.extents.gaps),
                [[s.sid, s.st.index, s.st.sc.length, len(s.out), bool(s.done)] for s in self.streams.values()],
                [[s.sid, s.st.index, s.filled] for s in self.filling],
                [[k.eid, k.base, k.size, k.top, k.tick, sorted(k.snaps)] for k in self.kept.values()]]

    # -- kept prompts ---------------------------------------------------------------------------------------------
    def _replay(self) -> bool:
        from . import engine as eng

        return bool(getattr(self.e, "replay_mode", eng.REPLAY))

    def _keys(self, s: Stream):
        """Rank 0: the prompt's ids, each image span's positions keyed by its picture instead (int64)."""

        import numpy as np

        keys = np.asarray(s.prompt, dtype=np.int64).copy()
        for start, pic in (s.vision.spans if s.vision is not None and getattr(s.vision, "spans", None) else []):
            keys[start:start + pic.tokens] = -2 - _picture_key(pic)
        return keys

    def _match(self, s: Stream, keys) -> list | None:
        """Rank 0: [kept id, boundary] for the kept prompt this one continues the furthest (the most recent of
        equals), else None. The boundary is one of its kept ones, within the prompts' shared start, and below the
        prompt's last row (with replay prefill, a window below)."""

        import numpy as np

        replay = self._replay()
        limit = len(s.prompt) - (self.m.cfg.window if replay else 1)
        best = None
        for k in self.kept.values():
            if k.replay != replay or k.keys is None:
                continue
            n = min(len(k.keys), len(keys))
            diff = np.flatnonzero(k.keys[:n] != keys[:n])
            common = int(diff[0]) if diff.size else n
            cut = max((b for b in k.snaps if b <= min(common, limit)), default=0)
            if cut and (best is None or (cut, k.tick) > (best[1], best[0].tick)):
                best = (k, cut)
        return None if best is None else [best[0].eid, best[1]]

    def _marks(self, n: int) -> set[int]:
        """The chunk boundaries a prompt of n tokens keeps its rings at: doubling from the first, and its last two."""

        from .engine import PREFILL_CHUNK as C

        marks = set(range(C, n + 1, C)[-2:])
        b = C
        while b <= n:
            marks.add(b)
            b *= 2
        if KEEP_SHRINK and n >= KEEP_SHRINK_MIN:     # cut points near a long prompt's end (_shrink)
            for num, den in ((3, 4), (7, 8)):
                b = n * num // den // C * C
                if b >= C:
                    marks.add(b)
        return marks

    def _snapshot(self, index: int):
        """Slot ``index``'s window rings (with replay prefill the encoder layers' only: a resumed prompt's decoder
        layers read no window key before its replay row) and compressor inputs, copied."""

        RS, n = self.pool.ring_size, len(self.pool.ring)
        layers = range(self.m.cfg.n_layers // 2) if self._replay() else range(n)
        ring = torch.stack([self.pool.ring[i][index * RS:(index + 1) * RS] for i in layers])
        raw = [torch.stack([x[index * RAW:(index + 1) * RAW] for x in pair])
               for _, pair in sorted(self.pool.comp_raw.items())]
        return ring, (torch.stack(raw) if raw else None)

    def _restore(self, index: int, snap) -> None:
        ring, raw = snap
        RS = self.pool.ring_size
        for i in range(ring.shape[0]):
            self.pool.ring[i][index * RS:(index + 1) * RS].copy_(ring[i])
        if raw is not None:
            for j, (_, pair) in enumerate(sorted(self.pool.comp_raw.items())):
                for h, x in enumerate(pair):
                    x[index * RAW:(index + 1) * RAW].copy_(raw[j, h])

    def _copy_rows(self, src: int, dst: int, n: int) -> None:
        """Positions [src, src + n) of the window's compressed, indexer and token rows to [dst, dst + n) (through a
        copy where the two overlap)."""

        if src == dst:
            return
        c = self.m.cfg
        overlap = src < dst + n and dst < src + n
        for planes in (self.pool.comp, self.pool.index_k):
            for i, x in planes.items():
                r = c.compress_ratios[i]
                a, b, k = src // r, dst // r, n // r
                for t in (x if isinstance(x, tuple) else (x,)):
                    t[b:b + k].copy_(t[a:a + k].clone() if overlap else t[a:a + k])
        t = self.pool.tokens
        t[dst:dst + n].copy_(t[src:src + n].clone() if overlap else t[src:src + n])

    def _forget(self, k: Kept, give: bool = True) -> None:
        self.kept.pop(k.eid, None)
        if give:
            self.extents.give(k.base, k.size)

    def _room(self, need: int) -> bool:
        """Rank 0: whether ``need`` positions fit once every kept prompt gave its room back."""

        ex = Extents(0, self.extents.align)
        ex.total, ex.gaps = self.extents.total, list(self.extents.gaps)
        for k in self.kept.values():
            ex.give(k.base, k.size)
        return any(b - a >= ex.size(need) for a, b in ex.gaps)

    def _shrink(self, k: Kept, need: int) -> bool:
        """Cut kept prompt ``k`` back to its largest kept boundary below its top whose freed tail makes room for
        ``need`` positions: its rows, snapshots, keys and host ids up to that boundary stay (a later prompt resumes
        there, as from any kept boundary), the rest of its extent goes back. False when no boundary makes the room.
        A function of state every rank holds (extents, boundaries), so every rank cuts the same."""

        for m in sorted((b for b in k.snaps if b < k.top), reverse=True):
            size = self.extents.size(m)
            if size >= k.size:
                continue
            trial = Extents(0, self.extents.align)
            trial.total, trial.gaps = self.extents.total, list(self.extents.gaps)
            trial.give(k.base + size, k.size - size)
            if any(b - a >= trial.size(need) for a, b in trial.gaps):
                self.extents.give(k.base + size, k.size - size)
                k.size, k.top = size, m
                k.snaps = {b: v for b, v in k.snaps.items() if b <= m}
                k.keys = None if k.keys is None else k.keys[:m].copy()
                k.host = k.host.copy(m)
                self.keep_stats["shrunk"] = self.keep_stats.get("shrunk", 0) + 1
                return True
        return False

    def _place(self, need: int, src: Kept | None) -> tuple[int | None, str]:
        """An extent for ``need`` positions (every rank the same): a free one (``src``'s rows get copied in, "copy"),
        else ``src``'s own extent grown in place ("here"), else after the oldest other kept prompts give theirs back;
        last, anywhere once ``src`` gave its own back too (its rows move, "move"). Without ``src``: "fresh"."""

        while True:
            base = self.extents.take(need)
            if base is not None:
                return base, "copy" if src is not None else "fresh"
            if src is not None:
                self.extents.give(src.base, src.size)
                if self.extents.take_at(src.base, need):
                    self._forget(src, give=False)
                    return src.base, "here"
                if not self.extents.take_at(src.base, src.size):
                    raise RuntimeError("a kept prompt's extent was not free to take back")
            others = sorted((k for k in self.kept.values() if k is not src), key=lambda k: k.tick)
            if others:
                if not (KEEP_SHRINK and self._shrink(others[0], need)):
                    self._forget(others[0])
                continue
            if src is None:
                return None, "fresh"
            self._forget(src)
            base = self.extents.take(need)
            return base, "move" if base is not None else "fresh"

    @staticmethod
    def _kept_marks(snaps: dict) -> list[int]:
        """The boundaries a finished stream keeps, KEEP_MARKS at most: the doubling ones, its last two, then the
        earliest."""

        from .engine import PREFILL_CHUNK as C

        marks = sorted(snaps)
        if len(marks) > KEEP_MARKS:
            must = {b for b in marks if (b // C) & (b // C - 1) == 0} | set(marks[-2:])
            if KEEP_SHRINK:
                must |= {b for b in marks if 2 * b > marks[-1]}       # the upper half's cut points (_shrink)
            rest = [b for b in marks if b not in must]
            marks = sorted(must | set(rest[:max(0, KEEP_MARKS - len(must))]))
        return marks

    def _keep(self, s: Stream) -> None:
        """A finished stream's prompt into ``kept``: its extent's rows to its last kept boundary stay, the rest goes
        back."""

        marks = self._kept_marks(s.snaps)
        top = marks[-1]
        size = self.extents.size(top)
        self.free.append(s.st.index)
        self.free.sort()
        if s.size > size:
            self.extents.give(s.base + size, s.size - size)
        keys = getattr(s, "keys", None)
        k = Kept(self.next_kept, s.base, size, top, s.st.sc.host.copy(top), {b: s.snaps[b] for b in marks},
                 self._replay(), None if keys is None else keys[:top], self.ticks)
        self.kept[k.eid] = k
        self.next_kept += 1
        if self.spill is not None:                       # write-through: its rows past what its lineage holds
            k.disk = self.spill.persist(f"{self.spill_nonce or 'run'}-{k.eid:06d}", base=k.base, top=k.top,
                                        replay=k.replay, snaps=k.snaps, host=k.host,
                                        lineage=getattr(s, "lineage", None), keys=k.keys)
        s.st.sc = None
        s.snaps = None

    def _covered(self, done: list[Stream]) -> list[int]:
        """Rank 0: the kept prompts the finishing streams' prompts will cover (the same ids to their top, every kept
        boundary of theirs kept again)."""

        import numpy as np

        drops: list[int] = []
        for s in done:
            snaps, keys = getattr(s, "snaps", None), getattr(s, "keys", None)
            if not snaps or keys is None or s.sid not in self.streams:
                continue
            marks = set(self._kept_marks(snaps))
            top = max(marks)
            for k in self.kept.values():
                if (k.eid not in drops and k.keys is not None and k.top <= top and k.replay == self._replay()
                        and set(k.snaps) <= marks and np.array_equal(k.keys, keys[:k.top])):
                    drops.append(k.eid)
        return drops

    def _ends(self, s: Stream) -> tuple[int, ...]:
        return self.eos if s.stop_eos else ()

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """A request into the lowest free slot: its prompt fills the slot now, then it decodes in the rounds."""

        if s.probabilities is not None:
            raise ValueError("logprobs are not served with --parallel on DeepSeek-V4.1")
        # A refused second stream waits for an active stream to release its slot;
        # do not repeat collective eviction/GC on every generated token.
        blocked = getattr(self, "_memory_wait_for", set())
        current = set(self.streams) | {f.sid for f in self.filling}
        if blocked and blocked <= current:
            raise NoRoom("memory_pressure: waiting for an active request to finish")
        self._memory_wait_for = set()
        room = self.extents.total - len(s.prompt) - self.max_rows - 8 - 4
        if room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.extents.total}-token "
                             "context")
        s.count = max(1, min(s.count, room))
        if not self.free:
            raise NoRoom("every stream slot is busy")
        if not self._room(self._need(s)):
            raise NoRoom("the window's free extents are too small for this request now")
        positions = s.vision.positions() if s.vision is not None and getattr(s.vision, "spans", None) else []
        index = self.free[0]
        s.keys = self._keys(s) if KEEP else None
        reuse = self._cache_reuse(s)
        workspace = image_workspace(s.vision, self.e.tower) if positions else (768 * 2**20,) * 2
        self._send(["admit", list(s.prompt), s.count, _pack(s.sampling), bool(s.draft), bool(s.stop_eos), index,
                    positions, reuse, pack_grammar(s.constraint), workspace])
        try:
            self._admit(s, index, positions, reuse, workspace)
        except NoRoom as exc:
            if "memory_pressure:" in str(exc):
                self._memory_wait_for = set(self.streams) | {f.sid for f in self.filling}
            raise

    def _cache_reuse(self, s: Stream) -> list | None:
        """Rank 0 selects RAM reuse and an exact SSD fallback before the collective guard.

        The guard may evict RAM snapshots on either node. Carry the disk plan in
        the admission message so both ranks can recover without weakening that
        guard or restarting a long prompt from zero.
        """
        ram = self._match(s, s.keys) if KEEP and self.kept else None
        far = None
        if self.spill is not None and s.keys is not None:
            far = self.spill.best(s.keys, len(s.prompt), window=self.m.cfg.window,
                                  replay=self._replay(), above=0)
        disk = ["disk", far[0], far[1]] if far is not None else None
        if disk and (not ram or disk[2] > ram[1]):
            plan = disk
        elif ram:
            plan = [ram[0], ram[1], disk]
        else:
            plan = None
        print(f"[tensorfold] prefix cache: prompt={len(s.prompt)} ram={ram[1] if ram else 0} "
              f"disk={disk[2] if disk else 0} selected={'disk' if plan and plan[0] == 'disk' else 'ram' if plan else 'fresh'}",
              flush=True)
        return plan

    def _need(self, s: Stream) -> int:
        """A stream's positions: its prompt, its reply, a round's rows and its extent's scratch row (a group long)."""

        return len(s.prompt) + s.count + self.max_rows + 8 + 4

    def _admit(self, s: Stream, index: int, positions: list[int], reuse: list | None = None,
               workspace: tuple[int, int] | None = None) -> None:
        """Its slot, its extent, its image rows (shared from rank 0) and its prompt's chunk steps; ``_fill`` runs
        them. ``reuse`` [kept id, boundary]: the prompt continues that kept prompt from the boundary; ["disk", name,
        boundary]: it continues a spilled one (TF_DS_SPILL_GIB), its rows read into a fresh extent, or it fills fresh
        when a rank cannot read them (every rank agrees first)."""

        e = self.e
        workspace = workspace or ((1536 if positions else 768) * 2**20,) * 2
        self._step(True)
        try:
            self._agree("admission", [self._shape(), list(s.prompt), s.count, _pack(s.sampling), bool(s.draft),
                                      bool(s.stop_eos), index, positions, reuse, pack_grammar(s.constraint), workspace])
            evicted = self._memory_guard(int(workspace[min(e.rank, 1)]), admission=True)
            if evicted and reuse and reuse[0] != "disk":
                reuse = reuse[2] if len(reuse) > 2 else None
                if e.rank == 0:
                    print(f"[tensorfold] prefix cache: RAM evicted by memory guard; "
                          f"SSD fallback={reuse[2] if reuse else 0}", flush=True)
            self.ticks += 1
            need = self._need(s)
            disk = bool(reuse) and reuse[0] == "disk"
            if disk and not self._all(self.spill is not None and self.spill.check(str(reuse[1]), int(reuse[2]))):
                if e.rank == 0:
                    print(f"[tensorfold] spilled prompt {reuse[1]}: not on every rank, the prompt fills fresh",
                          flush=True)
                reuse, disk = None, False
            src = self.kept.get(int(reuse[0])) if reuse and not disk else None
            if reuse and not disk and src is None:
                raise OutOfStep("an admission continues a kept prompt this rank does not hold")
            base, how = self._place(need, src)
            if base is None:                             # (every rank at the same point: the same extents)
                raise NoRoom("the window's free extents are too small for this request now")
            slot = self.slots[index]
            self.free.remove(index)
            s.base, s.size = base, self.extents.size(need)
            slot.sc = self.m.pool_view(self.pool, index, base, s.size)
            s.sid, s.st = self.next_id, slot
            self.next_id += 1
            cut, s.snaps, s.lineage = 0, {}, None
            if disk:
                name, at, t0 = str(reuse[1]), int(reuse[2]), time.perf_counter()
                try:
                    snaps, host, lineage = self.spill.load(name, at, base)
                    loaded = at in snaps
                except Exception as exc:                 # noqa: BLE001  (every rank agrees on the outcome below)
                    print(f"[tensorfold] rank {e.rank}: spilled prompt {name} did not load ({type(exc).__name__}: "
                          f"{exc})", flush=True)
                    loaded = False
                if self._all(loaded):                    # else the rows written so far get filled over, fresh
                    cut, how = at, "disk"
                    self._restore(index, snaps[cut])
                    slot.sc.host = HostIds(host)
                    s.snaps, s.lineage = snaps, lineage
                    self.keep_stats[how] = self.keep_stats.get(how, 0) + 1
                    if e.rank == 0:
                        print(f"[tensorfold] spilled prompt {name}: continued at {cut} of {len(s.prompt)} (disk, "
                              f"{time.perf_counter() - t0:.2f}s)", flush=True)
                elif e.rank == 0:
                    print(f"[tensorfold] spilled prompt {name}: a rank could not read it, the prompt fills fresh",
                          flush=True)
            elif how != "fresh":
                cut = int(reuse[1])
                if how in ("copy", "move"):
                    self._copy_rows(src.base, base, cut)
                if how == "copy":
                    src.tick, src.hits = self.ticks, src.hits + 1
                self.keep_stats[how] = self.keep_stats.get(how, 0) + 1
                if e.rank == 0:
                    print(f"[tensorfold] kept prompt {src.eid}: continued at {cut} of {len(s.prompt)} ({how})",
                          flush=True)
                self._restore(index, src.snaps[cut])
                slot.sc.host = src.host.copy(cut)
                s.snaps = {b: v for b, v in src.snaps.items() if b <= cut}
                if self.spill is not None:               # its spilled rows to the boundary are this prompt's too
                    s.lineage = self.spill.lineage(src.disk, cut)
            image = None
            later = [p for p in positions if p >= cut]
            if later:
                rows = None
                torch.cuda.synchronize()
                before_vision = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                if e.rank == 0:                          # the tower runs only for the spans past the boundary
                    rows = torch.cat([e.tower.span_rows(pic)[max(0, cut - start):] for start, pic in s.vision.spans
                                      if start + pic.tokens > cut])
                image = (later, e._share_rows(rows, len(later)))
                # Drop the redundant local reference; image owns the shared rows.
                rows = None
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - before_vision
                print(f"[tensorfold] rank {e.rank}: vision memory peak +{peak / 2**20:.1f} MiB, "
                      f"admission budget {workspace[min(e.rank, 1)] / 2**20:.1f} MiB, "
                      f"image rows {len(later)}", flush=True)
            s.draft = bool(s.draft) and e.drafter is not None
            snap = None
            if KEEP:
                marks = self._marks(len(s.prompt))

                def snap(end: int) -> None:
                    if end in marks:
                        s.snaps[end] = self._snapshot(index)

            s.steps = e.prefill_steps(slot.sc, slot.dc if s.draft else None, list(s.prompt), image, start=cut,
                                      snap=snap)
            s.filled, s.prefill_s, s.cached = cut, 0.0, cut
            self.filling.append(s)
        except Exception as exc:
            self._broken(exc)
            raise
        finally:
            self._step(False)

    def _quick(self, s: Stream) -> bool:
        return len(s.prompt) - s.filled <= QUICK_ROWS

    def _fill(self, s: Stream) -> list[Stream]:
        """One chunk of a filling stream's prompt; at its last one its first token, and it decodes from the next
        round. Returns it if it ended there."""

        self._step(True)
        try:
            if not self._agree("fill", [self._shape(), s.sid], guard_bytes=768 * 2**20):
                self._memory_guard(768 * 2**20)
            t0 = time.perf_counter()
            try:
                s.filled = next(s.steps)
                last = None
            except StopIteration as end:
                last, s.filled = end.value, len(s.prompt)
            dt = time.perf_counter() - t0
            s.prefill_s += dt
            self.chunk_s, self.since_fill = dt, 0.0
            self.round_end = None                      # (ROUND_STATS: a fill is not host time between rounds)
            if last is None:
                return []
            self.filling.remove(s)
            s.steps = None
            self.streams[s.sid] = s                   # (a raise below leaves it to ``drop``)
            try:
                first = self._first(s, last)
            except Exception as exc:                 # noqa: BLE001  (this stream's own host work: only it fails)
                first = None
                self._stream_failed(s, exc)
            s.started = time.perf_counter()
            s.pending = first
            if first is not None:
                try:
                    s.take([first], self._ends(s))
                except Exception as exc:             # noqa: BLE001  (its reply's callback: only it fails)
                    self._stream_failed(s, exc)
                self._grammar_end(s)
            return [s] if s.done else []
        except Exception as exc:
            self._broken(exc)
            raise
        finally:
            self._step(False)

    def _first(self, s: Stream, last) -> int:
        """The reply's first token from the prompt's last row, under the reply's grammar when it has one."""

        try:
            constrain.first(s.constraint, last)
        except GrammarError as exc:
            self._grammar_failed(s, exc)
        first = self.e._sample(last, [len(s.prompt)], s.sampling)[0]
        try:
            constrain.advance(s.constraint, [first])
        except GrammarError as exc:
            self._grammar_failed(s, exc)
        return first

    @staticmethod
    def _grammar_failed(s: Stream, exc: GrammarError) -> None:
        """A reply whose grammar failed ends with that error at this round's end (every rank walks the same grammar
        over the same tokens, so every rank ends it at the same round); the other streams go on."""

        s.error, s.constraint = exc, None

    def _stream_failed(self, s: Stream, exc: Exception) -> None:
        """A stream whose own host work raised (its grammar, its samples, its reply's callback) ends now with that
        error; the other streams go on. That work runs no collective, so the ranks stay in step. A rank that raised
        where the others did not ends the stream alone: rank 0's finish then ends it on every rank, or the next
        step's digest check fails that step on every rank (``OutOfStep``), and the lane serves on. A device fault
        (``_device_fault``) is not the stream's: the forward may have failed and the lane may be poisoned, so it is
        raised again to the step's own handler, which breaks the lane."""

        if _device_fault(exc):
            raise exc
        s.error, s.constraint = exc, None
        if not s.done:
            s.done, s.finished = True, time.perf_counter()
        print(f"[tensorfold] rank {self.e.rank}: stream {s.sid} failed ({type(exc).__name__}: {str(exc)[:160]}); "
              "the other streams go on", flush=True)

    @staticmethod
    def _emit(s: Stream, new: list[int], ends) -> None:
        """A round's kept tokens to the stream. A stream whose grammar failed in this round ends with that error and
        sends none of them: they were sampled from logits the grammar no longer masked. Every rank walks the same
        grammar over the same tokens, so every rank ends it at the same round."""

        if s.error is not None:
            s.done, s.finished = True, time.perf_counter()
            return
        s.take(new, ends)
        MultiDecoder._grammar_end(s)

    @staticmethod
    def _grammar_end(s: Stream) -> None:
        """A reply whose grammar has taken its stop token ends there, also with ignore_eos (``_ends`` is empty then).
        Every rank walks the same grammar over the same tokens, so every rank ends it at the same round."""

        if not s.done and constrain.ended(s.constraint):
            s.done, s.finished = True, time.perf_counter()

    def _absorb_eager(self, items: list):
        """absorb_many of ``items`` on the side stream once the forward (on the current stream) has run; returns the
        event that marks it done."""

        main = torch.cuda.current_stream()
        side = getattr(self, "_absorb_stream", None)
        if side is None:
            side = self._absorb_stream = torch.cuda.Stream()
        ready = torch.cuda.Event()
        ready.record(main)
        side.wait_event(ready)
        with torch.cuda.stream(side):
            self.e.drafter.absorb_many(self.dpool, self.m.pool_view(self.pool, 0, 0, self.extents.total) if
                                       self._table_view is None else self._table_view, items, eager=True)
            done = torch.cuda.Event()
            done.record(side)
        return done

    def _choose_k(self, conf: list[float], kmax: int, live: int) -> int:
        """The k in 0 .. kmax with the most expected tokens a millisecond: 1 + the prefix survivals' sum over the
        round's table cost (this stream's rows beside the other streams' at their table depth)."""

        import math

        others = (live - 1) * (self._depth(live) + 1)
        best, best_v, surv, exp_tokens = 0, -1.0, 1.0, 1.0
        for kk in range(0, kmax + 1):
            if kk > 0:
                surv *= 1.0 / (1.0 + math.exp(-float(conf[kk - 1])))
                exp_tokens += surv
            rows = min(len(ROUND_MS), others + kk + 1)
            v = exp_tokens * live / (ROUND_MS[rows - 1] + DRAFT_MS)
            if v > best_v:
                best, best_v = kk, v
        return best

    def _joint_k(self, live: list[Stream], drafting: list[Stream], confs: list, k: int) -> dict:
        """Choose both verification depths together using their actual total rows.

        Only the amount of speculative work changes; target-model verification,
        position-based sampling, precision, and context are untouched.
        """
        import itertools
        import math

        curves = []
        for s, cf in zip(drafting, confs):
            values, survival = [1.0], 1.0
            for confidence in cf[:min(k, s.count - len(s.out))]:
                x = max(-80.0, min(80.0, float(confidence)))
                survival *= 1.0 / (1.0 + math.exp(-x))
                values.append(values[-1] + survival)
            curves.append(values)
        best, score = None, -1.0
        for depths in itertools.product(*(range(len(c)) for c in curves)):
            rows = len(live) + sum(depths)
            if rows > self.max_rows:
                continue
            expected = len(live) - len(drafting) + sum(c[d] for c, d in zip(curves, depths))
            value = expected / (ROUND_MS[min(len(ROUND_MS), rows) - 1] + DRAFT_MS)
            if value > score:
                best, score = depths, value
        return {s.sid: d for s, d in zip(drafting, best)}

    def _even_k(self, live: list[Stream], drafting: list[Stream], confs: list, k: int) -> dict:
        """TF_DS_DEPTH_POLICY=even: each drafting stream's k (a pure function of state every rank holds)."""

        import math

        def expected(cf: list[float], n: int) -> float:
            surv, e = 1.0, 1.0
            for j in range(n):
                surv *= 1.0 / (1.0 + math.exp(-float(cf[j])))
                e += surv
            return e

        left = [(s.count - len(s.out)) / max(1.0, getattr(s, "per_round", 2.0)) for s in drafting]
        c = max(range(len(drafting)), key=lambda i: (left[i], -i))
        ks, rows = {}, len(live)
        for i, (s, cf) in enumerate(zip(drafting, confs)):
            if i == c:
                continue
            need = EVEN_MARGIN * (s.count - len(s.out)) / max(1.0, left[c])
            kk, top = 0, min(k, s.count - len(s.out), len(cf))
            while kk < top and expected(cf, kk) < need:
                kk += 1
            ks[s.sid] = kk
            rows += kk
        s, cf = drafting[c], confs[c]
        top = max(0, min(self.depth_most, s.count - len(s.out), len(cf), self.max_rows - rows))
        best, best_v = 0, -1.0
        for kk in range(top + 1):
            v = expected(cf, kk) / (ROUND_MS[min(len(ROUND_MS), rows + kk) - 1] + DRAFT_MS)
            if v > best_v:
                best, best_v = kk, v
        ks[s.sid] = best
        return ks

    def _depth(self, live: int) -> int:
        k = DEPTH_BY[min(live, len(DEPTH_BY)) - 1] if DEPTH_BY else self.depth_most
        return max(0, min(k, self.depth_most, self.max_rows // max(live, 1) - 1))

    @torch.no_grad()
    def round(self, told: list | None = None) -> list[Stream]:
        """One round over every live stream; returns the streams that ended."""

        self._t_enter = time.perf_counter()
        live = [s for s in self.streams.values() if not s.done]
        if told is None and self.filling:
            s = next((f for f in self.filling if self._quick(f)), self.filling[0])
            owed = self.chunk_s * DECODE_SHARE / max(1e-6, 1.0 - DECODE_SHARE)
            if not live or self._quick(s) or self.since_fill >= owed:
                self._send(["fill", s.sid])
                return self._fill(s)
        if not live and told is None:
            return []
        k = self._depth(len(live))
        if told is None:
            self._send(["round", k])
        elif not live or told[1] != k:
            self._differs(told[1])
        t0 = time.perf_counter()
        done = self._round(live, k)
        self.since_fill += time.perf_counter() - t0
        return done

    def _differs(self, k: int) -> None:
        """A follower cannot run rank 0's round: a stream ended on this rank alone (``_stream_failed``), so it has no
        live stream or another draft depth. Rank 0 already waits in the round's digest check, so this rank takes part
        in it with a plan rank 0 never sends (rank 0's depth and a marker): the check fails on every rank, also when
        only the depths differ (``TF_DS_PARALLEL_DEPTH`` set apart on one rank), and each rank drops its streams.
        Does not return."""

        self._step(True)
        try:
            self._agree("round", [self._shape(), k, "out of step"])
        except Exception as exc:
            self._broken(exc)
            raise
        finally:
            self._step(False)
        raise OutOfStep("the ranks hold different live streams or draft depths")

    def _round(self, live: list[Stream], k: int) -> list[Stream]:
        e = self.e
        if os.environ.get("SPARK1017_KERNEL_TRACE"):
            from spark1017.profiling import begin
            begin(self)
        t0 = time.perf_counter()
        self._step(True)
        try:
            ta0 = time.perf_counter()
            _ht("round start")
            if not self._agree("round", [self._shape(), k], guard_bytes=384 * 2**20):
                self._memory_guard(384 * 2**20)
            _ht("agreed")
            if ROUND_STATS:
                torch.cuda.synchronize()
            t_agree = time.perf_counter() - ta0
            between = ta0 - self.round_end if getattr(self, "round_end", None) else 0.0
            pre = ta0 - getattr(self, "_t_enter", ta0)                # inside round() before the step (op send)
            windows, kept = [], []
            marks = [time.perf_counter()]

            def mark():
                if ROUND_STATS:
                    torch.cuda.synchronize()
                    marks.append(time.perf_counter())

            want = {s.sid: (min(k, s.count - len(s.out)) if s.draft else 0) for s in live}
            drafting = [s for s in live if want[s.sid] > 0]
            lookup = {}
            if os.environ.get("SPARK1017_LOOKUP") == "1":
                from spark1017.lookup import propose as lookup_propose
                lookup = lookup_propose(self, live, ROUND_MS)
                drafting = [s for s in drafting if s.sid not in lookup]
            if drafting:                                 # the rows known before the drafts: their Engram rows now
                self.runner.engram_touch([[*s.st.sc.host[max(0, s.st.sc.length - 8):s.st.sc.length].tolist(),
                                           s.pending] for s in live])
            proposed, confs = dict(lookup), {}
            _ht("touched")
            if drafting:
                rows = self._drafts([s.pending for s in drafting], [s.st.sc.length for s in drafting],
                                    [s.st.index for s in drafting])
                conf_rows = self.drafters[len(drafting)].last_conf
                even = None
                if DEPTH_POLICY == "joint" and len(drafting) > 1:
                    even = self._joint_k(live, drafting, conf_rows, k)
                if DEPTH_POLICY == "even" and len(drafting) > 1 and \
                        max(s.rounds for s in drafting) - min(s.rounds for s in drafting) <= EVEN_COHORT:
                    even = self._even_k(live, drafting, conf_rows, k)
                if os.environ.get("SPARK1017_ADAPTIVE_DEPTH", "0") == "1":
                    from spark1017.integration import choose
                    even = choose(self, live, drafting, conf_rows, [want[s.sid] for s in drafting], ROUND_MS, DRAFT_MS)
                for s, r, cf in zip(drafting, rows, conf_rows):
                    kk = want[s.sid]
                    if even is not None:
                        kk = even[s.sid]
                    elif DEPTH_POLICY in ("conf", "even", "joint"):
                        kk = self._choose_k(cf, min(kk, len(cf)), len(live))
                    if CONF_LOG:
                        self.k_hist.setdefault(len(live), [0] * 8)[kk] += 1
                    proposed[s.sid] = r[:kk]
                    confs[s.sid] = cf
            mark()
            grammars: dict[int, object] = {}             # a constrained stream's verify window (``constrain.chain``)
            for s in live:
                sc = s.st.sc
                P = sc.length
                drafts: list[int] = proposed.get(s.sid, [])
                if s.constraint is not None:             # the drafts its grammar cannot keep are cut before the forward
                    try:
                        drafts, grammars[s.sid] = constrain.chain(s.constraint, s.pending, drafts)
                    except GrammarError as exc:
                        self._grammar_failed(s, exc)
                    except Exception as exc:             # noqa: BLE001  (it keeps its rows this round, no tokens)
                        self._stream_failed(s, exc)
                window = [s.pending] + drafts
                sc.host.set(P, window)
                windows.append((s.st.index, s.base, s.size, P, window, sc.host.view()))
                kept.append((s, P, drafts))
            _ht("drafts back + plan")
            logits, taps = self.runner.forward(windows)
            _ht("forward launched")
            absorb_done = None
            if EAGER_ABSORB and taps is not None:
                eager, r = [], 0
                for (s, P, _), (_, _, _, _, window, _) in zip(kept, windows):
                    if s.draft:
                        eager.append((s.st.index, taps[r:r + len(window)], P))
                    r += len(window)
                if eager:
                    absorb_done = self._absorb_eager(eager)
            mark()
            done, r0, tokens = [], 0, 0
            t_absorb = 0.0
            absorbs = []
            for (s, P, drafts), (_, _, _, _, window, _) in zip(kept, windows):
                n = len(window)
                if not s.done:                           # (a stream that failed before the forward: its rows, no tokens)
                    try:
                        if s.constraint is not None and s.sid in grammars:
                            try:
                                constrain.mask(s.constraint, logits[r0:r0 + n], grammars[s.sid])
                            except GrammarError as exc:
                                self._grammar_failed(s, exc)
                        target = e._sample(logits[r0:r0 + n], [P + 1 + i for i in range(n)], s.sampling)
                        a = 0
                        while a < len(drafts) and drafts[a] == target[a]:
                            a += 1
                        if os.environ.get("SPARK1017_LOOKUP") == "1":
                            from spark1017.lookup import observe as lookup_observe
                            lookup_observe(self, s, len(drafts), a, ROUND_MS, DRAFT_MS)
                        if hasattr(self, "_spark1017_depth") and s.sid not in lookup:
                            from spark1017.integration import observe
                            observe(self, confs.get(s.sid, []), len(drafts), a)
                        if CONF_LOG and drafts and s.sid not in lookup:
                            import math

                            cf = confs.get(s.sid, [])
                            for j in range(min(len(drafts), a + 1)):          # drafts reached: the first miss included
                                self.conf_depth[j][0] += 1
                                self.conf_depth[j][1] += int(j < a)
                                if j < len(cf):
                                    b = min(9, int(10 / (1 + math.exp(-cf[j]))))
                                    self.conf_bins[b][0] += 1
                                    self.conf_bins[b][1] += int(j < a)
                        new = drafts[:a] + [target[a]]
                        s.st.sc.length = P + a + 1
                        if s.draft and taps is not None:
                            if absorb_done is not None:
                                self.dpool.views[s.st.index].absorbed = P + a + 1
                            else:
                                absorbs.append((s.st.index, taps[r0:r0 + a + 1], P))
                        s.counted(n)
                        ends = self._ends(s)
                        for i, t in enumerate(new):
                            if t in ends:
                                new = new[:i + 1]
                                break
                        new = new[:s.count - len(s.out)]
                        if s.constraint is not None:
                            try:
                                constrain.advance(s.constraint, new)
                            except GrammarError as exc:
                                self._grammar_failed(s, exc)
                        s.per_round = 0.75 * getattr(s, "per_round", 2.0) + 0.25 * len(new)
                        s.pending = new[-1]
                        self._emit(s, new, ends)
                        tokens += len(new)
                    except Exception as exc:             # noqa: BLE001  (this stream's own host work: only it fails)
                        self._stream_failed(s, exc)
                r0 += n
                if s.done:
                    done.append(s)
            if CONF_LOG:
                self.conf_rounds += 1
                if self.conf_rounds % 200 == 0 and e.rank == 0:
                    dep = [f"{j + 1}:{kept}/{n}" for j, (n, kept) in enumerate(self.conf_depth) if n]
                    bins = [f"{b / 10:.1f}:{kept}/{n}" for b, (n, kept) in enumerate(self.conf_bins) if n]
                    ks = [f"{n}:{','.join(str(x) for x in h[:self.depth_most + 1])}" for n, h in sorted(self.k_hist.items())]
                    print(f"[tensorfold] drafts kept by depth {' '.join(dep)}; by confidence {' '.join(bins)}; "
                          f"k chosen by streams {' '.join(ks)}", flush=True)
            _ht("sampled")
            if absorb_done is not None:                  # what runs on the main stream next sees the rings written
                torch.cuda.current_stream().wait_event(absorb_done)
            if absorbs:                                  # every drafting stream's kept rows into its rings, one pass
                ta = time.perf_counter()
                e.drafter.absorb_many(self.dpool, self.m.pool_view(self.pool, 0, 0, self.extents.total) if
                                      self._table_view is None else self._table_view, absorbs)
                t_absorb += time.perf_counter() - ta
            self.rounds += 1
            if os.environ.get("SPARK1017_KERNEL_TRACE"):
                from spark1017.profiling import end
                end(self)
            if HOST_TRACE:
                _ht("absorbed")
                marks = _HT.pop("_marks", [])
                prev = _HT.get("_end")
                if prev is not None and marks:
                    _HT.setdefault("between (op wait incl.)", []).append(marks[0][1] - prev)
                for (a, ta), (b, tb) in zip(marks, marks[1:]):
                    _HT.setdefault(f"{a} -> {b}", []).append(tb - ta)
                _HT["_end"] = time.perf_counter()
                _HT["_n"] = _HT.get("_n", 0) + 1
                if _HT["_n"] % 100 == 0:
                    import statistics
                    parts = [f"{k} {1000 * statistics.median(v[-100:]):.2f}" for k, v in _HT.items()
                             if not k.startswith("_")]
                    print(f"[tensorfold] rank {e.rank} host trace (median ms, {len(live)} streams): " + "; ".join(parts),
                          flush=True)
            round_elapsed = time.perf_counter() - t0
            if os.environ.get("SPARK1017_PROFILE"):
                from spark1017.integration import record_round
                record_round(self, r0, len(live), round_elapsed)
            self.round_log.append((len(live), r0, round_elapsed, tokens))
            del self.round_log[:-4096]
            if ROUND_STATS:
                mark()
                st = self.stage.setdefault(len(live), [0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                st[9] += pre if between < 1.0 else 0.0
                st[0] += 1
                st[1] += r0
                st[2] += tokens
                st[3] += marks[1] - marks[0]                       # drafts
                st[4] += marks[2] - marks[1]                       # Engram rows + the round's forward
                st[5] += marks[3] - marks[2] - t_absorb            # sampling, acceptance, emit
                st[6] += t_absorb
                st[7] += t_agree
                st[8] += between if between < 1.0 else 0.0         # host time from the last round's end (no fills)
                if st[0] % 100 == 0:
                    r = st[0]
                    print(f"[tensorfold] rank {e.rank} rounds at {len(live)} streams: {st[1] / r:.1f} rows, "
                          f"{st[2] / r:.2f} tokens a round; ms between {1000 * st[8] / r:.1f} (in round() "
                          f"{1000 * st[9] / r:.1f}) agree {1000 * st[7] / r:.1f} "
                          f"drafts {1000 * st[3] / r:.1f} forward {1000 * st[4] / r:.1f} sample {1000 * st[5] / r:.1f} "
                          f"absorb {1000 * st[6] / r:.1f}", flush=True)
            self.round_end = time.perf_counter()
            return done
        except Exception as exc:
            self._broken(exc)
            raise
        finally:
            self._step(False)

    def finish(self, done: list[Stream]) -> None:
        if ROUND_STATS and done:
            self.round_end = None                      # (a finish's link step is not counted as host time either)
        if not done:
            return
        sids = [s.sid for s in done if s.sid in self.streams]
        if not sids:
            return
        drops = self._covered(done) if KEEP else []
        # the spill's run name rides along only with the spill on, so the default path's message stays as it was
        self._send(["finish", sids, drops] + ([self.spill_nonce] if self.spill is not None else []))
        self._finish(sids, drops)

    def cancel(self, streams: list[Stream]) -> None:
        """Drop only abandoned requests, on both ranks, without spilling cancelled work."""
        sids = [s.sid for s in streams]
        self._send(["cancel", sids])
        self._cancel(sids)

    def _cancel(self, sids: list[int]) -> None:
        torch.cuda.synchronize()  # also retire side-stream work before freeing its slot
        for sid in sids:
            s = self.streams.pop(sid, None)
            if s is None:
                s = next((f for f in self.filling if f.sid == sid), None)
                if s is not None:
                    self.filling.remove(s)
            if s is None:
                continue
            steps = getattr(s, "steps", None)
            if steps is not None:
                steps.close()
                s.steps = None
            s.snaps = None
            s.vision = None
            self._release(s)
        self._memory_wait_for = set()

    def _release(self, s: Stream) -> None:
        self.free.append(s.st.index)
        self.free.sort()
        self.extents.give(s.base, s.size)
        s.st.sc = None

    def _finish(self, sids: list[int], drops: list[int] = ()) -> None:
        """Finished streams give their slots and extents back; with KEEP their prompts stay as kept prompts, the
        ``drops`` (kept prompts they cover) go, and so do the oldest past KEEP_ENTRIES. Spilled (TF_DS_SPILL_GIB):
        a dropped prompt's manifest goes once every new kept prompt that should have spilled did (the new manifests
        name its rows); prompts that only leave the window stay on disk."""

        kept_now: list[Kept] = []
        for sid in sids:
            s = self.streams.pop(sid, None)
            if s is None:
                continue
            if KEEP and getattr(s, "snaps", None):
                self._keep(s)
                kept_now.append(self.kept[self.next_kept - 1])
            else:
                self._release(s)
        sp = self.spill
        spilled = sp is not None and all(k.disk is not None for k in kept_now if k.top >= sp.min_top)
        for eid in drops:
            if int(eid) in self.kept:
                k = self.kept[int(eid)]
                if spilled and k.disk is not None:
                    sp.forget(k.disk["name"])
                self._forget(k)
        while len(self.kept) > KEEP_ENTRIES:
            self._forget(min(self.kept.values(), key=lambda k: k.tick))
        if sp is not None and kept_now:
            sp.recycle(pinned={k.disk["name"] for k in self.kept.values() if k.disk is not None})

    def drop(self) -> list[Stream]:
        """Every live stream fails (a step raised); their slots are free again."""

        out = list(self.streams.values()) + list(self.filling)
        try:
            self._send(["drop"])
        except Exception:                                # noqa: BLE001  (the link may be what failed)
            pass
        self._drop()
        return out

    def _drop(self) -> None:
        for s in list(self.streams.values()) + self.filling:
            steps = getattr(s, "steps", None)
            if steps is not None:
                steps.close()
                s.steps = None
            s.snaps = None
            self._release(s)
        self.streams.clear()
        self.filling.clear()
        for k in list(self.kept.values()):              # (the ranks may disagree about them now)
            self._forget(k)

    # -- followers ----------------------------------------------------------------------------------------------
    def follow(self, link: Link) -> None:
        """A follower rank: rank 0's steps, in order, forever (or until the link or the lane breaks)."""

        self._spark1017_follow_link = link

        while True:
            op = link.receive()
            _ht("op received")
            if op is None:
                raise RuntimeError("rank 0's step link closed")
            kind = op[0]
            try:
                if kind == "admit":
                    _, prompt, count, sampling, draft, stop_eos, index, positions, reuse, packed, workspace = op
                    s = Stream(list(prompt), int(count), _unpack(sampling), draft=bool(draft), stop_eos=bool(stop_eos),
                               constraint=constrain.follow(self.e, packed))
                    s.emit = lambda new: None
                    s.keys = None
                    self._admit(s, int(index), list(positions), reuse, tuple(workspace))
                elif kind == "fill":
                    self._fill(next(f for f in self.filling if f.sid == int(op[1])))
                elif kind == "round":
                    self.round(told=op)
                elif kind == "finish":
                    if len(op) > 3 and op[3]:
                        self.spill_nonce = str(op[3])    # rank 0's name for this run's spilled prompts
                    self._finish([int(x) for x in op[1]], [int(x) for x in op[2]])
                elif kind == "cancel":
                    self._cancel([int(x) for x in op[1]])
                elif kind == "drop":
                    self._drop()
            except NoRoom as exc:
                print(f"[tensorfold] rank {self.e.rank}: {exc}", flush=True)
            except OutOfStep as exc:
                print(f"[tensorfold] rank {self.e.rank}: {exc}", flush=True)
                self._drop()
