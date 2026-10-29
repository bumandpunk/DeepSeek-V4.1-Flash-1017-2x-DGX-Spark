// SPDX-License-Identifier: Apache-2.0
// Derived from TensorFold's engram_io.cpp; see NOTICE for the pinned upstream.
// Spark1017 additions: partitioned scale residency, bounded exact row cache,
// within-batch miss coalescing and read-path counters. No numerical transforms.
// Parallel row reads for the Engram tables: one pread per row on a pool of threads, so a decode step's rows (dozens,
// random offsets in ~95 GiB files) cost one NVMe latency instead of one per row. The pool's threads start once and
// wait for work (a round's reads used to start and join up to 48 threads a call, four calls a round); a call reads a
// layer's weight rows and scale rows together. At exit each thread started here is stopped and joined before anything
// it waits on is destroyed: glibc's pthread_cond_destroy waits for the condition variable's waiters, so a thread left
// waiting on one that exit() destroys holds the process in exit() until it is killed.
#include <torch/extension.h>
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>
#include "tiered_store.hpp"

namespace {

spark1017::ResidentRanges resident;
std::atomic<uint64_t> resident_prompt_hits{0};
bool storage_configured = false; // set once, after warm-up and before admission

// fn() at exit, as a static's destructor: one made when its threads start is destroyed before the statics made earlier
// (touch_cv below). Only in the process that made it: a forked child has none of its threads to join.
struct AtExit {
    void (*fn)();
    pid_t pid = getpid();
    ~AtExit() {
        if (getpid() == pid) fn();
    }
};

class Pool {
  public:
    explicit Pool(int threads) {
        for (int k = 0; k < threads; ++k) workers_.emplace_back([this] { loop(); });
    }

    // run job(0 .. n - 1) on the pool and this thread; returns when every index ran
    void run(int64_t n, const std::function<void(int64_t)>& job) {
        if (n <= 0) return;
        std::unique_lock<std::mutex> call(call_);          // one caller at a time
        {
            std::lock_guard<std::mutex> g(m_);
            job_ = &job;
            n_ = n;
            next_ = 0;
            finished_ = 0;
            ++gen_;
        }
        cv_.notify_all();
        work();
        std::unique_lock<std::mutex> g(m_);
        done_.wait(g, [&] { return finished_ >= n_; });
        job_ = nullptr;
    }

    // the threads end (a job under way is finished first) and are joined; a later run() runs on its calling thread
    void stop() {
        {
            std::lock_guard<std::mutex> g(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
        workers_.clear();
    }

  private:
    void work() {
        for (;;) {
            int64_t i = next_.fetch_add(1);
            if (i >= n_) break;
            (*job_)(i);
            if (finished_.fetch_add(1) + 1 >= n_) {
                std::lock_guard<std::mutex> g(m_);
                done_.notify_all();
            }
        }
    }

    void loop() {
        int64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> g(m_);
                cv_.wait(g, [&] { return gen_ != seen || stop_; });
                if (stop_) return;
                seen = gen_;
                if (job_ == nullptr) continue;
            }
            work();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex m_, call_;
    std::condition_variable cv_, done_;
    const std::function<void(int64_t)>* job_ = nullptr;
    std::atomic<int64_t> n_{0}, next_{0}, finished_{0};
    int64_t gen_ = 0;
    bool stop_ = false;
};

Pool& pool(int threads, int which = 0) {
    // pool 0 serves the reads a step waits for; pool 1 the reads started ahead of a prompt chunk, pool 2 a round's
    // (never destroyed, so a call still running at exit finds them whole; their threads are stopped at exit)
    static Pool* p[3] = {new Pool(std::max(1, threads - 1)), new Pool(std::max(1, threads - 1)),
                         new Pool(std::max(1, threads - 1))};
    static AtExit stopped{[] { for (Pool* q : p) q->stop(); }};
    return *p[which];
}

bool read_row(int fd, uint8_t* dst, int64_t bytes, int64_t off) {
    int64_t got = 0;
    while (got < bytes) {
        ssize_t r = pread(fd, dst + got, bytes - got, off + got);
        if (r <= 0) return false;
        got += r;
    }
    return true;
}

}  // namespace

// A layer's rows: idx [n] row ids; weight rows (fd_w, base_w, row_w bytes) into out_w, scale rows into out_s.
static void gather_rows2_on(int which, int64_t fd_w, int64_t base_w, int64_t row_w, int64_t fd_s, int64_t base_s,
                            int64_t row_s, const at::Tensor& idx, at::Tensor out_w, at::Tensor out_s, int64_t threads) {
    TORCH_CHECK(idx.dtype() == at::kLong && idx.is_contiguous() && !idx.is_cuda(), "idx: contiguous int64 CPU");
    TORCH_CHECK(out_w.dtype() == at::kByte && out_w.is_contiguous() && !out_w.is_cuda(), "out_w: contiguous uint8 CPU");
    TORCH_CHECK(out_s.dtype() == at::kByte && out_s.is_contiguous() && !out_s.is_cuda(), "out_s: contiguous uint8 CPU");
    const int64_t n = idx.numel();
    TORCH_CHECK(out_w.numel() >= n * row_w && out_s.numel() >= n * row_s, "out too small");
    const int64_t* ix = idx.data_ptr<int64_t>();
    uint8_t* dw = out_w.data_ptr<uint8_t>();
    uint8_t* ds = out_s.data_ptr<uint8_t>();
    std::atomic<int> failed(0);
    bool scales_ready = storage_configured && n > 0;
    if (scales_ready) for (int64_t i = 0; i < n; ++i) {
        const auto off = base_s + ix[i] * row_s;
        if (off < 0 || !resident.get(int(fd_s), uint64_t(off), uint64_t(row_s), ds + i * row_s)) {
            scales_ready = false; break;
        }
    }
    if (scales_ready) resident_prompt_hits.fetch_add(n, std::memory_order_relaxed);
    std::function<void(int64_t)> job = [&](int64_t j) {
        if (scales_ready) j *= 2;
        const int64_t i = j >> 1;
        const bool scale = j & 1;
        const auto fd = int(scale ? fd_s : fd_w);
        const auto rb = scale ? row_s : row_w;
        const auto off = (scale ? base_s : base_w) + ix[i] * rb;
        auto* dst = scale ? ds + i * row_s : dw + i * row_w;
        bool ok;
        if (storage_configured && off >= 0 && resident.get(fd, uint64_t(off), uint64_t(rb), dst)) {
            resident_prompt_hits.fetch_add(1, std::memory_order_relaxed);
            ok = true;
        } else ok = read_row(fd, dst, rb, off);
        if (!ok) failed = 1;
    };
    pool((int)threads, which).run(scales_ready ? n : 2 * n, job);
    TORCH_CHECK(!failed, "Engram row read failed");
}

void gather_rows2(int64_t fd_w, int64_t base_w, int64_t row_w, int64_t fd_s, int64_t base_s, int64_t row_s,
                  const at::Tensor& idx, at::Tensor out_w, at::Tensor out_s, int64_t threads) {
    gather_rows2_on(0, fd_w, base_w, row_w, fd_s, base_s, row_s, idx, out_w, out_s, threads);
}

// The same on the second pool, for reads started ahead of the step that needs them.
void gather_rows2_bg(int64_t fd_w, int64_t base_w, int64_t row_w, int64_t fd_s, int64_t base_s, int64_t row_s,
                     const at::Tensor& idx, at::Tensor out_w, at::Tensor out_s, int64_t threads) {
    gather_rows2_on(1, fd_w, base_w, row_w, fd_s, base_s, row_s, idx, out_w, out_s, threads);
}

// The same on pool ``which`` (1: ahead of a prompt chunk, 2: ahead of a decode round's layers).
void gather_rows2_at(int64_t which, int64_t fd_w, int64_t base_w, int64_t row_w, int64_t fd_s, int64_t base_s,
                     int64_t row_s, const at::Tensor& idx, at::Tensor out_w, at::Tensor out_s, int64_t threads) {
    TORCH_CHECK(which >= 0 && which <= 2, "pool 0, 1 or 2");
    gather_rows2_on((int)which, fd_w, base_w, row_w, fd_s, base_s, row_s, idx, out_w, out_s, threads);
}

// The single-table call (kept for tools): rows of one table.
void gather_rows(int64_t fd, int64_t base, int64_t row_bytes, const at::Tensor& idx, at::Tensor out, int64_t threads) {
    TORCH_CHECK(idx.dtype() == at::kLong && idx.is_contiguous() && !idx.is_cuda(), "idx: contiguous int64 CPU");
    TORCH_CHECK(out.dtype() == at::kByte && out.is_contiguous() && !out.is_cuda(), "out: contiguous uint8 CPU");
    const int64_t n = idx.numel();
    TORCH_CHECK(out.numel() >= n * row_bytes, "out too small");
    const int64_t* ix = idx.data_ptr<int64_t>();
    uint8_t* dst = out.data_ptr<uint8_t>();
    std::atomic<int> failed(0);
    std::function<void(int64_t)> job = [&](int64_t i) {
        if (!read_row((int)fd, dst + i * row_bytes, row_bytes, base + ix[i] * row_bytes)) failed = 1;
    };
    pool((int)threads).run(n, job);
    TORCH_CHECK(!failed, "Engram row read failed");
}

// ---- decode rounds' reads without a thread hand-off: Linux AIO on O_DIRECT descriptors ------------------------------
// A round's Engram read waits on two thread wake-ups on the pool path (its Python reader thread, then the pool's), and
// a thread woken after a round's idle took ~0.6 ms on GB10 (deep CPU idle states): the read of a 6-row window's rows
// took ~2 ms that way, ~0.4-0.8 ms as kernel AIO submitted by the calling thread and reaped by it (polling, no sleep).
// Each read takes its row's 4 KB-aligned span (one or two blocks) into a bounce slot of a ring; a batch's rows are
// copied out to its destinations as its reads complete. The bytes are the preads' (checked by aio_selftest).
namespace {

constexpr int64_t AIO_BLK = 4096;
constexpr int64_t AIO_SLOT = 2 * AIO_BLK;     // a row's span: at most two blocks (rows are far smaller than a block)

struct AioBatch {
    int64_t first = 0, n = 0, left = 0;        // reads [first, first + n) of the ring; still in flight
    std::vector<uint8_t*> dst;                 // per read: where its row goes (null: nothing copied, a touch)
    std::vector<int64_t> skew, bytes;
    std::vector<spark1017::RowKey> keys;
    std::vector<std::vector<uint8_t*>> followers;
    bool failed = false;
    bool touch = false;                        // nobody waits for it: dropped once done
};

struct Aio {
    aio_context_t ctx = 0;
    uint8_t* ring = nullptr;
    int64_t slots = 0, head = 0;               // reads issued so far (slot = index % slots)
    int64_t next_id = 1;
    std::unordered_map<int64_t, AioBatch> batches;
    std::vector<std::pair<int64_t, int64_t>> live;   // (batch id, first read) in issue order
    std::mutex mu;
    spark1017::HotRows hot;
    bool coalesce = false;
    uint64_t requested = 0, resident_hits = 0, hot_hits = 0, merged = 0;
    uint64_t submitted = 0, disk_bytes = 0, completed = 0, batches_count = 0;
};

Aio* aio = nullptr;

long sys_io_setup(unsigned n, aio_context_t* c) { return syscall(__NR_io_setup, n, c); }
long sys_io_submit(aio_context_t c, long n, struct iocb** p) { return syscall(__NR_io_submit, c, n, p); }
long sys_io_getevents(aio_context_t c, long mn, long n, struct io_event* e, struct timespec* t) {
    return syscall(__NR_io_getevents, c, mn, n, e, t);
}

// reap whatever has completed (no wait): copy rows out, count batches down
void aio_reap(Aio& a) {
    struct io_event ev[256];
    struct timespec zero = {0, 0};
    for (;;) {
        const long r = sys_io_getevents(a.ctx, 0, 256, ev, &zero);
        if (r <= 0) return;
        for (long q = 0; q < r; ++q) {
            const int64_t id = (int64_t)(ev[q].data >> 20), j = (int64_t)(ev[q].data & 0xFFFFF);
            auto it = a.batches.find(id);
            if (it == a.batches.end()) continue;
            AioBatch& b = it->second;
            if ((int64_t)ev[q].res < b.skew[j] + b.bytes[j]) b.failed = true;
            else {
                const auto* src = a.ring + ((b.first + j) % a.slots) * AIO_SLOT + b.skew[j];
                if (storage_configured) a.hot.put(b.keys[j], src);
                if (b.dst[j]) std::memcpy(b.dst[j], src, b.bytes[j]);
                for (auto* dst : b.followers[j]) if (dst) std::memcpy(dst, src, b.bytes[j]);
                ++a.completed;
            }
            b.left -= 1;
        }
    }
}

}  // namespace

// Open the AIO context and a ring of ``slots`` bounce slots (8 KB each); false when the kernel refuses.
bool aio_init(int64_t slots) {
    if (aio) return true;
    auto* a = new Aio();
    if (sys_io_setup((unsigned)std::min<int64_t>(slots, 65536), &a->ctx) < 0) { delete a; return false; }
    void* p = nullptr;
    if (posix_memalign(&p, AIO_BLK, (size_t)slots * AIO_SLOT) != 0) { delete a; return false; }
    a->ring = (uint8_t*)p;
    a->slots = slots;
    aio = a;
    return true;
}

namespace {

// The submission itself (a.mu held): n rows of idx, copied to dw / ds as they complete when given (null: a touch).
int64_t aio_submit_locked(Aio& a, int64_t fd_w, int64_t base_w, int64_t row_w, int64_t fd_s, int64_t base_s,
                          int64_t row_s, const int64_t* ix, int64_t n, uint8_t* dw, uint8_t* ds) {
    const int64_t reads = 2 * n;
    const bool copy = dw != nullptr;
    // the ring's slots [head, head + reads) must be free: reap until every batch on them is done (a live batch
    // [first, first + n) holds them when it overlaps the last lap's [head - slots, head + reads - slots))
    for (;;) {
        bool busy = false;
        for (auto& [id, first] : a.live) {
            auto it = a.batches.find(id);
            if (it != a.batches.end() && it->second.left > 0 && first < a.head + reads - a.slots &&
                first + it->second.n > a.head - a.slots) {
                busy = true;
                break;
            }
        }
        if (!busy) break;
        aio_reap(a);
    }
    const int64_t id = a.next_id++;
    AioBatch& b = a.batches[id];
    b.first = a.head;
    b.n = reads;
    b.left = 0;
    b.touch = !copy;
    b.dst.assign(reads, nullptr);
    b.skew.assign(reads, 0);
    b.bytes.assign(reads, 0);
    b.keys.resize(reads);
    b.followers.resize(reads);
    std::vector<struct iocb> cbs(reads);
    std::vector<struct iocb*> ptrs;
    ptrs.reserve(reads);
    std::unordered_map<spark1017::RowKey, int64_t, spark1017::RowHash> pending;
    if (a.coalesce) pending.reserve(reads);
    ++a.batches_count;
    a.requested += reads;
    for (int64_t j = 0; j < reads; ++j) {
        const int64_t i = j >> 1;
        const bool sc = j & 1;
        const int64_t rb = sc ? row_s : row_w;
        const int64_t off = (sc ? base_s : base_w) + ix[i] * rb;
        const int64_t start = off & ~(AIO_BLK - 1);
        b.skew[j] = off - start;
        b.bytes[j] = rb;
        if (copy) b.dst[j] = sc ? ds + i * row_s : dw + i * row_w;
        const spark1017::RowKey key{int(sc ? fd_s : fd_w), uint32_t(rb), uint64_t(off)};
        b.keys[j] = key;
        if (storage_configured && resident.get(key.fd, key.offset, key.bytes, b.dst[j])) {
            ++a.resident_hits; continue;
        }
        if (storage_configured && a.hot.get(key, b.dst[j])) { ++a.hot_hits; continue; }
        if (a.coalesce) {
            auto [at, inserted] = pending.emplace(key, j);
            if (!inserted) {
                b.followers[at->second].push_back(b.dst[j]); ++a.merged; continue;
            }
        }
        std::memset(&cbs[j], 0, sizeof(struct iocb));
        cbs[j].aio_fildes = (uint32_t)(sc ? fd_s : fd_w);
        cbs[j].aio_lio_opcode = IOCB_CMD_PREAD;
        cbs[j].aio_buf = (uint64_t)(uintptr_t)(a.ring + ((a.head + j) % a.slots) * AIO_SLOT);
        cbs[j].aio_nbytes = (uint64_t)(((off + rb - start) + AIO_BLK - 1) & ~(AIO_BLK - 1));
        cbs[j].aio_offset = start;
        cbs[j].aio_data = ((uint64_t)id << 20) | (uint64_t)j;
        ptrs.push_back(&cbs[j]);
        ++b.left;
    }
    a.head += reads;
    a.live.emplace_back(id, b.first);
    int64_t sent = 0;
    const auto physical = int64_t(ptrs.size());
    while (sent < physical) {
        const long r = sys_io_submit(a.ctx, physical - sent, ptrs.data() + sent);
        if (r <= 0) {
            b.left -= physical - sent;            // never submitted
            b.failed = true;
            break;
        }
        for (int64_t q = sent; q < sent + r; ++q) a.disk_bytes += ptrs[q]->aio_nbytes;
        sent += r;
        a.submitted += r;
    }
    // finished touches leave the bookkeeping (a waited-for batch leaves in aio_wait)
    for (size_t k = 0; k < a.live.size();) {
        auto it = a.batches.find(a.live[k].first);
        if (it == a.batches.end() || (it->second.touch && it->second.left == 0)) {
            if (it != a.batches.end()) a.batches.erase(it);
            a.live.erase(a.live.begin() + k);
        } else {
            ++k;
        }
    }
    return id;
}

// Touches (a round's first rows, read only to wake the drive ahead of the round's own read; nobody waits for them) go
// through this thread, so a submission that blocks (the second touch's io_submit took ~0.6 ms on one node) never holds
// the round's thread. A woken thread's latency does not matter here: the round's read comes milliseconds later.
struct TouchReq {
    int64_t fd_w, base_w, row_w, fd_s, base_s, row_s;
    std::vector<int64_t> idx;
};
std::mutex touch_m;
std::condition_variable touch_cv;
std::vector<TouchReq> touch_q;
std::thread* touch_thread = nullptr;
bool touch_stop = false;                       // at exit: the thread ends, and a later touch is dropped

void touch_loop() {
    for (;;) {
        std::vector<TouchReq> todo;
        {
            std::unique_lock<std::mutex> g(touch_m);
            touch_cv.wait(g, [] { return !touch_q.empty() || touch_stop; });
            if (touch_stop) return;            // touches still queued are dropped (nobody waits for one)
            todo.swap(touch_q);
        }
        for (auto& t : todo) {
            std::lock_guard<std::mutex> g(aio->mu);
            aio_submit_locked(*aio, t.fd_w, t.base_w, t.row_w, t.fd_s, t.base_s, t.row_s, t.idx.data(),
                              (int64_t)t.idx.size(), nullptr, nullptr);
        }
    }
}

// The touch thread ends and is joined: at exit, before touch_cv is destroyed (its AtExit is made after touch_cv)
void stop_touch() {
    {
        std::lock_guard<std::mutex> g(touch_m);
        touch_stop = true;
    }
    touch_cv.notify_all();
    touch_thread->join();
}

}  // namespace

// Submit a layer's weight rows (O_DIRECT fd_w at base_w, row_w bytes each) and scale rows of idx; with copy, rows land
// in out_w / out_s as they complete (keep both alive until aio_wait returns); without (a touch), nothing is copied.
// Returns the batch id. Holds the GIL (the submission is a syscall or two).
int64_t aio_start(int64_t fd_w, int64_t base_w, int64_t row_w, int64_t fd_s, int64_t base_s, int64_t row_s,
                  const at::Tensor& idx, at::Tensor out_w, at::Tensor out_s, int64_t copy) {
    TORCH_CHECK(aio != nullptr, "aio_init first");
    TORCH_CHECK(idx.dtype() == at::kLong && idx.is_contiguous() && !idx.is_cuda(), "idx: contiguous int64 CPU");
    TORCH_CHECK(row_w + AIO_BLK <= AIO_SLOT && row_s + AIO_BLK <= AIO_SLOT, "rows wider than a block");
    Aio& a = *aio;
    std::lock_guard<std::mutex> g(a.mu);
    const int64_t n = idx.numel(), reads = 2 * n;
    TORCH_CHECK(reads <= a.slots && reads < (1 << 20), "too many rows for the AIO ring");
    if (copy) {
        TORCH_CHECK(out_w.dtype() == at::kByte && out_w.is_contiguous() && out_w.numel() >= n * row_w, "out_w");
        TORCH_CHECK(out_s.dtype() == at::kByte && out_s.is_contiguous() && out_s.numel() >= n * row_s, "out_s");
    }
    return aio_submit_locked(a, fd_w, base_w, row_w, fd_s, base_s, row_s, idx.data_ptr<int64_t>(), n,
                             copy ? out_w.data_ptr<uint8_t>() : nullptr, copy ? out_s.data_ptr<uint8_t>() : nullptr);
}

// A touch of idx's rows, submitted by the touch thread (returns at once; nothing is copied, nobody waits).
void aio_touch(int64_t fd_w, int64_t base_w, int64_t row_w, int64_t fd_s, int64_t base_s, int64_t row_s,
               const at::Tensor& idx) {
    TORCH_CHECK(aio != nullptr, "aio_init first");
    TORCH_CHECK(idx.dtype() == at::kLong && idx.is_contiguous() && !idx.is_cuda(), "idx: contiguous int64 CPU");
    TORCH_CHECK(row_w + AIO_BLK <= AIO_SLOT && row_s + AIO_BLK <= AIO_SLOT, "rows wider than a block");
    TORCH_CHECK(2 * idx.numel() <= aio->slots, "too many rows for the AIO ring");
    const int64_t* ix = idx.data_ptr<int64_t>();
    {
        std::lock_guard<std::mutex> g(touch_m);
        if (touch_stop) return;                // at exit: dropped
        if (!touch_thread) {
            touch_thread = new std::thread(touch_loop);
            static AtExit stopped{stop_touch};
        }
        touch_q.push_back({fd_w, base_w, row_w, fd_s, base_s, row_s, std::vector<int64_t>(ix, ix + idx.numel())});
    }
    touch_cv.notify_one();
}

// Reap (polling, the GIL released) until batch id's rows are all in; raises if one failed. A batch is waited for once.
void aio_wait(int64_t id) {
    TORCH_CHECK(aio != nullptr, "aio_init first");
    Aio& a = *aio;
    bool failed = false;
    {
        py::gil_scoped_release rel;
        for (;;) {
            std::lock_guard<std::mutex> g(a.mu);
            auto it = a.batches.find(id);
            if (it == a.batches.end()) break;
            aio_reap(a);
            if (it->second.left == 0) {
                failed = it->second.failed;
                a.batches.erase(it);
                for (size_t k = 0; k < a.live.size(); ++k)
                    if (a.live[k].first == id) { a.live.erase(a.live.begin() + k); break; }
                break;
            }
        }
    }
    TORCH_CHECK(!failed, "Engram row read (AIO) failed");
    if (storage_configured && id % 2000 == 0) {
        std::lock_guard<std::mutex> guard(a.mu);
        std::fprintf(stderr, "[spark1017] storage logical=%llu resident=%llu hot=%llu merged=%llu disk=%llu bytes=%llu prompt_resident=%llu\n",
            (unsigned long long)a.requested, (unsigned long long)a.resident_hits,
            (unsigned long long)a.hot_hits, (unsigned long long)a.merged,
            (unsigned long long)a.submitted, (unsigned long long)a.disk_bytes,
            (unsigned long long)resident_prompt_hits.load(std::memory_order_relaxed));
    }
}

// Initialization-only APIs. Mount immutable tables and keep their descriptors
// open. The adapter loads only this TP rank's hash-column ranges after warm-up.
void cache_resident_add(int64_t fd, int64_t offset, int64_t bytes, int64_t alias) {
    TORCH_CHECK(aio != nullptr && !storage_configured, "configure once after aio_init, before serving");
    TORCH_CHECK(offset >= 0 && bytes > 0, "positive resident range");
    std::lock_guard<std::mutex> guard(aio->mu);
    resident.add(int(fd), uint64_t(offset), uint64_t(bytes), int(alias));
}

uint64_t cache_configure(int64_t hot_bytes, bool coalesce) {
    TORCH_CHECK(aio != nullptr && !storage_configured, "configure once after aio_init, before serving");
    TORCH_CHECK(hot_bytes >= 0, "hot cache budget must be nonnegative");
    std::lock_guard<std::mutex> guard(aio->mu);
    resident.seal();
    auto allocated = aio->hot.configure(size_t(hot_bytes));
    aio->coalesce = coalesce;
    storage_configured = true;
    return allocated;
}

std::unordered_map<std::string, uint64_t> cache_stats() {
    TORCH_CHECK(aio != nullptr, "aio_init first");
    std::lock_guard<std::mutex> guard(aio->mu);
    return {{"logical_reads", aio->requested}, {"resident_hits", aio->resident_hits},
            {"hot_hits", aio->hot_hits}, {"coalesced", aio->merged},
            {"disk_reads_submitted", aio->submitted}, {"disk_bytes_submitted", aio->disk_bytes},
            {"disk_reads_completed", aio->completed}, {"batches", aio->batches_count},
            {"resident_bytes", resident.allocated_bytes()}, {"hot_bytes", aio->hot.allocated_bytes()},
            {"resident_prompt_hits", resident_prompt_hits.load(std::memory_order_relaxed)}};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cache_resident_add", &cache_resident_add);
    m.def("cache_configure", &cache_configure);
    m.def("cache_stats", &cache_stats);
    m.def("aio_init", &aio_init, "the decode lane's AIO context and bounce ring");
    m.def("aio_start", &aio_start, "submit a layer's weight and scale rows as AIO reads (O_DIRECT descriptors)");
    m.def("aio_wait", &aio_wait, "reap until a batch's rows are in");
    m.def("aio_touch", &aio_touch, "a touch submitted by the touch thread (returns at once)");
    m.def("gather_rows", &gather_rows, "parallel preads of rows into a CPU buffer",
          py::call_guard<py::gil_scoped_release>());
    m.def("gather_rows2", &gather_rows2, "a layer's weight and scale rows in one parallel pass",
          py::call_guard<py::gil_scoped_release>());
    m.def("gather_rows2_bg", &gather_rows2_bg, "gather_rows2 on the background pool",
          py::call_guard<py::gil_scoped_release>());
    m.def("gather_rows2_at", &gather_rows2_at, "gather_rows2 on reader pool 0, 1 or 2",
          py::call_guard<py::gil_scoped_release>());
}
