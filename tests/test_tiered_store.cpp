// SPDX-License-Identifier: Apache-2.0
#include "../native/tiered_store.hpp"
#include <cassert>
#include <fcntl.h>
#include <iostream>
#include <random>
#include <thread>

int main() {
    using namespace spark1017;
    char path[] = "/tmp/spark1017-store-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0);
    std::vector<uint8_t> source(1 << 20);
    std::mt19937_64 rng(1017);
    for (auto& v : source) v = uint8_t(rng());
    assert(write(fd, source.data(), source.size()) == ssize_t(source.size()));
    int alias = open(path, O_RDONLY); assert(alias >= 0);
    unlink(path);
    ResidentRanges store;
    store.add(fd, 37, 524288, alias);
    assert(store.allocated_bytes() == 524288);
    bool failed = false;
    try { store.add(fd, source.size() - 1, 2); } catch (const std::invalid_argument&) { failed = true; }
    assert(failed);
    store.seal(); failed = false;
    try { store.add(fd, 0, 1); } catch (const std::logic_error&) { failed = true; }
    assert(failed);
    std::vector<std::thread> readers;
    for (int t = 0; t < 8; ++t) readers.emplace_back([&, t] {
        std::mt19937_64 prng(1017 + t); std::array<uint8_t, 256> out;
        for (int i = 0; i < 10000; ++i) {
            size_t offset = 37 + prng() % (524288 - 256);
            size_t size = 1 + prng() % 256;
            assert(store.get(t % 2 ? fd : alias, offset, size, out.data()));
            assert(!memcmp(out.data(), source.data() + offset, size));
        }
    });
    for (auto& t : readers) t.join();
    std::array<uint8_t, 256> out;
    assert(!store.get(fd, 36, 1, out.data()));
    assert(!store.get(fd, 37 + 524288, 1, out.data()));
    assert(!store.get(fd, UINT64_MAX, 128, out.data()));
    assert(!store.get(fd + 1000, 37, 1, out.data()));
    HotRows hot;
    const size_t cap = hot.configure(16384); assert(cap <= 16384 && cap > 0);
    size_t hits = 0, misses = 0;
    // Heavy collisions and varied payload sizes: every hit must match source.
    for (int i = 0; i < 100000; ++i) {
        RowKey key{fd, uint32_t(1 + rng() % 256), uint64_t(rng() % (source.size() - 256))};
        if (i % 3 == 0) key = {alias, 256, 17};
        if (hot.get(key, out.data())) {
            ++hits; assert(!memcmp(out.data(), source.data() + key.offset, key.bytes));
        } else {
            ++misses; hot.put(key, source.data() + key.offset);
            assert(hot.get(key, out.data()));
            assert(!memcmp(out.data(), source.data() + key.offset, key.bytes));
        }
    }
    assert(hits > 10000 && misses > 10000);
    assert(hot.configure(0) == 0 && !hot.get({alias, 256, 17}, out.data()));
    close(alias); close(fd);
    std::cout << "80000 concurrent resident reads; 100000 collision-heavy cache reads: exact\n";
}
