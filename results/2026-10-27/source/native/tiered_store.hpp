// SPDX-License-Identifier: Apache-2.0
// Spark1017: bounded byte-preserving storage for immutable Engram tables.
#pragma once
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace spark1017 {

struct RowKey {
    int fd = -1;
    uint32_t bytes = 0;
    uint64_t offset = 0;
    bool operator==(const RowKey& b) const {
        return fd == b.fd && bytes == b.bytes && offset == b.offset;
    }
};
struct RowHash {
    size_t operator()(const RowKey& k) const {
        uint64_t x = k.offset ^ (uint64_t(uint32_t(k.fd)) << 32) ^ k.bytes;
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27; x *= 0x94d049bb133111ebULL;
        return size_t(x ^ (x >> 31));
    }
};

// The caller serializes get/put with the AIO ring's existing mutex. Four ways
// bound lookup and replacement cost; payloads never allocate on the decode path.
class HotRows {
    struct Entry {
        RowKey key;
        uint64_t tick = 0;
        std::array<uint8_t, 256> data{};
    };
    std::vector<Entry> entries_;
    size_t mask_ = 0;
    uint64_t clock_ = 0;
public:
    size_t configure(size_t budget) {
        entries_.clear(); entries_.shrink_to_fit(); clock_ = 0;
        size_t sets = 1;
        if (budget < 4 * sizeof(Entry)) return 0;
        while (sets <= budget / (8 * sizeof(Entry))) sets *= 2;
        entries_.resize(sets * 4); mask_ = sets - 1;
        return allocated_bytes();
    }
    size_t allocated_bytes() const { return entries_.size() * sizeof(Entry); }
    bool get(const RowKey& key, uint8_t* dst) {
        if (entries_.empty() || key.bytes == 0 || key.bytes > 256) return false;
        auto* set = entries_.data() + (RowHash{}(key) & mask_) * 4;
        for (int i = 0; i < 4; ++i) if (set[i].tick && set[i].key == key) {
            if (dst) std::memcpy(dst, set[i].data.data(), key.bytes);
            set[i].tick = ++clock_; return true;
        }
        return false;
    }
    void put(const RowKey& key, const uint8_t* src) {
        if (entries_.empty() || key.bytes == 0 || key.bytes > 256) return;
        auto* set = entries_.data() + (RowHash{}(key) & mask_) * 4;
        auto* victim = set;
        for (int i = 0; i < 4; ++i) {
            if (set[i].tick && set[i].key == key) { victim = &set[i]; break; }
            if (set[i].tick < victim->tick) victim = &set[i];
        }
        victim->key = key;
        std::memcpy(victim->data.data(), src, key.bytes);
        victim->tick = ++clock_;
    }
};

// Build before serving and seal once. Concurrent prompt-reader threads then
// perform only bounds checks and memcpy, without a mutex or reference counting.
// The process keeps source descriptors open and the source files immutable.
class ResidentRanges {
    struct Allocation {
        std::unique_ptr<uint8_t[]> data;
        uint64_t size;
        dev_t device;
        ino_t inode;
    };
    struct Range { int fd; uint64_t offset, bytes; const uint8_t* data; };
    std::vector<Allocation> allocations_;
    std::vector<Range> ranges_;
    bool sealed_ = false;
public:
    size_t allocated_bytes() const {
        size_t total = 0;
        for (const auto& a : allocations_) total += a.size;
        return total;
    }
    void add(int fd, uint64_t offset, uint64_t bytes, int alias = -1) {
        if (sealed_) throw std::logic_error("resident ranges already sealed");
        struct stat st{}, at{};
        if (!bytes || fstat(fd, &st) || st.st_size < 0 || offset > uint64_t(st.st_size) ||
            bytes > uint64_t(st.st_size) - offset || bytes > std::numeric_limits<size_t>::max())
            throw std::invalid_argument("resident range outside source file");
        if (alias >= 0 && (fstat(alias, &at) || at.st_dev != st.st_dev || at.st_ino != st.st_ino))
            throw std::invalid_argument("resident alias must name the same inode");
        Allocation a{std::unique_ptr<uint8_t[]>(new uint8_t[size_t(bytes)]), bytes, st.st_dev, st.st_ino};
        uint64_t done = 0;
        while (done < bytes) {
            size_t n = size_t(std::min<uint64_t>(bytes - done, 16ULL << 20));
            ssize_t got = pread(fd, a.data.get() + done, n, off_t(offset + done));
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) throw std::runtime_error("resident table read failed");
            done += uint64_t(got);
        }
        auto* data = a.data.get();
        allocations_.push_back(std::move(a));
        ranges_.push_back({fd, offset, bytes, data});
        if (alias >= 0 && alias != fd) ranges_.push_back({alias, offset, bytes, data});
    }
    void seal() { sealed_ = true; }
    bool get(int fd, uint64_t offset, uint64_t bytes, uint8_t* dst) const {
        for (const auto& r : ranges_) {
            if (r.fd != fd || offset < r.offset) continue;
            uint64_t relative = offset - r.offset;
            if (relative <= r.bytes && bytes <= r.bytes - relative) {
                if (dst && bytes) std::memcpy(dst, r.data + relative, size_t(bytes));
                return true;
            }
        }
        return false;
    }
};
} // namespace spark1017
