// An index of resources by the title memory they start at, so a lookup by address visits the few
// resources near it instead of every resource ever noted. It only narrows the candidates: the caller
// still applies its own test to each one (including its live-reference filter), so a lookup answers what
// a scan of all resources would. Entries stay as long as the canonical record does; a released resource
// is filtered by the caller, never removed here.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>

namespace x2w_startup {
// Whether [address, address + span) meets a resource of `bytes` bytes at `base` (a null base never does).
// A resource of unknown size (bytes == 0) is touched only at its base.
inline bool address_span_touches(std::uint64_t address, std::uint64_t span, std::uint64_t base, std::uint64_t bytes)
{
    return base && (address <= base ? base - address < span : address - base < bytes);
}

// The caller serializes access (note_lock: shared for lookups, exclusive for insert and erase).
template <class Key>
class AddressIndex {
public:
    // Makes (base, key) an entry of `bytes`: adds it, or replaces the entry the key already has at that
    // base. The new node is allocated before the old one goes, so a throw changes nothing.
    void insert(std::uint64_t base, std::uint64_t bytes, Key key)
    {
        const std::uint64_t extent = (std::max)(bytes, std::uint64_t{1});
        Group* old_group = nullptr;
        typename std::multimap<std::uint64_t, Entry>::iterator old_node;
        for (Group& group : groups_) {
            const auto [first, last] = group.by_base.equal_range(base);
            for (auto it = first; it != last; ++it)
                if (it->second.key == key) {
                    if (it->second.extent == extent) return;
                    old_group = &group;
                    old_node = it;
                }
        }
        Group& group = groups_[extent > large];
        group.by_base.emplace(base, Entry{key, extent});
        group.max_extent = (std::max)(group.max_extent, extent);
        if (old_group) old_group->by_base.erase(old_node);
    }
    // Removes the entry (base, key) if there is one; never throws.
    void erase(std::uint64_t base, Key key)
    {
        for (Group& group : groups_) {
            const auto [first, last] = group.by_base.equal_range(base);
            for (auto it = first; it != last; ++it)
                if (it->second.key == key) {
                    group.by_base.erase(it);
                    break;
                }
        }
    }
    // Calls f(key) for every resource whose [base, base + max(bytes, 1)) meets [from, from + span), in no
    // particular order; stops when f returns true. That includes every resource address_span_touches
    // accepts for (from, span), and every resource containing `from` when span is at least 1.
    template <class F>
    void for_each_reaching(std::uint64_t from, std::uint64_t span, F&& f) const
    {
        const std::uint64_t to = span > (std::numeric_limits<std::uint64_t>::max)() - from
            ? (std::numeric_limits<std::uint64_t>::max)() : from + span;
        for (const Group& group : groups_) {
            const auto start = group.by_base.lower_bound(from);
            for (auto it = start; it != group.by_base.end() && it->first < to; ++it)
                if (f(it->second.key)) return;
            // Nothing before a base that lies max_extent below `from` reaches it.
            for (auto it = start; it != group.by_base.begin();) {
                --it;
                if (from - it->first >= group.max_extent) break;
                if (from - it->first < it->second.extent && f(it->second.key)) return;
            }
        }
    }
    bool contains(std::uint64_t base, Key key) const
    {
        for (const Group& group : groups_) {
            const auto [first, last] = group.by_base.equal_range(base);
            for (auto it = first; it != last; ++it)
                if (it->second.key == key) return true;
        }
        return false;
    }
    std::size_t size() const { return groups_[0].by_base.size() + groups_[1].by_base.size(); }

private:
    // Large resources get their own group, so the backward search below an address stays short.
    static constexpr std::uint64_t large = 64 * 1024;
    struct Entry {
        Key key;
        std::uint64_t extent;
    };
    struct Group {
        std::multimap<std::uint64_t, Entry> by_base;
        std::uint64_t max_extent{};
    };
    Group groups_[2];
};

// Stores records[key] = info and indexes it at `base` (0: not indexed) as one step: if either allocation
// throws, the record and the index are as they were, so no stored record is missing from lookups. The
// index node comes first; the only throwing step left is adding a new key to `records`, and it is
// undone by erasing that node (replacing an existing record cannot throw).
template <class Records, class Key, class Info>
void store_indexed(AddressIndex<Key>& index, Records& records, const Key& key, std::uint64_t base,
                   std::uint64_t bytes, const Info& info)
{
    if (base) index.insert(base, bytes, key);
    const bool known = records.find(key) != records.end();
    try {
        records.insert_or_assign(key, info);
    } catch (...) {
        if (base && !known) index.erase(base, key);
        throw;
    }
}
} // namespace x2w_startup
