#pragma once

// Which prefix calls a replay can leave out (x2w_replay --prefix-live, v1). Pure logic, no Windows types: the player feeds the
// prefix calls in sequence order and asks, per call in add order, whether the replay still needs it.
//
// Rule (v1), counted on a real recording: every creation, AddRef and Release stays, so no object
// dependency or reference count has to be reasoned about. Left out:
//  - a descriptor write (device slots 17 to 22 and 70) that a later write to the SAME destination handle replaces. A copy
//    (device 23 or 24) pins every write before it: a copy may read them, so the table of "last writes" starts again after it;
//  - the repeats of a call that only teaches the player a translation, per (object, its creation): the first Map (resource
//    slot 8) and the first GetGPUVirtualAddress (resource slot 11). Every Unmap (resource slot 9) after that object's first Map
//    is left out; an Unmap before any Map stays. Heap starts (descriptor heap slots 9 and 10) are NEVER left out: a title calls
//    them in the struct-return form, so the recorded object is the caller's stack address, not the heap, and two heaps queried
//    from one stack slot would share a key (the second one's whole handle translation would be lost).
//    The creation is the sequence number of the last output naming the pointer (0 when none), so a pointer the title got again
//    for a new object starts over;
//  - pure queries that return a value: device slots 25 and 108 (resource allocation info).
// A later version can add the closure over dead objects (skip what was created and released before the window).

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace x2w_replay {

using u64 = std::uint64_t;

struct LiveCall {
    u64 seq;
    std::uint16_t kind, slot;
    u64 object;
    u64 args[11];          // the call's argument words after the object, as recorded
    const u64* outputs;    // the OUTP values of this call (objects it made), or null
    std::size_t output_count;
};

struct PrefixLiveCounts {
    u64 calls = 0, superseded_writes = 0, repeated_teaching = 0, queries = 0;
};

class PrefixLive {
public:
    static constexpr std::uint16_t device_kind = 0, resource_kind = 7;

    // Prefix calls, in sequence order.
    void add(const LiveCall& call)
    {
        const std::size_t index = left_out_.size();
        left_out_.push_back(false);
        ++counts_.calls;
        if (call.kind == device_kind) {
            add_device(call, index);
        } else if (call.kind == resource_kind) {
            add_resource(call, index);
        }
        for (std::size_t i = 0; i < call.output_count && call.outputs; ++i) created_[call.outputs[i]] = call.seq;
    }

    // `index` is the position in add order.
    bool left_out(std::size_t index) const { return index < left_out_.size() && left_out_[index]; }
    const PrefixLiveCounts& counts() const { return counts_; }

private:
    struct Key {
        u64 object, creation;
        bool operator==(const Key& other) const { return object == other.object && creation == other.creation; }
    };
    struct KeyHash {
        std::size_t operator()(const Key& key) const
        {
            return static_cast<std::size_t>(key.object * 0x9e3779b97f4a7c15ull ^ (key.creation + 0x7f4a7c15ull) * 0xff51afd7ed558ccdull);
        }
    };
    struct Seen {
        bool map = false, gpu_address = false;
    };

    // The argument index of the destination handle, or -1 for a slot that is not a descriptor write.
    static int destination_argument(std::uint16_t slot)
    {
        switch (slot) {
        case 17: case 22: return 1;
        case 18: case 20: case 21: case 70: return 2;
        case 19: return 3;
        default: return -1;
        }
    }

    Key key_of(const LiveCall& call) const
    {
        const auto found = created_.find(call.object);
        return {call.object, found == created_.end() ? 0 : found->second};
    }

    void leave_out(std::size_t index, u64& counter)
    {
        left_out_[index] = true;
        ++counter;
    }

    void add_device(const LiveCall& call, std::size_t index)
    {
        if (call.slot == 23 || call.slot == 24) {
            last_write_.clear();
        } else if (call.slot == 25 || call.slot == 108) {
            leave_out(index, counts_.queries);
        } else if (const int at = destination_argument(call.slot); at >= 0) {
            const u64 destination = call.args[at];
            if (!destination) return;  // no destination handle: nothing to compare
            const auto [it, fresh] = last_write_.try_emplace(destination, index);
            if (!fresh) {
                leave_out(it->second, counts_.superseded_writes);
                it->second = index;
            }
        }
    }

    void add_resource(const LiveCall& call, std::size_t index)
    {
        if (call.slot != 8 && call.slot != 9 && call.slot != 11) return;
        Seen& seen = seen_[key_of(call)];
        if (call.slot == 8) {
            if (seen.map) leave_out(index, counts_.repeated_teaching);
            seen.map = true;
        } else if (call.slot == 9) {
            if (seen.map) leave_out(index, counts_.repeated_teaching);  // an Unmap before any Map stays
        } else {
            if (seen.gpu_address) leave_out(index, counts_.repeated_teaching);
            seen.gpu_address = true;
        }
    }

    std::vector<bool> left_out_;
    PrefixLiveCounts counts_;
    std::unordered_map<u64, std::size_t> last_write_;  // destination handle -> add index of its last write since the last copy
    std::unordered_map<u64, u64> created_;             // pointer -> seq of the last call that output it
    std::unordered_map<Key, Seen, KeyHash> seen_;
};

}  // namespace x2w_replay
