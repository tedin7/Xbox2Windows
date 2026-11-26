#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>

namespace x2w_local {

// What a translated draw had bound, for the gray-world question: which pipeline key wrote the presented surface, and were its inputs
// real? One DrawProfile per draw in flight (filled while the draw binds), one KeyProfile per pipeline key (the last draw of the key).
struct DrawProfile {
    unsigned texture_real = 0, texture_default = 0, texture_null = 0;  // default = a view of another dimension than the shader asked for
    unsigned buffer_real = 0, buffer_null = 0;
    unsigned small_checked = 0, small_zero = 0;  // bindings of 64 KiB or less (constants) whose first 256 bytes were looked at / all zero
};

struct Counters {  // cumulative totals of the runtime at the moment of a draw
    long untranslated = 0, not_drawn = 0, null_bindings = 0;
};

struct KeyProfile {
    std::uint64_t draws = 0, draws_with_zero_constants = 0;
    DrawProfile last;
    Counters before;  // totals at the key's most recent draw
};

class KeyProfiles {
public:
    void begin_draw() { current_ = DrawProfile{}; }
    void texture(bool bound, bool default_view)
    {
        if (!bound) ++current_.texture_null;
        else if (default_view) ++current_.texture_default;
        else ++current_.texture_real;
    }
    void buffer(bool bound) { bound ? ++current_.buffer_real : ++current_.buffer_null; }
    // `first_bytes` is the start of a small binding; up to 256 bytes are looked at.
    void small_binding(const void* data, std::size_t bytes)
    {
        const std::size_t look = bytes < 256 ? bytes : 256;
        if (!look) return;
        ++current_.small_checked;
        const auto* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < look; ++i)
            if (p[i]) return;
        ++current_.small_zero;
    }
    void end_draw(std::uint64_t key, const Counters& totals)
    {
        Stored& profile = keys_[key];
        ++profile.draws;
        if (current_.small_zero) ++profile.draws_with_zero_constants;
        profile.last = current_;
        profile.before = totals;
    }
    // One line for `key` the first time it is asked for; false afterwards (and for a key that was never drawn).
    bool detail_once(std::uint64_t key, char* out, std::size_t size)
    {
        const auto found = keys_.find(key);
        if (found == keys_.end() || found->second.reported) return false;
        found->second.reported = true;
        const KeyProfile& k = found->second;
        std::snprintf(out, size,
                      "x2w-render present source detail key %016llx: %llu draws; last draw: textures real %u / default view %u / null %u, buffers real %u / null %u, "
                      "constants looked at %u, all zero %u (%llu of its draws had an all-zero small binding); before it, cumulative: untranslated %ld, not drawn %ld, null bindings %ld",
                      static_cast<unsigned long long>(key), static_cast<unsigned long long>(k.draws), k.last.texture_real, k.last.texture_default,
                      k.last.texture_null, k.last.buffer_real, k.last.buffer_null, k.last.small_checked, k.last.small_zero,
                      static_cast<unsigned long long>(k.draws_with_zero_constants), k.before.untranslated, k.before.not_drawn, k.before.null_bindings);
        return true;
    }

private:
    struct Stored : KeyProfile {
        bool reported = false;
    };
    DrawProfile current_;
    std::unordered_map<std::uint64_t, Stored> keys_;
};

}  // namespace x2w_local
