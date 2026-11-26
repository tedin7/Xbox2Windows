// Unit test of src/startup_stubs/texture_residency.h (pure logic, no GPU). Each case guards one rule; the
// "mutant" argument flips one rule off and the matching case must then fail (exit 1), proving the case bites.
#include "startup_stubs/texture_residency.h"

#include <cstdio>
#include <cstring>

using namespace x2w_residency;

namespace {
int failures = 0;
void expect(bool ok, const char* what)
{
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}
constexpr std::uint64_t MiB = 1u << 20;

Texture tex(std::uint64_t id, std::uint64_t mib, std::int64_t last_use)
{
    Texture t;
    t.id = id; t.bytes = mib * MiB; t.last_use = last_use;
    return t;
}
}  // namespace

int main(int argc, char** argv)
{
    const char* mutant = argc > 1 ? argv[1] : "";
    const Config c;
    const std::uint64_t budget = 1000 * MiB;

    // 1. Hysteresis: starts above 95%, stays on until 85% or less.
    {
        Pressure p;
        expect(!p.update(940 * MiB, budget, c), "94% does not start pressure");
        expect(p.update(960 * MiB, budget, c), "96% starts pressure");
        expect(p.update(900 * MiB, budget, c), "90% keeps pressure (between the marks)");
        expect(!p.update(850 * MiB, budget, c), "85% ends pressure");
        expect(!p.update(0, 0, c), "no budget means no pressure");
    }
    // 2. No pressure, no eviction, even for old textures.
    {
        std::vector<Texture> v{tex(1, 100, 0)};
        expect(plan_evictions(v, 500 * MiB, budget, 1000, c, false).empty(), "no eviction without pressure");
        expect(plan_evictions(v, 800 * MiB, budget, 1000, c, true).empty(), "no eviction when already under the low mark");
    }
    // 3. Oldest first, only as many as needed (to_free = 960 - 850 = 110 MiB).
    {
        std::vector<Texture> v{tex(1, 60, 500), tex(2, 60, 100), tex(3, 60, 300), tex(4, 60, 900)};
        const auto out = plan_evictions(v, 960 * MiB, budget, 1000, c, true);
        expect(out.size() == 2 && out[0] == 2 && out[1] == 3, "oldest first, stops once 110 MiB are covered (id 4 is too recent)");
    }
    // 4. Recently used (younger than min_idle) is never taken.
    {
        std::vector<Texture> v{tex(1, 500, 950)};
        expect(plan_evictions(v, 990 * MiB, budget, 1000, c, true).empty(), "used 50 frames ago stays");
    }
    // 5. In flight is never taken (fence rule). The mutant drops the rule.
    {
        std::vector<Texture> v{tex(1, 200, 0)};
        v[0].in_flight = true;
        if (!std::strcmp(mutant, "ignore-in-flight")) v[0].in_flight = false;
        expect(plan_evictions(v, 990 * MiB, budget, 1000, c, true).empty(), "a texture with an uncompleted submission is never evicted");
    }
    // 6. Unsafe (gpu written, title residency calls, materialize in progress) and pinned are never taken.
    {
        std::vector<Texture> v{tex(1, 200, 0), tex(2, 200, 0)};
        v[0].unsafe = true;
        v[1].pinned = true;
        if (!std::strcmp(mutant, "ignore-unsafe")) v[0].unsafe = false;
        expect(plan_evictions(v, 990 * MiB, budget, 1000, c, true).empty(), "unsafe and pinned textures are never evicted");
    }
    // 7. Not resident textures are not planned twice.
    {
        std::vector<Texture> v{tex(1, 200, 0)};
        v[0].resident = false;
        expect(plan_evictions(v, 990 * MiB, budget, 1000, c, true).empty(), "an evicted texture is not evicted again");
    }
    // 8. Use of an evicted texture asks for MakeResident once; a quick return pins it, a late one does not.
    {
        Texture t = tex(1, 10, 0);
        note_evicted(t, 1000);
        expect(note_use(t, 1100, c), "use after eviction asks for MakeResident");
        expect(t.resident && t.last_use == 1100, "the texture is resident and stamped");
        expect(t.pinned, "back within pin_frames: pinned");
        expect(!note_use(t, 1101, c), "a resident texture asks for nothing");
        Texture late = tex(2, 10, 0);
        note_evicted(late, 1000);
        note_use(late, 1000 + c.pin_frames + 1, c);
        expect(!late.pinned, "back after pin_frames: not pinned");
        if (!std::strcmp(mutant, "no-pin")) t.pinned = false;
        expect(t.pinned, "thrash guard keeps the quick-return texture");
    }
    std::printf(failures ? "startup_texture_residency: %d failure(s)\n" : "startup_texture_residency: ok\n", failures);
    return failures ? 1 : 0;
}
