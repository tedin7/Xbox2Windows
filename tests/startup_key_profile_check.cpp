// src/startup_stubs/key_profile.h: the per-key draw profile behind the `x2w-render present source detail` line. A key's last draw
// reports its bound inputs (real / default view / null textures, real / null buffers, all-zero constants) and the runtime totals
// before it, once per key.
// --mutate no-zero-check: constants are never looked at; the all-zero count must fail.
#include "startup_stubs/key_profile.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {
int failures;
void check(const char* name, bool ok)
{
    if (!ok) {
        std::printf("FAIL %s\n", name);
        ++failures;
    }
}
}  // namespace

int main(int argc, char** argv)
{
    const bool skip_zero = argc == 3 && !std::strcmp(argv[1], "--mutate") && !std::strcmp(argv[2], "no-zero-check");
    x2w_local::KeyProfiles profiles;
    char line[600];
    check("a key never drawn has no detail", !profiles.detail_once(7, line, sizeof(line)));
    unsigned char zeros[64] = {}, live[64] = {1};
    // A gray-world draw: two default views, a null texture, a real texture, one real and one null buffer, constants all zero.
    profiles.begin_draw();
    profiles.texture(true, true);
    profiles.texture(true, true);
    profiles.texture(false, false);
    profiles.texture(true, false);
    profiles.buffer(true);
    profiles.buffer(false);
    if (!skip_zero) profiles.small_binding(zeros, sizeof(zeros));
    profiles.end_draw(7, {100, 20, 5});
    // A later draw of another key must not disturb key 7.
    profiles.begin_draw();
    profiles.small_binding(live, sizeof(live));
    profiles.end_draw(9, {1, 1, 1});
    check("key 7 has a detail line", profiles.detail_once(7, line, sizeof(line)));
    const std::string text = line;
    check("names the key", text.find("key 0000000000000007") != std::string::npos);
    check("textures real 1 / default 2 / null 1", text.find("textures real 1 / default view 2 / null 1") != std::string::npos);
    check("buffers real 1 / null 1", text.find("buffers real 1 / null 1") != std::string::npos);
    check("constants all zero 1 of 1 looked at", text.find("constants looked at 1, all zero 1") != std::string::npos);
    check("totals before it", text.find("untranslated 100, not drawn 20, null bindings 5") != std::string::npos);
    check("once per key", !profiles.detail_once(7, line, sizeof(line)));
    check("a live constant buffer is not counted as zero", profiles.detail_once(9, line, sizeof(line)) && std::string(line).find("all zero 0") != std::string::npos);
    if (failures) return 1;
    std::printf("startup_key_profile: PASS\n");
    return 0;
}
