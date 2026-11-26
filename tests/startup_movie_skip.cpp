// Source-built control for src/startup_stubs/movie_skip.inl: the opt-in X2W_SKIP_MOVIES rewrite of the title's Bink imports.
//
// This EXE imports eight functions by name from bink2scarlett.dll, which is a stand-in DLL (startup_movie_skip_fake_bink) with an
// eight-frame movie, and plays it the way a title does: open, then wait / decode / test the header for the end / next frame.
// It patches its own import table with patch_movie_imports, as runtime_proxy.cpp does for the title's EXE.
//   plain      switch unset: nothing is rewritten, all eight frames are shown, no jump.
//   goto_end   X2W_SKIP_MOVIES=1: eight slots rewritten, at most two frames shown, the real BinkGoto saw (movie, last frame).
//   loop       as goto_end, with a title that restarts the movie three times (BinkGoto to frame 1): each playback ends at once.
//   mid        X2W_SKIP_MOVIES=1, the title seeks to frame 4 before its first next-frame: that is not a playback start, so the
//              frames 4 to 8 are all shown and the only real BinkGoto is the title's own.
//   frames1    X2W_SKIP_MOVIES=frames1: one frame is shown.
//   open_null  X2W_SKIP_MOVIES=open_null: the open fails and the real open is never reached.
//   log        X2W_SKIP_MOVIES=log: all eight frames are shown, the calls are logged, no jump.
// --no-hook skips the patch (the hook not installed). With a switch that must end the movie, that must print FAIL.
#include <windows.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "startup_stubs/local_log.h"
#include "startup_stubs/runtime_gpa_patch.h"

namespace x2w_local {
std::string fixture_log;
void fixture_logf(const char* format, ...)
{
    char line[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    fixture_log += line;
    fixture_log += '\n';
}
}  // namespace x2w_local

#define logf fixture_logf
namespace {
#include "../src/startup_stubs/movie_skip.inl"
}
#undef logf

extern "C" {
__declspec(dllimport) void* BinkOpenWithOptions(const char*, const void*, unsigned);
__declspec(dllimport) int BinkClose(void*);
__declspec(dllimport) int BinkNextFrame(void*);
__declspec(dllimport) int BinkGoto(void*, unsigned, int);
__declspec(dllimport) int BinkDoFrame(void*);
__declspec(dllimport) int BinkWait(void*);
__declspec(dllimport) int BinkShouldSkip(void*);
__declspec(dllimport) int BinkPause(void*, int);
__declspec(dllimport) void fake_bink_stats(unsigned* out);
}

namespace {
struct Stats {
    unsigned opens, closes, decoded, nexts, gotos, last_goto, waits, skips, pauses;
};
struct Header {
    unsigned width, height, frames, frame;
};

int failures = 0;
void check(const char* what, bool ok, long value = 0)
{
    if (ok) return;
    ++failures;
    std::printf("FAIL: %s (value %ld)\n", what, value);
}

// One playback the way a title runs it: wait, decode, test the header for the last frame, next frame. Bounded.
int play(void* movie)
{
    int shown = 0;
    for (;;) {
        BinkWait(movie);
        if (BinkShouldSkip(movie)) continue;
        BinkDoFrame(movie);
        ++shown;
        Header header;
        std::memcpy(&header, movie, sizeof(header));
        if (header.frame >= header.frames || shown >= 64) break;
        BinkNextFrame(movie);
    }
    return shown;
}
}  // namespace

int main(int argc, char** argv)
{
    const std::string which = argc > 1 ? argv[1] : "plain";
    bool hook = true;
    for (int i = 2; i < argc; ++i)
        if (!std::strcmp(argv[i], "--no-hook")) hook = false;

    const HMODULE self = GetModuleHandleW(nullptr);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(self);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(reinterpret_cast<const unsigned char*>(self) + dos->e_lfanew);
    const int patched = hook ? patch_movie_imports(reinterpret_cast<unsigned char*>(self), nt->OptionalHeader.SizeOfImage) : 0;

    // BinkPause is never called, so the linker would drop its import: take its address to keep all eight slots.
    static int (*volatile keep_pause_import)(void*, int) = BinkPause;
    (void)keep_pause_import;
    void* movie = BinkOpenWithOptions("movie.bk2", nullptr, 0);
    int shown = 0, playbacks = 0;
    if (movie) {
        if (which == "loop") {
            for (; playbacks < 3; ++playbacks) {
                shown += play(movie);
                BinkGoto(movie, 1, 0);  // the title's own restart
            }
        } else {
            if (which == "mid") BinkGoto(movie, 4, 0);
            shown = play(movie);
            playbacks = 1;
        }
        BinkClose(movie);
    }
    Stats stats{};
    fake_bink_stats(reinterpret_cast<unsigned*>(&stats));
    std::printf("movie: case %s, slots rewritten %d, opened %d, playbacks %d, frames shown %d, real BinkGoto %u (last %u), real opens %u\n",
                which.c_str(), patched, movie ? 1 : 0, playbacks, shown, stats.gotos, stats.last_goto, stats.opens);

    if (which == "plain") {
        check("nothing is rewritten without the switch", patched == 0, patched);
        check("all eight frames are shown", shown == 8, shown);
        check("no jump", stats.gotos == 0, stats.gotos);
    } else if (which == "goto_end") {
        check("the eight imports are rewritten", patched == movie_imports, patched);
        check("at most two frames are shown", movie && shown >= 1 && shown <= 2, shown);
        check("the real BinkGoto went to the last frame", stats.gotos == 1 && stats.last_goto == 8, stats.gotos);
    } else if (which == "loop") {
        check("the eight imports are rewritten", patched == movie_imports, patched);
        check("three playbacks of at most two frames", playbacks == 3 && shown >= 3 && shown <= 6, shown);
        check("each playback jumped once and each restart reached the real function", stats.gotos == 6, stats.gotos);
    } else if (which == "mid") {
        check("the eight imports are rewritten", patched == movie_imports, patched);
        check("frames 4 to 8 are shown", shown == 5, shown);
        check("only the title's own jump reached the real function", stats.gotos == 1 && stats.last_goto == 4, stats.gotos);
    } else if (which == "frames1") {
        check("the eight imports are rewritten", patched == movie_imports, patched);
        check("one frame is shown", shown == 1, shown);
        check("no jump", stats.gotos == 0, stats.gotos);
    } else if (which == "open_null") {
        check("the eight imports are rewritten", patched == movie_imports, patched);
        check("the open fails", movie == nullptr);
        check("the real open is never reached", stats.opens == 0, stats.opens);
    } else if (which == "log") {
        check("the eight imports are rewritten", patched == movie_imports, patched);
        check("all eight frames are shown", shown == 8, shown);
        check("no jump", stats.gotos == 0, stats.gotos);
        check("the open is logged", x2w_local::fixture_log.find("x2w-movie BinkOpenWithOptions") != std::string::npos);
    } else {
        std::printf("usage: startup_movie_skip <plain|goto_end|loop|mid|frames1|open_null|log> [--no-hook]\n");
        return 2;
    }
    std::printf("startup_movie_skip: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
