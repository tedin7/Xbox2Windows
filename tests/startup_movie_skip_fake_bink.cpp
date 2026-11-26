// Stand-in for the title's movie player DLL (built as bink2scarlett.dll) for startup_movie_skip. It exports the eight functions
// the title imports by name that the X2W_SKIP_MOVIES hook rewrites, plus fake_bink_stats, which counts what reached it.
// The movie is eight frames; the header is width, height, frames, frame number (32-bit each), frame numbers from 1.
#include <windows.h>

#include <cstring>

namespace {
struct Movie {
    unsigned width, height, frames, frame;
};
struct Stats {
    unsigned opens, closes, decoded, nexts, gotos, last_goto, waits, skips, pauses;
};
Movie movie{64, 36, 8, 1};
Stats stats{};
}  // namespace

extern "C" {
__declspec(dllexport) void* BinkOpenWithOptions(const char*, const void*, unsigned)
{
    ++stats.opens;
    movie = Movie{64, 36, 8, 1};
    return &movie;
}
__declspec(dllexport) int BinkClose(void*)
{
    ++stats.closes;
    return 0;
}
__declspec(dllexport) int BinkNextFrame(void*)
{
    ++stats.nexts;
    if (movie.frame < movie.frames) ++movie.frame;
    return 0;
}
__declspec(dllexport) int BinkGoto(void*, unsigned frame, int)
{
    ++stats.gotos;
    stats.last_goto = frame;
    movie.frame = frame;
    return 0;
}
__declspec(dllexport) int BinkDoFrame(void*)
{
    ++stats.decoded;
    return 0;
}
__declspec(dllexport) int BinkWait(void*)
{
    ++stats.waits;
    return 0;
}
__declspec(dllexport) int BinkShouldSkip(void*)
{
    ++stats.skips;
    return 0;
}
__declspec(dllexport) int BinkPause(void*, int)
{
    ++stats.pauses;
    return 0;
}
__declspec(dllexport) void fake_bink_stats(unsigned* out)
{
    std::memcpy(out, &stats, sizeof(stats));
}
}
