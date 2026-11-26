// Experimental, opt-in (docs/startup-runtime.md): end the title's Bink movies at once.
// Included inside runtime_proxy.cpp's anonymous namespace, and on its own by a Windows fixture. Needs <windows.h>,
// <atomic>, <cstring>, local_log.h and runtime_gpa_patch.h.
//
// The title imports its movie player from bink2scarlett.dll by name. With X2W_SKIP_MOVIES set, the stand-in rewrites those
// regular IAT slots of the EXE and package helpers (as the registry imports are) and forwards every call to the real function,
// except what the chosen mode changes. Values of X2W_SKIP_MOVIES:
//   log        forward everything; log the first calls of each import and the movie header (width, height, frames, frame).
//   goto_end   after the first BinkNextFrame of a playback (frame number 1), jump with the real BinkGoto to the last frame, so
//              the player's own end-of-movie test sees the end on its next check. The decoder state stays consistent.
//   open_null  BinkOpenWithOptions returns null: the title sees a movie that could not be opened.
//   frames1    after a successful open the header's frame count is set to 1.
//   1          same as goto_end.
// Nothing else is changed. A mode the title does not survive is found by running the modes one after the other on a real
// title: that is the test plan in pilot/skip_movies_plan.md; none of this is on by default.

enum MovieImport {
    movie_open, movie_close, movie_next_frame, movie_goto, movie_do_frame, movie_wait, movie_should_skip, movie_pause, movie_imports
};
constexpr const char* movie_import_names[movie_imports] = {"BinkOpenWithOptions", "BinkClose", "BinkNextFrame", "BinkGoto",
                                                           "BinkDoFrame", "BinkWait", "BinkShouldSkip", "BinkPause"};
enum class MovieMode { off, log, goto_end, open_null, frames1 };

// Every import takes at most four integer or pointer arguments and returns one in rax (the Windows x64 convention), so one
// signature forwards them all.
using MovieFn = ULONGLONG(WINAPI*)(ULONGLONG, ULONGLONG, ULONGLONG, ULONGLONG);

std::atomic<void*> real_movie[movie_imports];
LONG movie_calls[movie_imports];

MovieMode movie_mode() noexcept
{
    static const MovieMode mode = [] {
        char value[32]{};
        const DWORD n = GetEnvironmentVariableA("X2W_SKIP_MOVIES", value, sizeof(value));
        if (!n || n >= sizeof(value)) return MovieMode::off;
        if (!std::strcmp(value, "log")) return MovieMode::log;
        if (!std::strcmp(value, "open_null")) return MovieMode::open_null;
        if (!std::strcmp(value, "frames1")) return MovieMode::frames1;
        if (!std::strcmp(value, "0")) return MovieMode::off;
        return MovieMode::goto_end;  // "1", "goto_end"
    }();
    return mode;
}

int movie_import_index(const char* name) noexcept
{
    if (!name) return -1;
    for (int i = 0; i < movie_imports; ++i)
        if (std::strcmp(name, movie_import_names[i]) == 0) return i;
    return -1;
}

bool movie_provider_dll(const char* dll) noexcept
{
    if (!dll) return false;
    static constexpr char expect[] = "bink2scarlett.dll";
    for (std::size_t i = 0; expect[i]; ++i) {
        const char c = dll[i];
        if (((c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c) != expect[i]) return false;
    }
    return dll[sizeof(expect) - 1] == 0;
}

// The movie header the title reads: width, height, frames, frame number (32-bit each), as the debugger report decodes it.
struct MovieHeader { UINT width, height, frames, frame; };
bool movie_header(ULONGLONG handle, MovieHeader* out) noexcept
{
    if (!handle || !x2w_local::readable(reinterpret_cast<const void*>(handle), sizeof(MovieHeader))) return false;
    std::memcpy(out, reinterpret_cast<const void*>(handle), sizeof(MovieHeader));
    return true;
}

bool movie_log_this(int which) noexcept { return InterlockedIncrement(&movie_calls[which]) <= 8; }

ULONGLONG movie_forward(int which, ULONGLONG a, ULONGLONG b, ULONGLONG c, ULONGLONG d) noexcept
{
    auto* const real = reinterpret_cast<MovieFn>(real_movie[which].load(std::memory_order_acquire));
    return real ? real(a, b, c, d) : 0;
}

void movie_log_call(int which, ULONGLONG a, ULONGLONG b, ULONGLONG result) noexcept
{
    MovieHeader header{};
    const bool have = (which == movie_open ? movie_header(result, &header) : movie_header(a, &header));
    x2w_local::logf("x2w-movie %s(0x%llx, 0x%llx) -> 0x%llx header %s %u x %u, frames %u, frame %u", movie_import_names[which], a, b, result,
                    have ? "" : "unreadable", header.width, header.height, header.frames, header.frame);
}

ULONGLONG WINAPI movie_hook_open(ULONGLONG a, ULONGLONG b, ULONGLONG c, ULONGLONG d)
{
    const MovieMode mode = movie_mode();
    if (mode == MovieMode::open_null) {
        if (movie_log_this(movie_open)) x2w_local::logf("x2w-movie BinkOpenWithOptions refused (open_null)");
        return 0;
    }
    const ULONGLONG result = movie_forward(movie_open, a, b, c, d);
    if (mode == MovieMode::frames1 && result && x2w_local::writable(reinterpret_cast<void*>(result + 8), sizeof(UINT)))
        *reinterpret_cast<UINT*>(result + 8) = 1;
    if (movie_log_this(movie_open) || mode == MovieMode::log) movie_log_call(movie_open, a, b, result);
    return result;
}

ULONGLONG WINAPI movie_hook_next_frame(ULONGLONG a, ULONGLONG b, ULONGLONG c, ULONGLONG d)
{
    const ULONGLONG result = movie_forward(movie_next_frame, a, b, c, d);
    MovieHeader header{};
    if (movie_mode() == MovieMode::goto_end && movie_header(a, &header) && header.frame <= 1 && header.frames > 1 &&
        header.frames < (1u << 24)) {
        // The playback just started: go to the last frame with the real BinkGoto (frame, flags 0).
        movie_forward(movie_goto, a, header.frames, 0, 0);
        if (movie_log_this(movie_goto)) x2w_local::logf("x2w-movie playback ended at once: handle 0x%llx, %u frames", a, header.frames);
    } else if (movie_mode() == MovieMode::log && movie_log_this(movie_next_frame)) {
        movie_log_call(movie_next_frame, a, b, result);
    }
    return result;
}

ULONGLONG WINAPI movie_hook_other(int which, ULONGLONG a, ULONGLONG b, ULONGLONG c, ULONGLONG d)
{
    const ULONGLONG result = movie_forward(which, a, b, c, d);
    if (movie_mode() == MovieMode::log && movie_log_this(which)) movie_log_call(which, a, b, result);
    return result;
}
ULONGLONG WINAPI movie_hook_close(ULONGLONG a, ULONGLONG b, ULONGLONG c, ULONGLONG d) { return movie_hook_other(movie_close, a, b, c, d); }
ULONGLONG WINAPI movie_hook_goto(ULONGLONG a, ULONGLONG b, ULONGLONG c, ULONGLONG d) { return movie_hook_other(movie_goto, a, b, c, d); }
ULONGLONG WINAPI movie_hook_do_frame(ULONGLONG a, ULONGLONG b, ULONGLONG c, ULONGLONG d) { return movie_hook_other(movie_do_frame, a, b, c, d); }
ULONGLONG WINAPI movie_hook_wait(ULONGLONG a, ULONGLONG b, ULONGLONG c, ULONGLONG d) { return movie_hook_other(movie_wait, a, b, c, d); }
ULONGLONG WINAPI movie_hook_should_skip(ULONGLONG a, ULONGLONG b, ULONGLONG c, ULONGLONG d) { return movie_hook_other(movie_should_skip, a, b, c, d); }
ULONGLONG WINAPI movie_hook_pause(ULONGLONG a, ULONGLONG b, ULONGLONG c, ULONGLONG d) { return movie_hook_other(movie_pause, a, b, c, d); }

const MovieFn movie_hooks[movie_imports] = {movie_hook_open,      movie_hook_close,       movie_hook_next_frame, movie_hook_goto,
                                            movie_hook_do_frame,  movie_hook_wait,        movie_hook_should_skip, movie_hook_pause};

// Rewrites the regular Bink IAT slots of one module. Returns how many slots were rewritten. Does nothing unless
// X2W_SKIP_MOVIES selects a mode.
int patch_movie_imports(unsigned char* base, std::size_t image_size) noexcept
{
    if (movie_mode() == MovieMode::off) return 0;
    int patched = 0;
    const auto classify = [](const char* dll, const char* name, x2w_runtime_patch::Kind* kind) {
        if (movie_import_index(name) < 0 || !movie_provider_dll(dll)) return false;
        *kind = x2w_runtime_patch::Kind::movie;
        return true;
    };
    x2w_runtime_patch::visit_imports_with(base, image_size, classify, [&](const x2w_runtime_patch::Hit& hit) {
        if (hit.delay_import) return;
        const int which = movie_import_index(hit.import_name);
        auto* slot = reinterpret_cast<ULONGLONG*>(base + hit.first_thunk_rva);
        if (*slot == reinterpret_cast<ULONGLONG>(movie_hooks[which])) return;
        void* expected = nullptr;
        // The first module to be patched supplies the real function; every later slot must already name the same one.
        if (!real_movie[which].compare_exchange_strong(expected, reinterpret_cast<void*>(*slot)) &&
            expected != reinterpret_cast<void*>(*slot))
            return;
        DWORD old = 0;
        if (!VirtualProtect(slot, 8, PAGE_READWRITE, &old)) return;
        *slot = reinterpret_cast<ULONGLONG>(movie_hooks[which]);
        VirtualProtect(slot, 8, old, &old);
        ++patched;
    });
    return patched;
}
