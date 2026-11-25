// Input for the startup debugger's diagnostics check (tests/startup_debug_check.py). It reaches two
// breakpoints in its own image, the way a retail image keeps soft asserts, reserves one large
// range, reports whether the loader gave it the debug heap, stays alive for a moment and exits 0.
// With STARTUP_DEBUG_PROBE_DLL=<file> and STARTUP_DEBUG_PROBE_EXPORT=<name> it also loads that DLL
// and calls the export kCalls times (the call-hook case; tests/startup_debug_calls.cpp).
// No GPU and no other input is involved.
#include <windows.h>
#include <intrin.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <thread>
#include <vector>

namespace {
volatile int reached;

__declspec(noinline) void first_assert()
{
    reached = 1;
    __debugbreak();
}

__declspec(noinline) void second_assert()
{
    reached = 2;
    __debugbreak();
}

__declspec(noinline) void* reserve_large()
{
    // 256 MiB, reserved only: nothing is committed.
    return VirtualAlloc(nullptr, SIZE_T{256} << 20, MEM_RESERVE, PAGE_NOACCESS);
}

// The heap bits of the PEB's NtGlobalFlag (x64 offset 0xBC): tail check, free check and parameter
// validation (0x70). The loader sets them for a process created under a debugger unless the
// environment has _NO_DEBUG_HEAP=1.
unsigned long debug_heap_flags()
{
    const auto* const peb = reinterpret_cast<const unsigned char*>(__readgsqword(0x60));
    return *reinterpret_cast<const unsigned long*>(peb + 0xBC) & 0x70;
}

constexpr int kCalls = 7;
constexpr int kLogCalls = 19;
constexpr std::uint64_t kLogToken = 0x1122334455660000;

__declspec(noinline) std::uint64_t log_site(const std::uint64_t* words, std::uint64_t token, std::uint64_t marker)
{
    return words[0] == 0x1122334455667788 && token >= kLogToken ? marker : 0;
}

void call_log_site()
{
    const std::uint64_t words[] = {0x1020304050607080, 0x1122334455667788, 0x89abcdef01234567};
    auto* volatile call = log_site;
    const auto address = reinterpret_cast<std::uintptr_t>(call);
    const auto image = reinterpret_cast<std::uintptr_t>(GetModuleHandleW(nullptr));
    std::uint64_t total = 0;
    for (int i = 0; i < kLogCalls; ++i) {
        total += call(words + 1, kLogToken + i, 1);
    }
    // A breakpoint left armed after the cap would still have an INT3 byte here.
    const auto opcode = *reinterpret_cast<const volatile unsigned char*>(address);
    std::printf("debug-probe log site 0x%llx byte 0x%x calls %llu\n",
                static_cast<unsigned long long>(address - image), static_cast<unsigned int>(opcode),
                static_cast<unsigned long long>(total));
}

constexpr int kThreads = 8;
constexpr int kThreadCalls = 2000;

__declspec(noinline) std::uint64_t threaded_log_site(std::uint64_t value)
{
    return value & 1;
}

// Eight threads released together call one site, so INT3 events from several threads can be
// queued when a hit limit removes its breakpoint.
void call_threaded_log_site()
{
    auto* volatile call = threaded_log_site;
    const auto address = reinterpret_cast<std::uintptr_t>(call);
    const auto image = reinterpret_cast<std::uintptr_t>(GetModuleHandleW(nullptr));
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::atomic<std::uint64_t> total{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&] {
            ++ready;
            while (!go.load()) YieldProcessor();
            std::uint64_t mine = 0;
            for (int i = 0; i < kThreadCalls; ++i) mine += call(1);
            total += mine;
        });
    }
    while (ready.load() != kThreads) YieldProcessor();
    go = true;
    for (auto& thread : threads) thread.join();
    const auto opcode = *reinterpret_cast<const volatile unsigned char*>(address);
    std::printf("debug-probe threaded log site 0x%llx byte 0x%x calls %llu\n",
                static_cast<unsigned long long>(address - image), static_cast<unsigned int>(opcode),
                static_cast<unsigned long long>(total.load()));
}

// Returns the number of export calls made, or -1 when the DLL or the export is missing.
int call_export(const char* dll, const char* name)
{
    HMODULE const module = LoadLibraryA(dll);
    using Export = LONG (*)(void*);
    auto const function = module ? reinterpret_cast<Export>(GetProcAddress(module, name)) : nullptr;
    if (!function) {
        return -1;
    }
    int made = 0;
    for (int i = 0; i < kCalls; ++i) {
        made += function(nullptr) > 0 ? 1 : 0;
    }
    return made;
}
}  // namespace

int main()
{
    first_assert();
    second_assert();
    void* const range = reserve_large();
    std::printf("debug-probe reserved %s\n", range ? "ok" : "failed");
    std::printf("debug-probe heap flags 0x%lx\n", debug_heap_flags());
    call_log_site();
    call_threaded_log_site();
    const char* const dll = std::getenv("STARTUP_DEBUG_PROBE_DLL");
    const char* const name = std::getenv("STARTUP_DEBUG_PROBE_EXPORT");
    if (dll && name) {
        std::printf("debug-probe calls %d\n", call_export(dll, name));
    }
    std::fflush(stdout);
    Sleep(3000);
    return 0;
}
