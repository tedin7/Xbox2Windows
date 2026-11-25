// A refused DirectStorage file request leaves its destination untouched, reports a failure status, and the
// requests around it in the same Submit still complete in order (#1586, #1198).
//
// The inputs are generated here: a file of patterned bytes followed by one stored-deflate zlib stream.
// One Submit carries, for the raw read and again for the zlib path, in this order:
//   valid request, request at offset == end of file, request whose source size is implausible
//   (0x98cabbec, the size the issue quotes), valid request.
// Every request is followed by its status entry and by an EnqueueSignal fence. Every destination is poisoned
// before Submit. Checked: the exact bytes of the valid requests; the whole poisoned buffer of each refused
// one; the status value of every entry; that each fence ran after its own status entry and before the next
// one; that the fences ran in queue order; and the stand-in's own log names "read failed" for each refusal.
// Usage: startup_storage_refusal_order XCRDAPI.dll
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <string>
#include <vector>

namespace {

using u64 = std::uint64_t;
constexpr DWORD completion_timeout_ms = 15000;
constexpr unsigned implausible_size = 0x98cabbec;
constexpr unsigned piece = 4096;
constexpr unsigned request_count = 8;
constexpr unsigned char poison = 0xA5;

struct Object { void** vtable; };
using GetFactoryFn = long(__stdcall*)(const void*, void**);
using OpenFileFn = long(__stdcall*)(void*, const wchar_t*, const void*, void**);
using CloseFileFn = void(__stdcall*)(void*);
using CreateStatusArrayFn = long(__stdcall*)(void*, unsigned, const char*, const void*, void**);
using CreateQueueFn = long(__stdcall*)(void*, const void*, const void*, void**);
using EnqueueRequestFn = void(__stdcall*)(void*, const void*);
using EnqueueStatusFn = void(__stdcall*)(void*, void*, unsigned);
using EnqueueSignalFn = void(__stdcall*)(void*, void*, u64);
using SubmitFn = void(__stdcall*)(void*);
using IsCompleteFn = bool(__stdcall*)(void*, unsigned);
using GetHResultFn = long(__stdcall*)(void*, unsigned);

// Notes, when signaled, which status entries were complete and what they held.
struct Fence {
    void** vtable;
    Object* status;
    volatile LONG signaled;
    u64 value;
    bool complete_at_signal[request_count];
    long hr_at_signal[request_count];
    LONG position;
    static volatile LONG order;
    static long __stdcall signal(void* self, u64 value)
    {
        auto* fence = static_cast<Fence*>(self);
        for (unsigned i = 0; i < request_count; ++i) {
            fence->complete_at_signal[i] = reinterpret_cast<IsCompleteFn>(fence->status->vtable[3])(fence->status, i);
            fence->hr_at_signal[i] = reinterpret_cast<GetHResultFn>(fence->status->vtable[4])(fence->status, i);
        }
        fence->value = value;
        fence->position = InterlockedIncrement(&order);
        InterlockedExchange(&fence->signaled, 1);
        return 0;
    }
};
volatile LONG Fence::order = 0;

std::vector<unsigned char> stored_zlib(const std::vector<unsigned char>& data)
{
    std::vector<unsigned char> out{0x78, 0x01, 0x01};  // one final stored block
    const auto size = static_cast<std::uint16_t>(data.size());
    out.push_back(static_cast<unsigned char>(size));
    out.push_back(static_cast<unsigned char>(size >> 8));
    out.push_back(static_cast<unsigned char>(~size));
    out.push_back(static_cast<unsigned char>((~size) >> 8));
    out.insert(out.end(), data.begin(), data.end());
    std::uint32_t a = 1, b = 0;
    for (unsigned char byte : data) {
        a = (a + byte) % 65521;
        b = (b + a) % 65521;
    }
    const std::uint32_t adler = (b << 16) | a;
    for (int shift = 24; shift >= 0; shift -= 8) out.push_back(static_cast<unsigned char>(adler >> shift));
    return out;
}

std::string read_text(const wchar_t* path)
{
    std::string text;
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return text;
    char buffer[4096];
    DWORD got = 0;
    while (ReadFile(file, buffer, sizeof buffer, &got, nullptr) && got) text.append(buffer, got);
    CloseHandle(file);
    return text;
}

bool expect(bool condition, const char* what)
{
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", what);
    return condition;
}

bool all_equal(const std::vector<unsigned char>& bytes, unsigned char value)
{
    for (unsigned char byte : bytes)
        if (byte != value) return false;
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::fprintf(stderr, "usage: startup_storage_refusal_order XCRDAPI.dll\n");
        return 2;
    }
    SetEnvironmentVariableW(L"X2W_XMEM_PERMISSIVE", L"1");
    wchar_t temp_dir[MAX_PATH] = {}, log_path[MAX_PATH] = {}, data_path[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, temp_dir);
    std::swprintf(log_path, MAX_PATH, L"%lsx2w_refusal_order_%lu.log", temp_dir, GetCurrentProcessId());
    std::swprintf(data_path, MAX_PATH, L"%lsx2w_refusal_order_%lu.bin", temp_dir, GetCurrentProcessId());
    DeleteFileW(log_path);
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path);

    // File layout: raw[0, 2 * piece) patterned, then one zlib stream of piece bytes.
    std::vector<unsigned char> raw(2 * piece), plain(piece);
    for (std::size_t i = 0; i < raw.size(); ++i) raw[i] = static_cast<unsigned char>((i * 7 + (i >> 8)) ^ 0x3c);
    for (std::size_t i = 0; i < plain.size(); ++i) plain[i] = static_cast<unsigned char>((i % 241) ^ 0x6b);
    const auto stream = stored_zlib(plain);
    std::vector<unsigned char> file_bytes = raw;
    const u64 zlib_offset = file_bytes.size();
    file_bytes.insert(file_bytes.end(), stream.begin(), stream.end());
    const u64 file_size = file_bytes.size();
    {
        HANDLE out = CreateFileW(data_path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        DWORD written = 0;
        if (out == INVALID_HANDLE_VALUE ||
            !WriteFile(out, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written, nullptr) ||
            written != file_bytes.size()) {
            std::fprintf(stderr, "cannot write the test file\n");
            return 1;
        }
        CloseHandle(out);
    }

    HMODULE library = LoadLibraryA(argv[1]);
    auto* get_factory = library ? reinterpret_cast<GetFactoryFn>(GetProcAddress(library, "DStorageGetFactory")) : nullptr;
    void* object = nullptr;
    if (!get_factory || get_factory(nullptr, &object) != 0 || !object) {
        std::fprintf(stderr, "cannot load the DirectStorage stand-in\n");
        return 1;
    }
    auto* factory = static_cast<Object*>(object);
    void* file_ptr = nullptr;
    if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, data_path, nullptr, &file_ptr) != 0 || !file_ptr) return 1;
    void *status_ptr = nullptr, *queue_ptr = nullptr;
    unsigned char description[0x28] = {};
    *reinterpret_cast<std::uint16_t*>(description + 8) = 16;
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, request_count, "Refusal", nullptr, &status_ptr) != 0 ||
        reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, description, nullptr, &queue_ptr) != 0)
        return 1;
    auto* status = static_cast<Object*>(status_ptr);
    auto* queue = static_cast<Object*>(queue_ptr);
    void* fence_vtable[11] = {};
    fence_vtable[10] = reinterpret_cast<void*>(&Fence::signal);

    struct Plan { bool zlib; u64 offset; unsigned size; bool valid; };
    const Plan plan[request_count] = {
        {false, 0, piece, true},
        {false, file_size, 16, false},                // offset equals the end of the file
        {false, 0, implausible_size, false},          // implausible size at a valid offset
        {false, piece, piece, true},
        {true, zlib_offset, static_cast<unsigned>(stream.size()), true},
        {true, file_size, static_cast<unsigned>(stream.size()), false},
        {true, 0, implausible_size, false},
        {true, zlib_offset, static_cast<unsigned>(stream.size()), true},
    };
    std::vector<unsigned char> destination[request_count];
    Fence fences[request_count] = {};
    for (unsigned i = 0; i < request_count; ++i) {
        destination[i].assign(piece, poison);
        alignas(8) unsigned char request[0x48] = {};
        *reinterpret_cast<u64*>(request + 0x00) = plan[i].zlib ? 1 : 0;
        *reinterpret_cast<unsigned char**>(request + 0x08) = destination[i].data();
        *reinterpret_cast<unsigned*>(request + 0x18) = piece;
        *reinterpret_cast<void**>(request + 0x20) = file_ptr;
        *reinterpret_cast<u64*>(request + 0x28) = plan[i].offset;
        *reinterpret_cast<unsigned*>(request + 0x30) = plan[i].size;
        *reinterpret_cast<unsigned*>(request + 0x38) = piece;
        reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, request);
        reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, i);
        fences[i].vtable = fence_vtable;
        fences[i].status = status;
        reinterpret_cast<EnqueueSignalFn>(queue->vtable[5])(queue, &fences[i], 0x100 + i);
    }
    reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);

    bool ok = true;
    const DWORD start = GetTickCount();
    for (unsigned i = 0; i < request_count; ++i)
        while (!fences[i].signaled && GetTickCount() - start < completion_timeout_ms) Sleep(2);
    for (unsigned i = 0; i < request_count; ++i) ok = expect(fences[i].signaled, "a fence was never signaled") && ok;
    if (!ok) return 1;

    for (unsigned i = 0; i < request_count; ++i) {
        char what[96];
        const bool done = reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, i);
        const long hr = reinterpret_cast<GetHResultFn>(status->vtable[4])(status, i);
        std::snprintf(what, sizeof what, "request %u (%s) status complete", i, plan[i].zlib ? "zlib" : "raw");
        ok = expect(done, what) && ok;
        std::snprintf(what, sizeof what, "request %u status is %s, got 0x%08lx", i,
                      plan[i].valid ? "S_OK" : "ERROR_HANDLE_EOF", static_cast<unsigned long>(hr));
        ok = expect(hr == (plan[i].valid ? S_OK : HRESULT_FROM_WIN32(ERROR_HANDLE_EOF)), what) && ok;
        if (plan[i].valid) {
            const std::vector<unsigned char>& expected = plan[i].zlib ? plain : std::vector<unsigned char>(
                raw.begin() + static_cast<std::ptrdiff_t>(plan[i].offset),
                raw.begin() + static_cast<std::ptrdiff_t>(plan[i].offset) + piece);
            std::snprintf(what, sizeof what, "request %u destination holds the exact bytes", i);
            ok = expect(destination[i] == expected, what) && ok;
        } else {
            std::snprintf(what, sizeof what, "request %u refused but its destination changed", i);
            ok = expect(all_equal(destination[i], poison), what) && ok;
        }
        // The fence behind request i runs after status i and before status i + 1.
        std::snprintf(what, sizeof what, "fence %u ran out of order with its status entries", i);
        bool order = fences[i].value == 0x100u + i && (i == 0 || fences[i].position > fences[i - 1].position);
        for (unsigned k = 0; k < request_count; ++k) {
            order = order && fences[i].complete_at_signal[k] == (k <= i);
            if (k <= i) order = order && fences[i].hr_at_signal[k] == (plan[k].valid ? S_OK : HRESULT_FROM_WIN32(ERROR_HANDLE_EOF));
        }
        ok = expect(order, what) && ok;
    }

    // The stand-in's own account: each refusal is a failed read, each valid request is not.
    const std::string log_text = read_text(log_path);
    auto outcome_of = [&](u64 offset, unsigned size, unsigned options) {
        char key[96];
        std::snprintf(key, sizeof key, "options=0x%x ", options);
        char tail[64];
        std::snprintf(tail, sizeof tail, "offset=0x%llx size=0x%x -> ", static_cast<unsigned long long>(offset), size);
        for (std::size_t at = 0; (at = log_text.find(tail, at)) != std::string::npos; ++at) {
            const std::size_t line = log_text.rfind('\n', at);
            const std::size_t from = line == std::string::npos ? 0 : line + 1;
            const std::string head = log_text.substr(from, at - from);
            if (head.find(key) == std::string::npos) continue;
            const std::size_t end = log_text.find('\r', at);
            return log_text.substr(at + std::strlen(tail), end == std::string::npos ? std::string::npos : end - at - std::strlen(tail));
        }
        return std::string("(not logged)");
    };
    for (unsigned i = 0; i < request_count; ++i) {
        const std::string outcome = outcome_of(plan[i].offset, plan[i].size, plan[i].zlib ? 1u : 0u);
        char what[160];
        std::snprintf(what, sizeof what, "request %u logged outcome \"%s\"", i, outcome.c_str());
        ok = expect(plan[i].valid ? outcome.rfind("read failed", 0) != 0 && outcome != "(not logged)"
                                  : outcome == "read failed", what) && ok;
    }

    reinterpret_cast<CloseFileFn>(static_cast<Object*>(file_ptr)->vtable[3])(file_ptr);
    DeleteFileW(data_path);
    DeleteFileW(log_path);
    if (!ok) return 1;
    std::puts("refused requests (offset == EOF, implausible size; raw and zlib) left their destinations untouched; "
              "valid requests before and after completed with exact bytes; statuses and fences in queue order");
    return 0;
}
