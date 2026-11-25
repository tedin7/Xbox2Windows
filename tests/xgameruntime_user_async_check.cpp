// Synthetic title-level check for XUserAddAsync and XUserAddResult in the
// fixture-only xgameruntime.dll, on the shim-owned manual task queue (#2179,
// #2170). The DLL is loaded from its explicit fixture path. The block mirrors
// the three leading public XAsyncBlock fields and pads the rest with a sentinel
// pattern, which the shim must leave untouched because it keeps its own state.
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace {
namespace fs = std::filesystem;

constexpr HRESULT not_implemented = static_cast<HRESULT>(0x80004001L);
constexpr HRESULT e_illegal_state_change = static_cast<HRESULT>(0x8000000DL);
constexpr std::uint64_t synthetic_user_id = 0x0009000000000001ull;
constexpr std::uint8_t sentinel = 0xA5;
constexpr std::uint32_t opt_none = 0;
constexpr std::uint32_t opt_silently = 1;
constexpr std::uint32_t opt_allow_guests = 2;
constexpr std::uint32_t opt_allowing_ui = 4;

namespace xgameruntime_shim {
struct XUserHandleStorage;
struct XGameSaveProviderStorage;
}
using XUserHandle = xgameruntime_shim::XUserHandleStorage*;
using XGameSaveProviderHandle = xgameruntime_shim::XGameSaveProviderStorage*;

struct XAsyncBlock;
using CompletionRoutine = void CALLBACK(XAsyncBlock*);
struct XAsyncBlock {
    void* queue;
    void* context;
    CompletionRoutine* callback;
    std::uint8_t opaque[sizeof(void*) * 4];
};

struct XTaskQueueHandleStorage;
using XTaskQueueHandle = XTaskQueueHandleStorage*;
constexpr std::uint32_t mode_manual = 0;
constexpr std::uint32_t mode_thread_pool = 1;
constexpr std::uint32_t port_completion = 1;

using UserAddAsync = HRESULT(WINAPI*)(std::uint32_t, XAsyncBlock*);
using UserAddResult = HRESULT(WINAPI*)(XAsyncBlock*, XUserHandle*);
using UserGetId = HRESULT(WINAPI*)(XUserHandle, std::uint64_t*);
using CloseUserHandle = void(WINAPI*)(XUserHandle);
using UserFindById = HRESULT(WINAPI*)(std::uint64_t, XUserHandle*);
using ProviderStart = HRESULT(WINAPI*)(XUserHandle, const char*, bool, XAsyncBlock*);
using CloseProvider = void(WINAPI*)(XGameSaveProviderHandle);
using ProviderResult = HRESULT(WINAPI*)(XAsyncBlock*, XGameSaveProviderHandle*);
using GetStatus = HRESULT(WINAPI*)(XAsyncBlock*, bool);
using GetResultSize = HRESULT(WINAPI*)(XAsyncBlock*, std::size_t*);
using QueueCreate =HRESULT(WINAPI*)(std::uint32_t, std::uint32_t, XTaskQueueHandle*);
using QueueDispatch = bool(WINAPI*)(XTaskQueueHandle, std::uint32_t, std::uint32_t);
using QueueClose = void(WINAPI*)(XTaskQueueHandle);

struct Api {
    UserAddAsync add = nullptr;
    UserAddResult result = nullptr;
    UserGetId get_id = nullptr;
    CloseUserHandle close_user = nullptr;
    UserFindById find_user = nullptr;
    ProviderStart provider_start = nullptr;
    CloseProvider close_provider = nullptr;
    ProviderResult provider_result = nullptr;
    GetStatus get_status = nullptr;
    GetResultSize get_result_size = nullptr;
    QueueCreate queue_create = nullptr;
    QueueDispatch dispatch = nullptr;
    QueueClose queue_close = nullptr;
};

XTaskQueueHandle queue = nullptr;

struct Probe {
    int calls = 0;
    UserAddResult result_proc = nullptr;
    HRESULT inside_status = E_FAIL;
    XUserHandle inside_user = nullptr;
};

void CALLBACK record(XAsyncBlock* block) { ++static_cast<Probe*>(block->context)->calls; }

// Claims the result from inside the callback, as a title commonly does.
void CALLBACK claim(XAsyncBlock* block)
{
    auto* probe = static_cast<Probe*>(block->context);
    ++probe->calls;
    probe->inside_status = probe->result_proc(block, &probe->inside_user);
}

XAsyncBlock make_block(Probe* probe, CompletionRoutine* callback)
{
    XAsyncBlock block{};
    block.context = probe;
    block.callback = callback;
    block.queue = queue;
    std::memset(block.opaque, sentinel, sizeof(block.opaque));
    return block;
}

bool opaque_untouched(const XAsyncBlock& block)
{
    for (const std::uint8_t byte : block.opaque) {
        if (byte != sentinel) return false;
    }
    return true;
}

bool expect(bool condition, const char* what)
{
    if (!condition) std::printf("FAIL: %s\n", what);
    return condition;
}

bool expect_hr(HRESULT actual, HRESULT expected, const char* what)
{
    if (actual == expected) return true;
    std::printf("FAIL: %s returned 0x%08lx, expected 0x%08lx\n", what,
                static_cast<unsigned long>(actual), static_cast<unsigned long>(expected));
    return false;
}

std::wstring normalized_path(const fs::path& path)
{
    std::error_code error;
    const fs::path canonical = fs::weakly_canonical(path, error);
    return (error ? path.lexically_normal() : canonical).wstring();
}

bool same_path(const fs::path& left, const fs::path& right)
{
    return _wcsicmp(normalized_path(left).c_str(), normalized_path(right).c_str()) == 0;
}

bool fixture_path_checks(const wchar_t* dll_path, const wchar_t* fixture_dir,
                         const wchar_t* normal_shim_dir)
{
    const fs::path dll = dll_path;
    if (dll.filename() != L"xgameruntime.dll" || !same_path(dll.parent_path(), fixture_dir)) {
        std::printf("FAIL: DLL is not at its fixture path %ls\n", fixture_dir);
        return false;
    }
    if (fs::exists(fs::path(normal_shim_dir) / L"xgameruntime.dll")) {
        std::printf("FAIL: normal shim directory contains xgameruntime.dll\n");
        return false;
    }
    return true;
}

template <typename Proc>
bool find_proc(HMODULE module, const char* name, Proc& proc)
{
    const FARPROC found = GetProcAddress(module, name);
    if (!found) {
        std::printf("FAIL: missing export %s\n", name);
        return false;
    }
    HMODULE owner = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(found), &owner) ||
        owner != module) {
        std::printf("FAIL: export %s is not owned by the fixture DLL\n", name);
        return false;
    }
    proc = reinterpret_cast<Proc>(found);
    return true;
}

bool load_api(HMODULE module, Api& api)
{
    return find_proc(module, "XUserAddAsync", api.add) &&
           find_proc(module, "XUserAddResult", api.result) &&
           find_proc(module, "XUserGetId", api.get_id) &&
           find_proc(module, "XUserCloseHandle", api.close_user) &&
           find_proc(module, "XUserFindUserById", api.find_user) &&
           find_proc(module, "XGameSaveInitializeProviderAsync", api.provider_start) &&
           find_proc(module, "XGameSaveCloseProvider", api.close_provider) &&
           find_proc(module, "XGameSaveInitializeProviderResult", api.provider_result) &&
           find_proc(module, "XAsyncGetStatus", api.get_status) &&
           find_proc(module, "XAsyncGetResultSize", api.get_result_size) &&
           find_proc(module, "XTaskQueueCreate", api.queue_create) &&
           find_proc(module, "XTaskQueueDispatch", api.dispatch) &&
           find_proc(module, "XTaskQueueCloseHandle", api.queue_close);
}

bool claimed_user_is_synthetic(const Api& api, XUserHandle user)
{
    std::uint64_t id = 0;
    return expect(user != nullptr, "claimed user handle is valid") &&
           expect_hr(api.get_id(user, &id), S_OK, "get id of the claimed user") &&
           expect(id == synthetic_user_id, "claimed user is the synthetic user");
}

// E_PENDING before dispatch (and not consuming), callback once on dispatch,
// then S_OK with one handle claimed once. Both supported options behave alike.
bool manual_contract(const Api& api, std::uint32_t option, const char* label)
{
    Probe probe;
    XAsyncBlock block = make_block(&probe, record);
    XUserHandle user = reinterpret_cast<XUserHandle>(&probe);
    bool ok = expect_hr(api.add(option, &block), S_OK, label);
    ok = ok && expect(probe.calls == 0, "no callback before XTaskQueueDispatch");
    // Completion and callback dispatch are separate: status and size answer
    // right after completion, before the callback pump (contract of #2200).
    std::size_t size = 0;
    ok = ok && expect_hr(api.get_status(&block, false), S_OK, "status right after completion") &&
         expect_hr(api.get_result_size(&block, &size), S_OK, "result size right after completion") &&
         expect(size == sizeof(XUserHandle), "result size is one user handle") &&
         expect(probe.calls == 0, "status and size do not run the callback");
    ok = ok && expect_hr(api.result(&block, &user), E_PENDING, "result before dispatch") &&
         expect(user == nullptr, "pending result leaves no handle");
    ok = ok && expect_hr(api.result(&block, &user), E_PENDING, "early result does not consume");
    ok = ok && expect(api.dispatch(queue, port_completion, 0), "dispatch runs the callback") &&
         expect(probe.calls == 1, "callback fires once on dispatch") &&
         expect(!api.dispatch(queue, port_completion, 0), "nothing left to dispatch");
    ok = ok && expect_hr(api.result(&block, &user), S_OK, "result after dispatch") &&
         claimed_user_is_synthetic(api, user);
    XUserHandle again = reinterpret_cast<XUserHandle>(&probe);
    ok = ok && expect_hr(api.result(&block, &again), E_INVALIDARG, "second result is refused") &&
         expect(again == nullptr, "second result leaves no handle") &&
         expect(opaque_untouched(block), "opaque block bytes untouched");
    if (user) api.close_user(user);
    return ok;
}

bool claim_inside_callback(const Api& api)
{
    Probe probe;
    probe.result_proc = api.result;
    XAsyncBlock block = make_block(&probe, claim);
    bool ok = expect_hr(api.add(opt_silently, &block), S_OK, "start with a claiming callback") &&
              expect(api.dispatch(queue, port_completion, 0), "dispatch") &&
              expect(probe.calls == 1, "claiming callback ran once") &&
              expect_hr(probe.inside_status, S_OK, "result inside the callback") &&
              claimed_user_is_synthetic(api, probe.inside_user);
    if (probe.inside_user) api.close_user(probe.inside_user);
    return ok;
}

bool null_callback_and_reuse(const Api& api)
{
    Probe probe;
    XAsyncBlock block = make_block(&probe, nullptr);
    XUserHandle user = nullptr;
    bool ok = expect_hr(api.add(opt_silently, &block), S_OK, "start without a callback") &&
              expect_hr(api.add(opt_silently, &block), e_illegal_state_change,
                        "reusing an in-flight block") &&
              expect(!api.dispatch(queue, port_completion, 0), "no callback to dispatch") &&
              expect_hr(api.result(&block, &user), S_OK, "result of the original start") &&
              claimed_user_is_synthetic(api, user);
    if (user) api.close_user(user);
    return ok;
}

// Unsupported options and queues, a null block and null out pointer are refused
// without tracking the block or using any capacity.
bool refusals(const Api& api)
{
    Probe probe;
    bool ok = true;
    const std::array<std::uint32_t, 5> bad_options{opt_none, opt_allow_guests,
                                                   opt_silently | opt_allow_guests,
                                                   opt_silently | opt_allowing_ui, 8};
    for (const std::uint32_t option : bad_options) {
        XAsyncBlock block = make_block(&probe, record);
        XUserHandle user = reinterpret_cast<XUserHandle>(&probe);
        ok = expect_hr(api.add(option, &block), not_implemented, "unsupported option") && ok;
        ok = expect_hr(api.result(&block, &user), E_INVALIDARG, "refused block is untracked") && ok;
    }
    ok = expect_hr(api.add(opt_silently, nullptr), E_INVALIDARG, "null block") && ok;

    XAsyncBlock null_queue = make_block(&probe, record);
    null_queue.queue = nullptr;
    XUserHandle user = nullptr;
    ok = expect_hr(api.add(opt_silently, &null_queue), not_implemented, "null queue") && ok;
    ok = expect_hr(api.result(&null_queue, &user), E_INVALIDARG, "null-queue block untracked") && ok;

    int bogus = 0;
    XAsyncBlock foreign = make_block(&probe, record);
    foreign.queue = &bogus;
    ok = expect_hr(api.add(opt_allowing_ui, &foreign), not_implemented, "foreign queue") && ok;
    ok = expect_hr(api.result(&foreign, &user), E_INVALIDARG, "foreign-queue block untracked") && ok;

    XTaskQueueHandle threaded = nullptr;
    ok = expect_hr(api.queue_create(mode_thread_pool, mode_thread_pool, &threaded), not_implemented,
                   "thread pool queue is not created") && ok;

    XAsyncBlock fine = make_block(&probe, record);
    ok = expect_hr(api.result(&fine, nullptr), E_POINTER, "null out pointer") && ok;
    ok = expect_hr(api.result(nullptr, &user), E_INVALIDARG, "null result block") && ok;
    ok = expect(probe.calls == 0, "refusals never call back") && ok;
    ok = expect(!api.dispatch(queue, port_completion, 0), "refusals queue nothing") && ok;
    return ok;
}

// A user-add block is not a game-save block, and the pending start survives a
// wrong-kind result and an attempt to close its queue.
bool wrong_kind_and_close_guard(const Api& api)
{
    Probe probe;
    XAsyncBlock block = make_block(&probe, record);
    XGameSaveProviderHandle provider = nullptr;
    XUserHandle user = nullptr;
    bool ok = expect_hr(api.add(opt_silently, &block), S_OK, "start") &&
              expect_hr(api.provider_result(&block, &provider), E_INVALIDARG,
                        "provider result on a user-add block");
    api.queue_close(queue);  // refused while a callback waits for dispatch
    ok = ok && expect(api.dispatch(queue, port_completion, 0), "queue still dispatches") &&
         expect(probe.calls == 1, "callback still fires once") &&
         expect_hr(api.result(&block, &user), S_OK, "result after the guarded close") &&
         claimed_user_is_synthetic(api, user);
    if (user) api.close_user(user);
    return ok;
}

bool shared_queue_results(const Api& api)
{
    XUserHandle requesting_user = nullptr;
    if (!expect_hr(api.find_user(synthetic_user_id, &requesting_user), S_OK, "find save user")) return false;
    Probe user_probe, provider_probe;
    XAsyncBlock user_block = make_block(&user_probe, record);
    XAsyncBlock provider_block = make_block(&provider_probe, record);
    bool ok = expect_hr(api.add(opt_silently, &user_block), S_OK, "start user-add on shared queue") &&
              expect_hr(api.provider_start(requesting_user, "synthetic-user-and-save", false, &provider_block),
                        S_OK, "start save provider on shared queue");
    std::size_t user_size = 0, provider_size = 0;
    ok = ok && expect_hr(api.get_status(&user_block, false), S_OK, "user status before dispatch") &&
         expect_hr(api.get_status(&provider_block, false), S_OK, "provider status before dispatch") &&
         expect_hr(api.get_result_size(&user_block, &user_size), S_OK, "user size before dispatch") &&
         expect_hr(api.get_result_size(&provider_block, &provider_size), S_OK, "provider size before dispatch") &&
         expect(user_size == sizeof(XUserHandle) && provider_size == sizeof(XGameSaveProviderHandle),
                "both results report their typed handle size") &&
         expect(user_probe.calls == 0 && provider_probe.calls == 0, "queries did not dispatch");
    XUserHandle user = nullptr;
    XGameSaveProviderHandle provider = nullptr;
    ok = ok && expect_hr(api.provider_result(&user_block, &provider), E_INVALIDARG, "provider rejects user block") &&
         expect_hr(api.result(&provider_block, &user), E_INVALIDARG, "user rejects provider block") &&
         expect(api.dispatch(queue, port_completion, 0) && user_probe.calls == 1 && provider_probe.calls == 0,
                "user callback dispatched first") &&
         expect(api.dispatch(queue, port_completion, 0) && provider_probe.calls == 1,
                "provider callback dispatched second") &&
         expect(!api.dispatch(queue, port_completion, 0), "shared queue drained") &&
         expect_hr(api.result(&user_block, &user), S_OK, "claim user once") &&
         claimed_user_is_synthetic(api, user) &&
         expect_hr(api.provider_result(&provider_block, &provider), S_OK, "claim provider once") &&
         expect(provider != nullptr, "provider is valid");
    if (user) api.close_user(user);
    if (provider) api.close_provider(provider);
    api.close_user(requesting_user);
    return ok && expect_hr(api.result(&user_block, &user), E_INVALIDARG, "user already consumed") &&
           expect_hr(api.provider_result(&provider_block, &provider), E_INVALIDARG, "provider already consumed");
}

// The user table is finite: a start that cannot get a handle fails, frees its
// slot and block, and later starts work again once handles are closed.
bool capacity(const Api& api)
{
    std::vector<XUserHandle> held;
    std::array<Probe, 24> probes{};
    std::array<XAsyncBlock, 24> blocks{};
    bool ok = true;
    HRESULT last = S_OK;
    std::size_t started = 0;
    for (; started < blocks.size() && last == S_OK; ++started) {
        blocks[started] = make_block(&probes[started], record);
        last = api.add(opt_silently, &blocks[started]);
        if (last != S_OK) break;
        XUserHandle user = nullptr;
        while (api.dispatch(queue, port_completion, 0)) {}
        ok = expect_hr(api.result(&blocks[started], &user), S_OK, "capacity start result") && ok;
        if (user) held.push_back(user);
    }
    ok = expect(started < blocks.size(), "user table fills before the blocks run out") && ok;
    ok = expect_hr(last, E_OUTOFMEMORY, "start with no free user handle") && ok;
    XUserHandle none = nullptr;
    ok = expect_hr(api.result(&blocks[started], &none), E_INVALIDARG,
                   "failed start leaves its block untracked") && ok;
    for (XUserHandle user : held) api.close_user(user);
    Probe probe;
    XAsyncBlock again = make_block(&probe, record);
    XUserHandle user = nullptr;
    ok = expect_hr(api.add(opt_silently, &again), S_OK, "start after closing handles") && ok;
    while (api.dispatch(queue, port_completion, 0)) {}
    ok = expect_hr(api.result(&again, &user), S_OK, "result after closing handles") && ok;
    if (user) api.close_user(user);
    return ok;
}

bool run(const wchar_t* dll_path, const wchar_t* fixture_dir, const wchar_t* normal_shim_dir)
{
    if (!fixture_path_checks(dll_path, fixture_dir, normal_shim_dir)) return false;
    HMODULE module = LoadLibraryW(dll_path);
    if (!module) {
        std::printf("FAIL: LoadLibraryW(%ls) error=%lu\n", dll_path, GetLastError());
        return false;
    }
    Api api;
    bool passed = load_api(module, api);
    if (passed) {
        passed = expect_hr(api.queue_create(mode_manual, mode_manual, &queue), S_OK, "create queue") &&
                 manual_contract(api, opt_silently, "start AddDefaultUserSilently") &&
                 manual_contract(api, opt_allowing_ui, "start AddDefaultUserAllowingUI") &&
                 claim_inside_callback(api) && null_callback_and_reuse(api) && refusals(api) &&
                 wrong_kind_and_close_guard(api) && shared_queue_results(api) && capacity(api);
        if (queue) api.queue_close(queue);
    }
    FreeLibrary(module);
    return passed;
}
}  // namespace

int wmain(int argc, wchar_t** argv)
{
    if (argc != 4) {
        std::printf("usage: xgameruntime_user_async_check <dll> <fixture-dir> <normal-shim-dir>\n");
        return 2;
    }
    const bool passed = run(argv[1], argv[2], argv[3]);
    std::printf("xgameruntime user async: add+result manual-queue contract, refusals and "
                "capacity controls %s\n",
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
