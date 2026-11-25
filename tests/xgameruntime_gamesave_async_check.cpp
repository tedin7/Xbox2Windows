// Synthetic title-level check for XGameSaveInitializeProviderAsync and
// XGameSaveInitializeProviderResult in the fixture-only xgameruntime.dll. The DLL
// is loaded from its explicit fixture path. The block mirrors the three leading
// public XAsyncBlock fields and pads the rest with a sentinel pattern, which
// the shim must leave untouched because it keeps its own state.
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <thread>

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
constexpr std::uint64_t synthetic_user_id = 0x0009000000000001ull;
constexpr HRESULT e_illegal_state_change = static_cast<HRESULT>(0x8000000DL);
constexpr std::uint8_t sentinel = 0xA5;

namespace xgameruntime_shim {
struct XUserHandleStorage;
struct XGameSaveProviderStorage;
struct XGameSaveContainerStorage;
}
using XUserHandle = xgameruntime_shim::XUserHandleStorage*;
using XGameSaveProviderHandle = xgameruntime_shim::XGameSaveProviderStorage*;
using XGameSaveContainerHandle = xgameruntime_shim::XGameSaveContainerStorage*;

struct XAsyncBlock;
using CompletionRoutine = void CALLBACK(XAsyncBlock*);
struct XAsyncBlock {
    void* queue;
    void* context;
    CompletionRoutine* callback;
    std::uint8_t opaque[sizeof(void*) * 4];
};

struct XGameRuntimeShimLogEntry {
    std::uint32_t sequence;
    HRESULT result;
    const char* component;
    const char* function;
};

using FindUserById = HRESULT(WINAPI*)(std::uint64_t, XUserHandle*);
using CloseUserHandle = void(WINAPI*)(XUserHandle);
using InitializeProviderAsync = HRESULT(WINAPI*)(XUserHandle, const char*, bool, XAsyncBlock*);
using InitializeProviderResult = HRESULT(WINAPI*)(XAsyncBlock*, XGameSaveProviderHandle*);
using CreateContainer = HRESULT(WINAPI*)(XGameSaveProviderHandle, const char*,
                                        XGameSaveContainerHandle*);
using CloseProvider = void(WINAPI*)(XGameSaveProviderHandle);
using CloseContainer = void(WINAPI*)(XGameSaveContainerHandle);
struct XTaskQueueHandleStorage;
using XTaskQueueHandle = XTaskQueueHandleStorage*;
// Shared manual queue: a null XAsyncBlock queue is refused, so every block uses it.
XTaskQueueHandle shared_queue = nullptr;
constexpr std::uint32_t mode_manual = 0;
constexpr std::uint32_t mode_thread_pool = 1;
constexpr std::uint32_t port_work = 0;
constexpr std::uint32_t port_completion = 1;
using QueueCreate = HRESULT(WINAPI*)(std::uint32_t, std::uint32_t, XTaskQueueHandle*);
using QueueDispatch = bool(WINAPI*)(XTaskQueueHandle, std::uint32_t, std::uint32_t);
using QueueClose = void(WINAPI*)(XTaskQueueHandle);
using GetLog = std::size_t(WINAPI*)(XGameRuntimeShimLogEntry*, std::size_t);
using ClearLog = void(WINAPI*)();

struct CallbackProbe {
    int calls = 0;
    XAsyncBlock* seen = nullptr;
    XGameSaveProviderHandle result_inside = nullptr;
    HRESULT result_status = E_FAIL;
    InitializeProviderResult result_proc = nullptr;
};

void CALLBACK record_completion(XAsyncBlock* block)
{
    auto* probe = static_cast<CallbackProbe*>(block->context);
    ++probe->calls;
    probe->seen = block;
}

// Claims the result from inside the callback, as a title commonly does.
void CALLBACK claim_in_callback(XAsyncBlock* block)
{
    auto* probe = static_cast<CallbackProbe*>(block->context);
    ++probe->calls;
    probe->seen = block;
    probe->result_status = probe->result_proc(block, &probe->result_inside);
}

XAsyncBlock make_block(CallbackProbe* probe, CompletionRoutine* callback)
{
    XAsyncBlock block{};
    block.context = probe;
    block.callback = callback;
    block.queue = shared_queue;
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

fs::path normalized_path(const fs::path& path)
{
    std::error_code error;
    const auto canonical = fs::weakly_canonical(path, error);
    return error ? fs::absolute(path).lexically_normal() : canonical;
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

struct Api {
    FindUserById find_user = nullptr;
    CloseUserHandle close_user = nullptr;
    InitializeProviderAsync initialize_async = nullptr;
    InitializeProviderResult initialize_result = nullptr;
    CreateContainer create_container = nullptr;
    CloseProvider close_provider = nullptr;
    CloseContainer close_container = nullptr;
    QueueCreate queue_create = nullptr;
    QueueDispatch queue_dispatch = nullptr;
    QueueClose queue_close = nullptr;
};

bool load_api(HMODULE module, Api& api)
{
    return find_proc(module, "XUserFindUserById", api.find_user) &&
           find_proc(module, "XUserCloseHandle", api.close_user) &&
           find_proc(module, "XGameSaveInitializeProviderAsync", api.initialize_async) &&
           find_proc(module, "XGameSaveInitializeProviderResult", api.initialize_result) &&
           find_proc(module, "XGameSaveCreateContainer", api.create_container) &&
           find_proc(module, "XGameSaveCloseProvider", api.close_provider) &&
           find_proc(module, "XGameSaveCloseContainer", api.close_container) &&
           find_proc(module, "XTaskQueueCreate", api.queue_create) &&
           find_proc(module, "XTaskQueueDispatch", api.queue_dispatch) &&
           find_proc(module, "XTaskQueueCloseHandle", api.queue_close);
}

constexpr char configuration_id[] = "synthetic-scid";

bool positive_completion(const Api& api, XUserHandle user)
{
    CallbackProbe probe;
    XAsyncBlock block = make_block(&probe, record_completion);
    bool ok = expect_hr(api.initialize_async(user, configuration_id, false, &block), S_OK,
                        "async initialize");
    ok = expect(probe.calls == 0, "no callback before the completion port is pumped") && ok;
    ok = expect(api.queue_dispatch(shared_queue, port_completion, 0) && probe.calls == 1 &&
                    probe.seen == &block,
                "callback ran once with the caller's block after dispatch") && ok;
    ok = expect(opaque_untouched(block) && block.context == &probe &&
                    block.callback == record_completion && block.queue == shared_queue,
                "the caller's block fields are left as written") && ok;

    XGameSaveProviderHandle provider = reinterpret_cast<XGameSaveProviderHandle>(1);
    ok = expect_hr(api.initialize_result(&block, &provider), S_OK, "result") && ok;
    ok = expect(provider != nullptr, "result returns a provider") && ok;

    XGameSaveContainerHandle container = nullptr;
    ok = expect_hr(api.create_container(provider, "async-slot", &container), S_OK,
                   "provider from the async path creates a container") && ok;
    ok = expect(container != nullptr, "container handle is set") && ok;
    api.close_container(container);
    api.close_provider(provider);

    XGameSaveProviderHandle again = reinterpret_cast<XGameSaveProviderHandle>(1);
    ok = expect_hr(api.initialize_result(&block, &again), E_INVALIDARG,
                   "second result for the same operation") && ok;
    ok = expect(again == nullptr, "failed result clears the output") && ok;
    return ok;
}

bool null_callback_and_callback_claim(const Api& api, XUserHandle user)
{
    CallbackProbe silent;
    XAsyncBlock plain = make_block(&silent, nullptr);
    XGameSaveProviderHandle first = nullptr;
    bool ok = expect_hr(api.initialize_async(user, configuration_id, true, &plain), S_OK,
                        "async initialize without a callback");
    ok = expect_hr(api.initialize_result(&plain, &first), S_OK, "result without a callback") && ok;
    ok = expect(first != nullptr && silent.calls == 0, "no callback is invented") && ok;
    api.close_provider(first);

    CallbackProbe claimed;
    claimed.result_proc = api.initialize_result;
    XAsyncBlock block = make_block(&claimed, claim_in_callback);
    ok = expect_hr(api.initialize_async(user, configuration_id, false, &block), S_OK,
                   "async initialize claimed inside the callback") && ok;
    ok = expect(api.queue_dispatch(shared_queue, port_completion, 0), "dispatch ran the callback") && ok;
    ok = expect(claimed.calls == 1 && claimed.result_status == S_OK &&
                    claimed.result_inside != nullptr,
                "the callback can retrieve the result") && ok;
    api.close_provider(claimed.result_inside);
    XGameSaveProviderHandle leftover = nullptr;
    ok = expect_hr(api.initialize_result(&block, &leftover), E_INVALIDARG,
                   "result already claimed in the callback") && ok;
    return ok;
}

bool argument_errors(const Api& api, XUserHandle user)
{
    CallbackProbe probe;
    XAsyncBlock block = make_block(&probe, record_completion);
    XGameSaveProviderHandle provider = nullptr;
    bool ok = expect_hr(api.initialize_async(user, configuration_id, false, nullptr), E_INVALIDARG,
                        "null block");
    ok = expect_hr(api.initialize_async(user, nullptr, false, &block), E_INVALIDARG,
                   "null configuration id") && ok;
    ok = expect_hr(api.initialize_async(user, "", false, &block), E_INVALIDARG,
                   "empty configuration id") && ok;
    ok = expect_hr(api.initialize_async(nullptr, configuration_id, false, &block), E_HANDLE,
                   "null user") && ok;

    XUserHandle closed = nullptr;
    ok = expect_hr(api.find_user(synthetic_user_id, &closed), S_OK, "second user handle") && ok;
    api.close_user(closed);
    ok = expect_hr(api.initialize_async(closed, configuration_id, false, &block), E_HANDLE,
                   "closed user handle") && ok;
    ok = expect(probe.calls == 0 && opaque_untouched(block),
                "rejected calls never complete or touch the block") && ok;
    ok = expect_hr(api.initialize_result(&block, &provider), E_INVALIDARG,
                   "rejected call leaves nothing to claim") && ok;

    ok = expect_hr(api.initialize_result(&block, nullptr), E_POINTER, "result null output") && ok;
    provider = reinterpret_cast<XGameSaveProviderHandle>(1);
    ok = expect_hr(api.initialize_result(nullptr, &provider), E_INVALIDARG, "result null block") && ok;
    ok = expect(provider == nullptr, "result null block clears the output") && ok;
    return ok;
}

bool reuse_and_capacity(const Api& api, XUserHandle user)
{
    CallbackProbe probe;
    XAsyncBlock block = make_block(&probe, record_completion);
    bool ok = expect_hr(api.initialize_async(user, configuration_id, false, &block), S_OK,
                        "first start");
    ok = expect_hr(api.initialize_async(user, configuration_id, false, &block),
                   e_illegal_state_change, "reusing an unclaimed block") && ok;
    ok = expect(probe.calls == 0, "refused reuse queues nothing more") && ok;
    ok = expect(api.queue_dispatch(shared_queue, port_completion, 0) &&
                    !api.queue_dispatch(shared_queue, port_completion, 0) && probe.calls == 1,
                "exactly one callback for the original start") && ok;
    XGameSaveProviderHandle held = nullptr;
    ok = expect_hr(api.initialize_result(&block, &held), S_OK, "original result survives") && ok;

    // The fixture holds four providers. Fill the rest, then show a refused start
    // consumes no async state and a freed provider makes room again.
    std::array<XAsyncBlock, 4> blocks{};
    std::array<XGameSaveProviderHandle, 4> providers{};
    std::size_t filled = 0;
    for (; filled < blocks.size(); ++filled) {
        blocks[filled] = make_block(&probe, nullptr);
        providers[filled] = nullptr;
        if (api.initialize_async(user, configuration_id, false, &blocks[filled]) != S_OK) break;
        if (FAILED(api.initialize_result(&blocks[filled], &providers[filled]))) {
            std::printf("FAIL: result for filled provider %zu\n", filled);
            return false;
        }
    }
    ok = expect(filled == 3, "three more providers fit beside the held one") && ok;
    XAsyncBlock refused = make_block(&probe, nullptr);
    ok = expect_hr(api.initialize_async(user, configuration_id, false, &refused), E_OUTOFMEMORY,
                   "provider storage exhausted") && ok;
    XGameSaveProviderHandle none = nullptr;
    ok = expect_hr(api.initialize_result(&refused, &none), E_INVALIDARG,
                   "refused start left no async state") && ok;
    api.close_provider(held);
    ok = expect_hr(api.initialize_async(user, configuration_id, false, &refused), S_OK,
                   "start after a provider was freed") && ok;
    XGameSaveProviderHandle revived = nullptr;
    ok = expect_hr(api.initialize_result(&refused, &revived), S_OK, "result after freeing") && ok;
    api.close_provider(revived);
    for (std::size_t i = 0; i < filled; ++i) api.close_provider(providers[i]);
    return ok;
}

bool log_checks(HMODULE module)
{
    GetLog get_log = nullptr;
    if (!find_proc(module, "XGameRuntimeShimGetLog", get_log)) return false;
    std::array<XGameRuntimeShimLogEntry, 256> entries{};
    const std::size_t count = get_log(entries.data(), entries.size());
    bool async_ok = false, async_handle = false, result_ok = false, result_invalid = false;
    for (std::size_t i = 0; i < count && i < entries.size(); ++i) {
        const auto& entry = entries[i];
        if (!entry.function) return false;
        if (entry.result == not_implemented) {
            std::printf("FAIL: %s logged E_NOTIMPL\n", entry.function);
            return false;
        }
        if (std::strcmp(entry.function, "XGameSaveInitializeProviderAsync") == 0) {
            async_ok = async_ok || entry.result == S_OK;
            async_handle = async_handle || entry.result == E_HANDLE;
        }
        if (std::strcmp(entry.function, "XGameSaveInitializeProviderResult") == 0) {
            result_ok = result_ok || entry.result == S_OK;
            result_invalid = result_invalid || entry.result == E_INVALIDARG;
        }
    }
    return expect(async_ok && async_handle && result_ok && result_invalid,
                  "log carries the positive and negative async calls");
}

// Fills every free provider slot, then returns how many fit.
std::size_t fill_providers(const Api& api, XUserHandle user, std::array<XAsyncBlock, 8>& blocks,
                           std::array<XGameSaveProviderHandle, 8>& providers, CallbackProbe& probe)
{
    std::size_t filled = 0;
    for (; filled < blocks.size(); ++filled) {
        blocks[filled] = make_block(&probe, nullptr);
        if (api.initialize_async(user, configuration_id, false, &blocks[filled]) != S_OK) break;
        api.initialize_result(&blocks[filled], &providers[filled]);
    }
    return filled;
}

// Manual completion port: no callback before XTaskQueueDispatch, one after, and
// the callback claims the provider. A null queue is refused, so every block here
// uses a queue and the zero-before-dispatch checks stand in for an inline callback.
bool manual_queue(const Api& api, XUserHandle user)
{
    XTaskQueueHandle queue = nullptr;
    bool ok = expect_hr(api.queue_create(mode_manual, mode_manual, &queue), S_OK, "create manual queue");
    if (!ok || !queue) return false;
    CallbackProbe probe;
    probe.result_proc = api.initialize_result;
    XAsyncBlock block = make_block(&probe, claim_in_callback);
    block.queue = queue;
    ok = expect_hr(api.initialize_async(user, configuration_id, false, &block), S_OK,
                   "start on manual queue") && ok;
    ok = expect(probe.calls == 0, "zero callbacks before completion-port dispatch") && ok;
    ok = expect(!api.queue_dispatch(queue, port_work, 0) && probe.calls == 0,
                "work port dispatch runs nothing") && ok;
    ok = expect(opaque_untouched(block), "opaque block bytes untouched") && ok;
    ok = expect(api.queue_dispatch(queue, port_completion, 0), "completion dispatch ran a callback") && ok;
    ok = expect(probe.calls == 1 && probe.seen == &block, "one callback after dispatch") && ok;
    ok = expect_hr(probe.result_status, S_OK, "callback claimed the provider") && ok;
    ok = expect(probe.result_inside != nullptr, "callback got a provider") && ok;
    ok = expect(!api.queue_dispatch(queue, port_completion, 0) && probe.calls == 1,
                "second dispatch runs nothing") && ok;
    if (probe.result_inside) api.close_provider(probe.result_inside);

    // Early result retrieval (Manual/Manual): the result is refused, nothing is lost.
    CallbackProbe early_claim;
    XAsyncBlock early_block = make_block(&early_claim, record_completion);
    early_block.queue = queue;
    ok = expect_hr(api.initialize_async(user, configuration_id, false, &early_block), S_OK,
                   "start before an early result") && ok;
    XGameSaveProviderHandle early_provider = reinterpret_cast<XGameSaveProviderHandle>(1);
    ok = expect_hr(api.initialize_result(&early_block, &early_provider), E_PENDING,
                   "result before dispatch is refused") && ok;
    ok = expect(early_provider == nullptr && early_claim.calls == 0,
                "early result hands out no provider and fires nothing") && ok;
    api.queue_close(queue);
    ok = expect(api.queue_dispatch(queue, port_completion, 0) && early_claim.calls == 1,
                "the queued callback still dispatches after an early result") && ok;
    ok = expect(!api.queue_dispatch(queue, port_completion, 0) && early_claim.calls == 1,
                "second dispatch finds nothing") && ok;
    ok = expect_hr(api.initialize_result(&early_block, &early_provider), S_OK,
                   "result after dispatch") && ok;
    ok = expect(early_provider != nullptr, "provider delivered once") && ok;
    XGameSaveProviderHandle twice = reinterpret_cast<XGameSaveProviderHandle>(1);
    ok = expect_hr(api.initialize_result(&early_block, &twice), E_INVALIDARG,
                   "provider is not delivered twice") && ok;
    ok = expect(twice == nullptr, "second delivery clears the output") && ok;
    if (early_provider) api.close_provider(early_provider);

    // Early-callback mutant: a callback fired inline would be counted, so the
    // zero-before-dispatch checks above can fail. Prove the probe sees one.
    CallbackProbe mutant;
    XAsyncBlock mutant_block = make_block(&mutant, record_completion);
    mutant_block.callback(&mutant_block);
    ok = expect(mutant.calls == 1, "control: an early callback is visible to the probe") && ok;

    const ULONGLONG started = GetTickCount64();
    ok = expect(!api.queue_dispatch(queue, port_completion, 75) && GetTickCount64() - started >= 45,
                "finite wait on an empty port expires without dispatch") && ok;
    CallbackProbe timed;
    XAsyncBlock timed_block = make_block(&timed, record_completion);
    timed_block.queue = queue;
    HRESULT start_status = E_FAIL;
    std::thread producer([&] {
        Sleep(40);
        start_status = api.initialize_async(user, configuration_id, false, &timed_block);
    });
    const bool dispatched = api.queue_dispatch(queue, port_completion, 2000);
    producer.join();
    ok = expect_hr(start_status, S_OK, "start while dispatch is waiting") && ok;
    ok = expect(dispatched && timed.calls == 1, "waiting dispatch receives later completion once") && ok;
    ok = expect(!api.queue_dispatch(queue, port_completion, 0), "zero timeout finds no second callback") && ok;
    XGameSaveProviderHandle timed_provider = nullptr;
    api.initialize_result(&timed_block, &timed_provider);
    if (timed_provider) api.close_provider(timed_provider);

    // A pending callback keeps its queue open.
    CallbackProbe pending;
    XAsyncBlock pending_block = make_block(&pending, record_completion);
    pending_block.queue = queue;
    ok = expect_hr(api.initialize_async(user, configuration_id, false, &pending_block), S_OK,
                   "pending start") && ok;
    api.queue_close(queue);
    ok = expect(api.queue_dispatch(queue, port_completion, 0) && pending.calls == 1,
                "queue with a pending callback survives close") && ok;
    XGameSaveProviderHandle pending_provider = nullptr;
    api.initialize_result(&pending_block, &pending_provider);
    if (pending_provider) api.close_provider(pending_provider);
    api.queue_close(queue);
    return ok;
}

// An unsupported queue is refused with E_NOTIMPL, and block and provider
// capacity stay untouched: every slot is still usable afterwards.
bool unsupported_queue(const Api& api, XUserHandle user)
{
    XTaskQueueHandle none = nullptr;
    bool ok = expect_hr(api.queue_create(mode_thread_pool, mode_thread_pool, &none), not_implemented,
                        "thread-pool queue is unsupported") && expect(none == nullptr, "no handle returned");
    int stray = 0;
    CallbackProbe probe;
    XAsyncBlock block = make_block(&probe, record_completion);
    block.queue = &stray;
    ok = expect_hr(api.initialize_async(user, configuration_id, false, &block), not_implemented,
                   "unknown queue refused") && ok;
    ok = expect(probe.calls == 0 && opaque_untouched(block), "refusal fires nothing and writes nothing") && ok;
    XGameSaveProviderHandle none_provider = nullptr;
    ok = expect_hr(api.initialize_result(&block, &none_provider), E_INVALIDARG,
                   "refusal left no async state") && ok;
    // A null queue means the process default queue in the public API: refused too.
    block.queue = nullptr;
    ok = expect_hr(api.initialize_async(user, configuration_id, false, &block), not_implemented,
                   "null queue refused") && ok;
    ok = expect(probe.calls == 0 && opaque_untouched(block), "null queue fires and writes nothing") && ok;
    ok = expect_hr(api.initialize_result(&block, &none_provider), E_INVALIDARG,
                   "null queue left no async state") && ok;
    block.queue = &stray;
    // Capacity untouched: the whole provider pool is still free after many refusals.
    for (int i = 0; i < 3; ++i) api.initialize_async(user, configuration_id, false, &block);
    std::array<XAsyncBlock, 8> blocks{};
    std::array<XGameSaveProviderHandle, 8> providers{};
    const std::size_t filled = fill_providers(api, user, blocks, providers, probe);
    ok = expect(filled == 4, "all four provider slots still free after refusals") && ok;
    for (std::size_t i = 0; i < filled; ++i) api.close_provider(providers[i]);
    return ok;
}

// The log keeps 64 entries, so this runs on a cleared log right after the refusals.
bool refusal_log_checks(HMODULE module)
{
    GetLog get_log = nullptr;
    if (!find_proc(module, "XGameRuntimeShimGetLog", get_log)) return false;
    std::array<XGameRuntimeShimLogEntry, 64> entries{};
    const std::size_t count = get_log(entries.data(), entries.size());
    bool create_refused = false, async_refused = false;
    for (std::size_t i = 0; i < count && i < entries.size(); ++i) {
        const auto& entry = entries[i];
        if (!entry.function) return false;
        if (entry.result != not_implemented) continue;
        create_refused = create_refused || std::strcmp(entry.function, "XTaskQueueCreate") == 0;
        async_refused = async_refused ||
                        std::strcmp(entry.function, "XGameSaveInitializeProviderAsync") == 0;
        if (std::strcmp(entry.function, "XTaskQueueCreate") != 0 &&
            std::strcmp(entry.function, "XGameSaveInitializeProviderAsync") != 0) {
            std::printf("FAIL: %s logged E_NOTIMPL\n", entry.function);
            return false;
        }
    }
    return expect(create_refused && async_refused, "log carries the refused queue calls");
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
    ClearLog clear_log = nullptr;
    bool passed = load_api(module, api) && find_proc(module, "XGameRuntimeShimClearLog", clear_log);
    XUserHandle user = nullptr;
    if (passed) {
        clear_log();
        passed = expect_hr(api.find_user(synthetic_user_id, &user), S_OK, "find user") &&
                 expect_hr(api.queue_create(mode_manual, mode_manual, &shared_queue), S_OK,
                           "create shared queue") &&
                 positive_completion(api, user) && null_callback_and_callback_claim(api, user) &&
                 argument_errors(api, user) && reuse_and_capacity(api, user) &&
                 log_checks(module) && manual_queue(api, user) && (clear_log(), true) &&
                 unsupported_queue(api, user) && refusal_log_checks(module);
        if (shared_queue) api.queue_close(shared_queue);
        if (user) api.close_user(user);
    }
    FreeLibrary(module);
    return passed;
}
}  // namespace

int wmain(int argc, wchar_t** argv)
{
    if (argc != 4) {
        std::printf("usage: xgameruntime_gamesave_async_check <dll> <fixture-dir> <normal-shim-dir>\n");
        return 2;
    }
    const bool passed = run(argv[1], argv[2], argv[3]);
    std::printf("xgameruntime game-save async: initialize+result positive, callback, argument, "
                "reuse and capacity controls %s\n",
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
