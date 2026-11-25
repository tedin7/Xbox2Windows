// Synthetic title-level check for XAsyncGetStatus and XAsyncGetResultSize in the
// fixture-only xgameruntime.dll. Blocks come from XGameSaveInitializeProviderAsync
// on a manual task queue (#2179); the completion-port contract is the one pinned
// by startup_async_manual_completion (#2170). Public docs:
// https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xasync/functions/xasyncgetstatus
// https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xasync/functions/xasyncgetresultsize
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace {
namespace fs = std::filesystem;

constexpr std::uint64_t synthetic_user_id = 0x0009000000000001ull;
constexpr std::uint8_t sentinel = 0xA5;
constexpr char configuration_id[] = "synthetic-scid";

struct XUserHandleStorage;
struct XGameSaveProviderStorage;
using XUserHandle = XUserHandleStorage*;
using XGameSaveProviderHandle = XGameSaveProviderStorage*;
struct XTaskQueueHandleStorage;
using XTaskQueueHandle = XTaskQueueHandleStorage*;

struct XAsyncBlock;
using CompletionRoutine = void CALLBACK(XAsyncBlock*);
struct XAsyncBlock {
    void* queue;
    void* context;
    CompletionRoutine* callback;
    std::uint8_t opaque[sizeof(void*) * 4];
};

constexpr std::uint32_t mode_manual = 0;
constexpr std::uint32_t port_completion = 1;

using FindUserById = HRESULT(WINAPI*)(std::uint64_t, XUserHandle*);
using CloseUserHandle = void(WINAPI*)(XUserHandle);
using InitializeAsync = HRESULT(WINAPI*)(XUserHandle, const char*, bool, XAsyncBlock*);
using InitializeResult = HRESULT(WINAPI*)(XAsyncBlock*, XGameSaveProviderHandle*);
using CloseProvider = void(WINAPI*)(XGameSaveProviderHandle);
using GetStatus = HRESULT(WINAPI*)(XAsyncBlock*, bool);
using GetResultSize = HRESULT(WINAPI*)(XAsyncBlock*, std::size_t*);
using QueueCreate = HRESULT(WINAPI*)(std::uint32_t, std::uint32_t, XTaskQueueHandle*);
using QueueDispatch = bool(WINAPI*)(XTaskQueueHandle, std::uint32_t, std::uint32_t);
using QueueClose = void(WINAPI*)(XTaskQueueHandle);

struct Api {
    FindUserById find_user = nullptr;
    CloseUserHandle close_user = nullptr;
    InitializeAsync initialize_async = nullptr;
    InitializeResult initialize_result = nullptr;
    CloseProvider close_provider = nullptr;
    GetStatus get_status = nullptr;
    GetResultSize get_result_size = nullptr;
    QueueCreate queue_create = nullptr;
    QueueDispatch queue_dispatch = nullptr;
    QueueClose queue_close = nullptr;
};

struct Probe {
    int calls = 0;
    HRESULT status_inside = E_FAIL;
    GetStatus get_status = nullptr;
};

void CALLBACK count_callback(XAsyncBlock* block)
{
    ++static_cast<Probe*>(block->context)->calls;
}

// A title commonly polls the status from inside its completion callback.
void CALLBACK poll_in_callback(XAsyncBlock* block)
{
    auto* probe = static_cast<Probe*>(block->context);
    ++probe->calls;
    probe->status_inside = probe->get_status(block, false);
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

bool opaque_untouched(const XAsyncBlock& block)
{
    for (const std::uint8_t byte : block.opaque) {
        if (byte != sentinel) return false;
    }
    return true;
}

XAsyncBlock make_block(XTaskQueueHandle queue, Probe* probe, CompletionRoutine* callback)
{
    XAsyncBlock block{};
    block.queue = queue;
    block.context = probe;
    block.callback = callback;
    std::memset(block.opaque, sentinel, sizeof(block.opaque));
    return block;
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
    return find_proc(module, "XUserFindUserById", api.find_user) &&
           find_proc(module, "XUserCloseHandle", api.close_user) &&
           find_proc(module, "XGameSaveInitializeProviderAsync", api.initialize_async) &&
           find_proc(module, "XGameSaveInitializeProviderResult", api.initialize_result) &&
           find_proc(module, "XGameSaveCloseProvider", api.close_provider) &&
           find_proc(module, "XAsyncGetStatus", api.get_status) &&
           find_proc(module, "XAsyncGetResultSize", api.get_result_size) &&
           find_proc(module, "XTaskQueueCreate", api.queue_create) &&
           find_proc(module, "XTaskQueueDispatch", api.queue_dispatch) &&
           find_proc(module, "XTaskQueueCloseHandle", api.queue_close);
}

// Both queries report the completed operation before and after the one dispatch; consuming the result ends tracking of the block.
bool manual_queue_lifecycle(const Api& api, XUserHandle user, XTaskQueueHandle queue)
{
    Probe probe;
    XAsyncBlock block = make_block(queue, &probe, count_callback);
    bool ok = expect_hr(api.initialize_async(user, configuration_id, false, &block), S_OK,
                        "async initialize");
    std::size_t size = 123;
    // Completion is separate from callback dispatch: both queries report the
    // finished operation before the first dispatch, and fire no callback.
    ok = expect_hr(api.get_status(&block, false), S_OK, "status before dispatch") && ok;
    ok = expect_hr(api.get_result_size(&block, &size), S_OK, "result size before dispatch") && ok;
    ok = expect(size == sizeof(XGameSaveProviderHandle), "result size before dispatch is one provider handle") && ok;
    ok = expect(probe.calls == 0, "status queries fire no callback") && ok;

    // wait=true never blocks on an operation that already completed.
    ok = expect_hr(api.get_status(&block, true), S_OK, "waiting status of a completed operation") && ok;
    ok = expect(probe.calls == 0, "waiting status does not dispatch the callback") && ok;

    ok = expect(api.queue_dispatch(queue, port_completion, 0) && probe.calls == 1,
                "one dispatch runs the callback") && ok;
    ok = expect_hr(api.get_status(&block, false), S_OK, "status after dispatch") && ok;
    ok = expect_hr(api.get_result_size(&block, &size), S_OK, "result size after dispatch") && ok;
    ok = expect(size == sizeof(XGameSaveProviderHandle), "result size is one provider handle") && ok;
    ok = expect_hr(api.get_result_size(&block, &size), S_OK, "result size is repeatable") && ok;
    ok = expect(opaque_untouched(block) && block.queue == queue && block.context == &probe &&
                    block.callback == count_callback,
                "queries leave the caller's block as written") && ok;

    XGameSaveProviderHandle provider = nullptr;
    ok = expect_hr(api.initialize_result(&block, &provider), S_OK, "result after status queries") && ok;
    ok = expect(provider != nullptr, "provider survives the status queries") && ok;
    if (provider) api.close_provider(provider);

    size = 77;
    ok = expect_hr(api.get_status(&block, false), E_INVALIDARG, "status after the result was consumed") && ok;
    ok = expect_hr(api.get_result_size(&block, &size), E_INVALIDARG,
                   "result size after the result was consumed") && ok;
    ok = expect(size == 0, "refused result size clears the output") && ok;
    return ok;
}

// Without a callback nothing waits for dispatch, so the operation is complete at once.
bool no_callback_block(const Api& api, XUserHandle user, XTaskQueueHandle queue)
{
    Probe probe;
    XAsyncBlock block = make_block(queue, &probe, nullptr);
    std::size_t size = 0;
    bool ok = expect_hr(api.initialize_async(user, configuration_id, true, &block), S_OK,
                        "async initialize without a callback");
    ok = expect_hr(api.get_status(&block, false), S_OK, "status without a callback") && ok;
    ok = expect_hr(api.get_result_size(&block, &size), S_OK, "result size without a callback") && ok;
    ok = expect(size == sizeof(XGameSaveProviderHandle), "exact result size without a callback") && ok;
    XGameSaveProviderHandle provider = nullptr;
    ok = expect_hr(api.initialize_result(&block, &provider), S_OK, "result without a callback") && ok;
    if (provider) api.close_provider(provider);
    return ok;
}

// The completion callback already sees a completed operation.
bool status_inside_callback(const Api& api, XUserHandle user, XTaskQueueHandle queue)
{
    Probe probe;
    probe.get_status = api.get_status;
    XAsyncBlock block = make_block(queue, &probe, poll_in_callback);
    bool ok = expect_hr(api.initialize_async(user, configuration_id, false, &block), S_OK,
                        "async initialize with a polling callback");
    ok = expect(api.queue_dispatch(queue, port_completion, 0) && probe.calls == 1,
                "polling callback ran once") && ok;
    ok = expect_hr(probe.status_inside, S_OK, "status inside the callback") && ok;
    XGameSaveProviderHandle provider = nullptr;
    ok = expect_hr(api.initialize_result(&block, &provider), S_OK, "result after polling callback") && ok;
    if (provider) api.close_provider(provider);
    return ok;
}

bool refusals(const Api& api, XTaskQueueHandle queue)
{
    std::size_t size = 55;
    XAsyncBlock untracked = make_block(queue, nullptr, nullptr);
    bool ok = expect_hr(api.get_status(nullptr, false), E_INVALIDARG, "status of a null block");
    ok = expect_hr(api.get_status(nullptr, true), E_INVALIDARG, "waiting status of a null block") && ok;
    ok = expect_hr(api.get_status(&untracked, false), E_INVALIDARG, "status of an untracked block") && ok;
    ok = expect_hr(api.get_status(&untracked, true), E_INVALIDARG, "waiting status of an untracked block") && ok;
    ok = expect_hr(api.get_result_size(nullptr, &size), E_INVALIDARG, "result size of a null block") && ok;
    ok = expect(size == 0, "null block clears the size output") && ok;
    size = 55;
    ok = expect_hr(api.get_result_size(&untracked, &size), E_INVALIDARG,
                   "result size of an untracked block") && ok;
    ok = expect(size == 0, "untracked block clears the size output") && ok;
    ok = expect_hr(api.get_result_size(&untracked, nullptr), E_POINTER, "null size output") && ok;
    ok = expect(opaque_untouched(untracked), "refusals leave the block untouched") && ok;
    return ok;
}

// A queue close is refused while the callback waits, so the pending state and
// the later dispatch both still work.
bool close_guard(const Api& api, XUserHandle user)
{
    XTaskQueueHandle queue = nullptr;
    bool ok = expect_hr(api.queue_create(mode_manual, mode_manual, &queue), S_OK, "create guard queue");
    if (!ok) return false;
    Probe probe;
    XAsyncBlock block = make_block(queue, &probe, count_callback);
    ok = expect_hr(api.initialize_async(user, configuration_id, false, &block), S_OK,
                   "async initialize on the guard queue") && ok;
    api.queue_close(queue);
    ok = expect_hr(api.get_status(&block, false), S_OK, "status after a refused queue close") && ok;
    ok = expect(api.queue_dispatch(queue, port_completion, 0) && probe.calls == 1,
                "dispatch still runs after a refused queue close") && ok;
    ok = expect_hr(api.get_status(&block, false), S_OK, "status after guarded dispatch") && ok;
    XGameSaveProviderHandle provider = nullptr;
    ok = expect_hr(api.initialize_result(&block, &provider), S_OK, "result on the guard queue") && ok;
    if (provider) api.close_provider(provider);
    api.queue_close(queue);
    return ok;
}

bool run(const wchar_t* dll_path, const wchar_t* fixture_dir, const wchar_t* normal_shim_dir)
{
    const fs::path dll = dll_path;
    if (dll.filename() != L"xgameruntime.dll" ||
        _wcsicmp(fs::absolute(dll.parent_path()).lexically_normal().c_str(),
                 fs::absolute(fixture_dir).lexically_normal().c_str()) != 0 ||
        fs::exists(fs::path(normal_shim_dir) / L"xgameruntime.dll")) {
        std::printf("FAIL: fixture DLL path checks\n");
        return false;
    }
    HMODULE module = LoadLibraryW(dll_path);
    if (!module) {
        std::printf("FAIL: LoadLibraryW(%ls) error=%lu\n", dll_path, GetLastError());
        return false;
    }
    Api api;
    XUserHandle user = nullptr;
    XTaskQueueHandle queue = nullptr;
    bool passed = load_api(module, api) &&
                  expect_hr(api.find_user(synthetic_user_id, &user), S_OK, "find user") &&
                  expect_hr(api.queue_create(mode_manual, mode_manual, &queue), S_OK, "create queue");
    if (passed) {
        passed = manual_queue_lifecycle(api, user, queue) && no_callback_block(api, user, queue) &&
                 status_inside_callback(api, user, queue) && refusals(api, queue) &&
                 close_guard(api, user);
    }
    if (queue) api.queue_close(queue);
    if (user) api.close_user(user);
    FreeLibrary(module);
    return passed;
}
}  // namespace

int wmain(int argc, wchar_t** argv)
{
    if (argc != 4) {
        std::printf("usage: xgameruntime_async_status_check <dll> <fixture-dir> <normal-shim-dir>\n");
        return 2;
    }
    const bool passed = run(argv[1], argv[2], argv[3]);
    std::printf("xgameruntime async status: pending vs dispatched status and result size, "
                "refusals and close guard %s\n",
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
