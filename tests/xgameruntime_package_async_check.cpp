// Synthetic title-level check for XPackageMountWithUiAsync and
// XPackageMountWithUiResult in the fixture-only xgameruntime.dll, on the
// shim-owned manual task queue (same contract as #2179 and #2201). The DLL is
// loaded from its explicit fixture path. The block mirrors the three leading
// public XAsyncBlock fields and pads the rest with a sentinel pattern, which
// the shim must leave untouched because it keeps its own state.
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <string>
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
constexpr std::uint8_t sentinel = 0xA5;
constexpr const char* package_name = "synthetic-package";

namespace xgameruntime_shim {
struct XUserHandleStorage;
struct XGameSaveProviderStorage;
struct XPackageMountStorage;
}
using XUserHandle = xgameruntime_shim::XUserHandleStorage*;
using XGameSaveProviderHandle = xgameruntime_shim::XGameSaveProviderStorage*;
using XPackageMountHandle = xgameruntime_shim::XPackageMountStorage*;

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
constexpr std::uint32_t port_completion = 1;

using MountAsync = HRESULT(WINAPI*)(const char*, XAsyncBlock*);
using MountResult = HRESULT(WINAPI*)(XAsyncBlock*, XPackageMountHandle*);
using MountSync = HRESULT(WINAPI*)(const char*, XPackageMountHandle*);
using PathSize = HRESULT(WINAPI*)(XPackageMountHandle, std::size_t*);
using PathGet = HRESULT(WINAPI*)(XPackageMountHandle, std::size_t, char*);
using CloseMount = void(WINAPI*)(XPackageMountHandle);
using UserAddAsync = HRESULT(WINAPI*)(std::uint32_t, XAsyncBlock*);
using UserAddResult = HRESULT(WINAPI*)(XAsyncBlock*, XUserHandle*);
using CloseUser = void(WINAPI*)(XUserHandle);
using GetStatus = HRESULT(WINAPI*)(XAsyncBlock*, bool);
using GetResultSize = HRESULT(WINAPI*)(XAsyncBlock*, std::size_t*);
using QueueCreate = HRESULT(WINAPI*)(std::uint32_t, std::uint32_t, XTaskQueueHandle*);
using QueueDispatch = bool(WINAPI*)(XTaskQueueHandle, std::uint32_t, std::uint32_t);
using QueueClose = void(WINAPI*)(XTaskQueueHandle);

struct Api {
    MountAsync start = nullptr;
    MountResult result = nullptr;
    MountSync mount_sync = nullptr;
    PathSize path_size = nullptr;
    PathGet path_get = nullptr;
    CloseMount close_mount = nullptr;
    UserAddAsync user_start = nullptr;
    UserAddResult user_result = nullptr;
    CloseUser close_user = nullptr;
    GetStatus get_status = nullptr;
    GetResultSize get_result_size = nullptr;
    QueueCreate queue_create = nullptr;
    QueueDispatch dispatch = nullptr;
    QueueClose queue_close = nullptr;
};

XTaskQueueHandle queue = nullptr;

struct Probe {
    int calls = 0;
    MountResult result_proc = nullptr;
    HRESULT inside_status = E_FAIL;
    XPackageMountHandle inside_mount = nullptr;
};

void CALLBACK record(XAsyncBlock* block) { ++static_cast<Probe*>(block->context)->calls; }

// Claims the result from inside the callback, as a title commonly does.
void CALLBACK claim(XAsyncBlock* block)
{
    auto* probe = static_cast<Probe*>(block->context);
    ++probe->calls;
    probe->inside_status = probe->result_proc(block, &probe->inside_mount);
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
    return find_proc(module, "XPackageMountWithUiAsync", api.start) &&
           find_proc(module, "XPackageMountWithUiResult", api.result) &&
           find_proc(module, "XPackageMount", api.mount_sync) &&
           find_proc(module, "XPackageGetMountPathSize", api.path_size) &&
           find_proc(module, "XPackageGetMountPath", api.path_get) &&
           find_proc(module, "XPackageCloseMountHandle", api.close_mount) &&
           find_proc(module, "XUserAddAsync", api.user_start) &&
           find_proc(module, "XUserAddResult", api.user_result) &&
           find_proc(module, "XUserCloseHandle", api.close_user) &&
           find_proc(module, "XAsyncGetStatus", api.get_status) &&
           find_proc(module, "XAsyncGetResultSize", api.get_result_size) &&
           find_proc(module, "XTaskQueueCreate", api.queue_create) &&
           find_proc(module, "XTaskQueueDispatch", api.dispatch) &&
           find_proc(module, "XTaskQueueCloseHandle", api.queue_close);
}

// The claimed handle is live and reports the same path as a synchronous mount
// of the same identifier; after the caller closes it, the handle is gone.
bool claimed_mount_is_live(const Api& api, XPackageMountHandle mount)
{
    if (!expect(mount != nullptr, "claimed mount handle is valid")) return false;
    std::size_t size = 0;
    if (!expect_hr(api.path_size(mount, &size), S_OK, "path size of the claimed mount") ||
        !expect(size > 1, "path size is non-trivial")) {
        return false;
    }
    std::vector<char> path(size, 0);
    if (!expect_hr(api.path_get(mount, size, path.data()), S_OK, "path of the claimed mount")) return false;
    XPackageMountHandle sync_mount = nullptr;
    if (!expect_hr(api.mount_sync(package_name, &sync_mount), S_OK, "synchronous mount of the same package")) {
        return false;
    }
    std::size_t sync_size = 0;
    std::vector<char> sync_path;
    bool ok = expect_hr(api.path_size(sync_mount, &sync_size), S_OK, "sync path size") &&
              expect(sync_size == size, "async and sync mounts report the same path size");
    if (ok) {
        sync_path.assign(sync_size, 0);
        ok = expect_hr(api.path_get(sync_mount, sync_size, sync_path.data()), S_OK, "sync path") &&
             expect(std::memcmp(path.data(), sync_path.data(), size) == 0,
                    "async and sync mounts report the same path");
    }
    api.close_mount(sync_mount);
    return ok;
}

bool closed_mount_is_gone(const Api& api, XPackageMountHandle mount)
{
    api.close_mount(mount);
    std::size_t size = 0;
    return expect_hr(api.path_size(mount, &size), E_HANDLE, "closed mount handle is invalid");
}

// E_PENDING before dispatch (and not consuming), callback once on dispatch,
// then S_OK with one handle claimed once.
bool manual_contract(const Api& api)
{
    Probe probe;
    XAsyncBlock block = make_block(&probe, record);
    XPackageMountHandle mount = reinterpret_cast<XPackageMountHandle>(&probe);
    bool ok = expect_hr(api.start(package_name, &block), S_OK, "start XPackageMountWithUiAsync");
    ok = ok && expect(probe.calls == 0, "no callback before XTaskQueueDispatch");
    std::size_t size = 0;
    ok = ok && expect_hr(api.get_status(&block, false), S_OK, "status right after completion") &&
         expect_hr(api.get_result_size(&block, &size), S_OK, "result size right after completion") &&
         expect(size == sizeof(XPackageMountHandle), "result size is one mount handle") &&
         expect(probe.calls == 0, "status and size do not run the callback");
    ok = ok && expect_hr(api.result(&block, &mount), E_PENDING, "result before dispatch") &&
         expect(mount == nullptr, "pending result leaves no handle");
    ok = ok && expect_hr(api.result(&block, &mount), E_PENDING, "early result does not consume");
    ok = ok && expect(api.dispatch(queue, port_completion, 0), "dispatch runs the callback") &&
         expect(probe.calls == 1, "callback fires once on dispatch") &&
         expect(!api.dispatch(queue, port_completion, 0), "nothing left to dispatch");
    ok = ok && expect_hr(api.result(&block, &mount), S_OK, "result after dispatch") &&
         claimed_mount_is_live(api, mount);
    XPackageMountHandle again = reinterpret_cast<XPackageMountHandle>(&probe);
    ok = ok && expect_hr(api.result(&block, &again), E_INVALIDARG, "second result is refused") &&
         expect(again == nullptr, "second result leaves no handle") &&
         expect(opaque_untouched(block), "opaque block bytes untouched");
    if (mount) ok = closed_mount_is_gone(api, mount) && ok;
    return ok;
}

bool claim_inside_callback(const Api& api)
{
    Probe probe;
    probe.result_proc = api.result;
    XAsyncBlock block = make_block(&probe, claim);
    bool ok = expect_hr(api.start(package_name, &block), S_OK, "start with a claiming callback") &&
              expect(api.dispatch(queue, port_completion, 0), "dispatch") &&
              expect(probe.calls == 1, "claiming callback ran once") &&
              expect_hr(probe.inside_status, S_OK, "result inside the callback") &&
              claimed_mount_is_live(api, probe.inside_mount);
    if (probe.inside_mount) api.close_mount(probe.inside_mount);
    return ok;
}

bool null_callback_and_reuse(const Api& api)
{
    Probe probe;
    XAsyncBlock block = make_block(&probe, nullptr);
    XPackageMountHandle mount = nullptr;
    bool ok = expect_hr(api.start(package_name, &block), S_OK, "start without a callback") &&
              expect_hr(api.start(package_name, &block), e_illegal_state_change,
                        "reusing an in-flight block") &&
              expect(!api.dispatch(queue, port_completion, 0), "no callback to dispatch") &&
              expect_hr(api.result(&block, &mount), S_OK, "result of the original start") &&
              claimed_mount_is_live(api, mount);
    if (mount) api.close_mount(mount);
    return ok;
}

// Bad identifiers, unsupported queues, a null block and null out pointer are
// refused without tracking the block or using any capacity (negative control).
bool refusals(const Api& api)
{
    Probe probe;
    bool ok = true;
    const std::array<const char*, 6> bad_identifiers{nullptr, "", "..", "a..b", "bad/name", "bad name"};
    for (const char* identifier : bad_identifiers) {
        XAsyncBlock block = make_block(&probe, record);
        XPackageMountHandle mount = reinterpret_cast<XPackageMountHandle>(&probe);
        ok = expect_hr(api.start(identifier, &block), E_INVALIDARG, "invalid package identifier") && ok;
        ok = expect_hr(api.result(&block, &mount), E_INVALIDARG, "refused block is untracked") && ok;
    }
    const std::string too_long(80, 'a');
    XAsyncBlock long_block = make_block(&probe, record);
    ok = expect_hr(api.start(too_long.c_str(), &long_block), E_INVALIDARG, "over-long identifier") && ok;
    ok = expect_hr(api.start(package_name, nullptr), E_INVALIDARG, "null block") && ok;

    XPackageMountHandle mount = nullptr;
    XAsyncBlock null_queue = make_block(&probe, record);
    null_queue.queue = nullptr;
    ok = expect_hr(api.start(package_name, &null_queue), not_implemented, "null queue") && ok;
    ok = expect_hr(api.result(&null_queue, &mount), E_INVALIDARG, "null-queue block untracked") && ok;

    int bogus = 0;
    XAsyncBlock foreign = make_block(&probe, record);
    foreign.queue = &bogus;
    ok = expect_hr(api.start(package_name, &foreign), not_implemented, "foreign queue") && ok;
    ok = expect_hr(api.result(&foreign, &mount), E_INVALIDARG, "foreign-queue block untracked") && ok;

    XAsyncBlock fine = make_block(&probe, record);
    ok = expect_hr(api.result(&fine, nullptr), E_POINTER, "null out pointer") && ok;
    ok = expect_hr(api.result(nullptr, &mount), E_INVALIDARG, "null result block") && ok;
    ok = expect(probe.calls == 0, "refusals never call back") && ok;
    ok = expect(!api.dispatch(queue, port_completion, 0), "refusals queue nothing") && ok;
    return ok;
}

// A package block is not a user-add block (and the reverse), and the pending
// start survives a wrong-kind result and an attempt to close its queue.
bool wrong_kind_and_close_guard(const Api& api)
{
    Probe probe, user_probe;
    XAsyncBlock block = make_block(&probe, record);
    XAsyncBlock user_block = make_block(&user_probe, record);
    XUserHandle user = nullptr;
    XPackageMountHandle mount = nullptr;
    bool ok = expect_hr(api.start(package_name, &block), S_OK, "start package mount") &&
              expect_hr(api.user_start(1, &user_block), S_OK, "start user add on the same queue") &&
              expect_hr(api.user_result(&block, &user), E_INVALIDARG, "user result on a package block") &&
              expect_hr(api.result(&user_block, &mount), E_INVALIDARG, "package result on a user block");
    api.queue_close(queue);  // refused while callbacks wait for dispatch
    ok = ok && expect(api.dispatch(queue, port_completion, 0) && probe.calls == 1 && user_probe.calls == 0,
                      "package callback dispatched first") &&
         expect(api.dispatch(queue, port_completion, 0) && user_probe.calls == 1,
                "user callback dispatched second") &&
         expect_hr(api.result(&block, &mount), S_OK, "package result after the guarded close") &&
         claimed_mount_is_live(api, mount) &&
         expect_hr(api.user_result(&user_block, &user), S_OK, "user result after the guarded close");
    if (mount) api.close_mount(mount);
    if (user) api.close_user(user);
    return ok;
}

// The mount table is finite: a start that cannot get a mount fails, frees its
// slot and block, and later starts work again once handles are closed.
bool capacity(const Api& api)
{
    std::vector<XPackageMountHandle> held;
    std::array<Probe, 16> probes{};
    std::array<XAsyncBlock, 16> blocks{};
    bool ok = true;
    HRESULT last = S_OK;
    std::size_t started = 0;
    for (; started < blocks.size(); ++started) {
        blocks[started] = make_block(&probes[started], record);
        last = api.start(package_name, &blocks[started]);
        if (last != S_OK) break;
        XPackageMountHandle mount = nullptr;
        while (api.dispatch(queue, port_completion, 0)) {}
        ok = expect_hr(api.result(&blocks[started], &mount), S_OK, "capacity start result") && ok;
        if (mount) held.push_back(mount);
    }
    ok = expect(started < blocks.size(), "mount table fills before the blocks run out") && ok;
    ok = expect_hr(last, E_OUTOFMEMORY, "start with no free mount") && ok;
    XPackageMountHandle none = nullptr;
    ok = expect_hr(api.result(&blocks[started], &none), E_INVALIDARG,
                   "failed start leaves its block untracked") && ok;
    for (XPackageMountHandle mount : held) api.close_mount(mount);
    Probe probe;
    XAsyncBlock again = make_block(&probe, record);
    XPackageMountHandle mount = nullptr;
    ok = expect_hr(api.start(package_name, &again), S_OK, "start after closing handles") && ok;
    while (api.dispatch(queue, port_completion, 0)) {}
    ok = expect_hr(api.result(&again, &mount), S_OK, "result after closing handles") && ok;
    if (mount) api.close_mount(mount);
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
                 manual_contract(api) && claim_inside_callback(api) && null_callback_and_reuse(api) &&
                 refusals(api) && wrong_kind_and_close_guard(api) && capacity(api);
        if (queue) api.queue_close(queue);
    }
    FreeLibrary(module);
    return passed;
}
}  // namespace

int wmain(int argc, wchar_t** argv)
{
    if (argc != 4) {
        std::printf("usage: xgameruntime_package_async_check <dll> <fixture-dir> <normal-shim-dir>\n");
        return 2;
    }
    const bool passed = run(argv[1], argv[2], argv[3]);
    std::printf("xgameruntime package async: mount-with-ui+result manual-queue contract, refusals and "
                "capacity controls %s\n",
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
