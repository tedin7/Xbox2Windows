// Synthetic title-level check for XGameSaveEnumerateBlobInfo and
// XGameSaveEnumerateBlobInfoByName in the fixture-only xgameruntime.dll. The DLL
// is loaded from its explicit fixture path. The callback signature follows the
// public XGameSave.h declaration: return false to stop the enumeration.
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

constexpr std::uint64_t synthetic_user_id = 0x0009000000000001ull;
constexpr HRESULT e_gs_handle_expired = static_cast<HRESULT>(0x8083000DL);

namespace xgameruntime_shim {
struct XUserHandleStorage;
struct XGameSaveProviderStorage;
struct XGameSaveContainerStorage;
struct XGameSaveUpdateStorage;
}
using XUserHandle = xgameruntime_shim::XUserHandleStorage*;
using XGameSaveProviderHandle = xgameruntime_shim::XGameSaveProviderStorage*;
using XGameSaveContainerHandle = xgameruntime_shim::XGameSaveContainerStorage*;
using XGameSaveUpdateHandle = xgameruntime_shim::XGameSaveUpdateStorage*;

struct XGameSaveBlobInfo {
    const char* name;
    std::uint32_t size;
};
using XGameSaveBlobInfoCallback = bool CALLBACK(const XGameSaveBlobInfo*, void*);

using FindUserById = HRESULT(WINAPI*)(std::uint64_t, XUserHandle*);
using CloseUserHandle = void(WINAPI*)(XUserHandle);
using InitializeProvider = HRESULT(WINAPI*)(XUserHandle, const char*, bool, XGameSaveProviderHandle*);
using CreateContainer = HRESULT(WINAPI*)(XGameSaveProviderHandle, const char*, XGameSaveContainerHandle*);
using CreateUpdate = HRESULT(WINAPI*)(XGameSaveContainerHandle, const char*, XGameSaveUpdateHandle*);
using SubmitBlobWrite = HRESULT(WINAPI*)(XGameSaveUpdateHandle, const char*, const std::uint8_t*, std::size_t);
using SubmitUpdate = HRESULT(WINAPI*)(XGameSaveUpdateHandle);
using EnumerateBlobs = HRESULT(WINAPI*)(XGameSaveContainerHandle, void*, XGameSaveBlobInfoCallback*);
using EnumerateBlobsByName =
    HRESULT(WINAPI*)(XGameSaveContainerHandle, const char*, void*, XGameSaveBlobInfoCallback*);
using CloseProvider = void(WINAPI*)(XGameSaveProviderHandle);
using CloseContainer = void(WINAPI*)(XGameSaveContainerHandle);
using CloseUpdate = void(WINAPI*)(XGameSaveUpdateHandle);

struct Api {
    FindUserById find_user = nullptr;
    CloseUserHandle close_user = nullptr;
    InitializeProvider initialize = nullptr;
    CreateContainer create_container = nullptr;
    CreateUpdate create_update = nullptr;
    SubmitBlobWrite write = nullptr;
    SubmitUpdate submit = nullptr;
    EnumerateBlobs enumerate = nullptr;
    EnumerateBlobsByName enumerate_by_name = nullptr;
    CloseProvider close_provider = nullptr;
    CloseContainer close_container = nullptr;
    CloseUpdate close_update = nullptr;
};

template <typename Proc>
bool find_proc(HMODULE module, const char* name, Proc& proc)
{
    const auto found = GetProcAddress(module, name);
    if (!found) {
        std::printf("FAIL: missing export %s\n", name);
        return false;
    }
    proc = reinterpret_cast<Proc>(found);
    return true;
}

bool load_api(HMODULE module, Api& api)
{
    return find_proc(module, "XUserFindUserById", api.find_user) &&
           find_proc(module, "XUserCloseHandle", api.close_user) &&
           find_proc(module, "XGameSaveInitializeProvider", api.initialize) &&
           find_proc(module, "XGameSaveCreateContainer", api.create_container) &&
           find_proc(module, "XGameSaveCreateUpdate", api.create_update) &&
           find_proc(module, "XGameSaveSubmitBlobWrite", api.write) &&
           find_proc(module, "XGameSaveSubmitUpdate", api.submit) &&
           find_proc(module, "XGameSaveEnumerateBlobInfo", api.enumerate) &&
           find_proc(module, "XGameSaveEnumerateBlobInfoByName", api.enumerate_by_name) &&
           find_proc(module, "XGameSaveCloseProvider", api.close_provider) &&
           find_proc(module, "XGameSaveCloseContainer", api.close_container) &&
           find_proc(module, "XGameSaveCloseUpdate", api.close_update);
}

struct Seen {
    std::vector<std::string> names;
    std::vector<std::uint32_t> sizes;
    std::size_t stop_after = 0;  // 0 = never stop early
    const Api* api = nullptr;
    XGameSaveContainerHandle reenter = nullptr;
    HRESULT reenter_result = E_FAIL;
};

bool CALLBACK collect(const XGameSaveBlobInfo* info, void* context)
{
    auto* seen = static_cast<Seen*>(context);
    seen->names.emplace_back(info->name);
    seen->sizes.push_back(info->size);
    if (seen->reenter) {
        // The callback runs outside the shim's lock, so it may call back in.
        seen->reenter_result = seen->api->enumerate(
            seen->reenter, nullptr, [](const XGameSaveBlobInfo*, void*) -> bool { return false; });
        seen->reenter = nullptr;
    }
    return seen->stop_after == 0 || seen->names.size() < seen->stop_after;
}

bool fail(const char* what)
{
    std::printf("FAIL: %s\n", what);
    return false;
}

bool write_blob(const Api& api, XGameSaveContainerHandle container, const char* blob,
                std::size_t size)
{
    XGameSaveUpdateHandle update = nullptr;
    const std::array<std::uint8_t, 16> payload{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const bool ok = SUCCEEDED(api.create_update(container, "enum", &update)) && update &&
                    SUCCEEDED(api.write(update, blob, payload.data(), size)) &&
                    SUCCEEDED(api.submit(update));
    if (update) api.close_update(update);
    return ok;
}

bool run_checks(const Api& api)
{
    XUserHandle user = nullptr;
    XGameSaveProviderHandle provider = nullptr;
    XGameSaveContainerHandle container = nullptr;
    XGameSaveContainerHandle other = nullptr;
    if (FAILED(api.find_user(synthetic_user_id, &user)) || !user ||
        FAILED(api.initialize(user, "synthetic-scid", false, &provider)) || !provider ||
        FAILED(api.create_container(provider, "enum-a", &container)) || !container ||
        FAILED(api.create_container(provider, "enum-b", &other)) || !other) {
        return fail("could not open provider and containers");
    }

    // Negative: arguments and empty container.
    Seen seen;
    if (api.enumerate(nullptr, &seen, collect) != E_INVALIDARG) return fail("null container");
    if (api.enumerate(container, &seen, nullptr) != E_INVALIDARG) return fail("null callback");
    if (api.enumerate_by_name(container, nullptr, &seen, collect) != E_INVALIDARG) {
        return fail("null prefix");
    }
    if (api.enumerate_by_name(container, "a", &seen, nullptr) != E_INVALIDARG) {
        return fail("by-name null callback");
    }
    if (FAILED(api.enumerate(container, &seen, collect)) || !seen.names.empty()) {
        return fail("an empty container must succeed without calling back");
    }

    // Three blobs with distinct sizes, one in the sibling container.
    if (!write_blob(api, container, "alpha.bin", 6) || !write_blob(api, container, "alpha2.bin", 3) ||
        !write_blob(api, container, "beta.bin", 16) || !write_blob(api, other, "alpha.bin", 1)) {
        return fail("could not write blobs");
    }

    seen = Seen{};
    if (FAILED(api.enumerate(container, &seen, collect)) || seen.names.size() != 3 ||
        seen.names[0] != "alpha.bin" || seen.sizes[0] != 6 || seen.names[1] != "alpha2.bin" ||
        seen.sizes[1] != 3 || seen.names[2] != "beta.bin" || seen.sizes[2] != 16) {
        return fail("full enumeration did not report the three committed blobs");
    }

    seen = Seen{};
    if (FAILED(api.enumerate_by_name(container, "alpha", &seen, collect)) ||
        seen.names.size() != 2 || seen.names[0] != "alpha.bin" || seen.names[1] != "alpha2.bin") {
        return fail("prefix enumeration");
    }
    seen = Seen{};
    if (FAILED(api.enumerate_by_name(container, "", &seen, collect)) || seen.names.size() != 3) {
        return fail("an empty prefix must match every blob");
    }
    seen = Seen{};
    if (FAILED(api.enumerate_by_name(container, "Alpha", &seen, collect)) || !seen.names.empty()) {
        return fail("prefix matching is case sensitive");
    }
    seen = Seen{};
    if (FAILED(api.enumerate_by_name(container, "alpha.bin.longer", &seen, collect)) ||
        !seen.names.empty()) {
        return fail("a prefix longer than the name must not match");
    }
    const std::string too_long(200, 'x');
    seen = Seen{};
    if (FAILED(api.enumerate_by_name(container, too_long.c_str(), &seen, collect)) ||
        !seen.names.empty()) {
        return fail("an over-capacity prefix must succeed with no match");
    }

    // The callback can stop early and can re-enter the shim.
    seen = Seen{};
    seen.stop_after = 1;
    if (FAILED(api.enumerate(container, &seen, collect)) || seen.names.size() != 1) {
        return fail("returning false must stop the enumeration with S_OK");
    }
    seen = Seen{};
    seen.api = &api;
    seen.reenter = container;
    if (FAILED(api.enumerate(container, &seen, collect)) || FAILED(seen.reenter_result) ||
        seen.names.size() != 3) {
        return fail("a callback must be able to call back into the shim");
    }

    // A sibling container only sees its own blob.
    seen = Seen{};
    if (FAILED(api.enumerate(other, &seen, collect)) || seen.names.size() != 1 ||
        seen.sizes[0] != 1) {
        return fail("sibling container isolation");
    }

    // Closed container handle and closed provider.
    api.close_container(other);
    seen = Seen{};
    if (api.enumerate(other, &seen, collect) != E_HANDLE || !seen.names.empty()) {
        return fail("a closed container handle must return E_HANDLE");
    }
    api.close_provider(provider);
    if (api.enumerate(container, &seen, collect) != E_HANDLE &&
        api.enumerate(container, &seen, collect) != e_gs_handle_expired) {
        return fail("a container of a closed provider must be refused");
    }
    if (!seen.names.empty()) return fail("callback ran for a refused enumeration");
    api.close_user(user);
    return true;
}
}  // namespace

int wmain(int argc, wchar_t** argv)
{
    if (argc != 2) {
        std::printf("usage: xgameruntime_gamesave_enumerate_check <xgameruntime.dll>\n");
        return 2;
    }
    const fs::path dll = fs::absolute(argv[1]);
    HMODULE module = LoadLibraryW(dll.c_str());
    if (!module) {
        std::printf("FAIL: LoadLibraryW(%ls) error=%lu\n", dll.c_str(), GetLastError());
        return 1;
    }
    Api api;
    const bool passed = load_api(module, api) && run_checks(api);
    FreeLibrary(module);
    std::printf(passed ? "PASS\n" : "FAIL\n");
    return passed ? 0 : 1;
}
