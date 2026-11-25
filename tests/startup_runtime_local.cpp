#include <windows.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <intrin.h>
#include <new>
#include <string>
#include <utility>
#include <vector>
// Everything runtime_proxy.cpp includes, before the logf/GetProcAddress macros below, so that the
// macros rename only the proxy's own calls and not declarations in the standard headers.
#include "startup_stubs/api_trace.h"
#include "startup_stubs/local_log.h"
#include "startup_stubs/save_blob_pack.h"
#include "startup_stubs/save_store.h"
#include "startup_stubs/storage_root.h"
#include "startup_stubs/storage_root_utf8.h"

namespace {
FARPROC WINAPI fixture_get_proc_address(HMODULE, LPCSTR name);
unsigned activation_logs{}, unanswered_logs{}, rejected_iid_logs{};
}

namespace x2w_local {
// Count the dedicated diagnostics separately from ordinary method tracing.
void fixture_logf(const char* format, ...)
{
    if (std::strstr(format, "-> S_OK (local runtime)")) ++activation_logs;
    if (std::strstr(format, "has no answer, E_NOTIMPL")) ++unanswered_logs;
    if (std::strstr(format, "QueryInterface refused IID, E_NOINTERFACE")) ++rejected_iid_logs;
}
}

#define GetProcAddress fixture_get_proc_address
#define logf fixture_logf
#define X2W_RUNTIME_PROXY_FIXTURE 1
#include "../src/startup_stubs/runtime_proxy.cpp"
#undef X2W_RUNTIME_PROXY_FIXTURE
#undef logf
#undef GetProcAddress

namespace {
constexpr const char* exports[]{"InitializeApiImpl", "InitializeApiImplEx", "InitializeApiImplEx2"};
// Independent fixture inputs from the observed QueryApiImpl calls.
constexpr ApiPair providers[]{
    {{0x01acd177, 0x91f9, 0x4763, {0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0}},
     {0x01acd177, 0x91f9, 0x4763, {0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0}}},
    {{0x01acd177, 0x91f9, 0x4763, {0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0}},
     {0xcef4fac0, 0x7676, 0x4a94, {0xa1, 0x19, 0x4c, 0x43, 0xf9, 0xeb, 0x5b, 0x74}}},
    {{0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6f, 0xe2}},
     {0x1bfff3af, 0xf14a, 0x40a3, {0x8e, 0x35, 0x9a, 0xda, 0x90, 0x65, 0x93, 0xf9}}},
    {{0xaf406016, 0xe850, 0x4aa8, {0xa8, 0x8d, 0x2f, 0x3d, 0xcb, 0x9d, 0xac, 0x7e}},
     {0x0b9d556e, 0xab70, 0x43f4, {0xba, 0xa0, 0x99, 0x99, 0xac, 0x4c, 0xb0, 0x40}}},
    {{0x973a344e, 0x24bf, 0x4d0f, {0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29}},
     {0x2549f142, 0x6419, 0x4a06, {0x97, 0xb5, 0x93, 0x1a, 0xab, 0x7c, 0x2f, 0x34}}},
    {{0x37e56907, 0x2f10, 0x41e8, {0xb7, 0x2f, 0x36, 0xed, 0xb1, 0x85, 0x33, 0x1a}},
     {0xbf2346b2, 0x39af, 0x4658, {0xb5, 0xea, 0x44, 0x71, 0x3c, 0x7e, 0x83, 0xb3}}},
    {{0x0dd112ac, 0x7c24, 0x448c, {0xb9, 0x2b, 0x39, 0x60, 0xfb, 0x5b, 0xd3, 0x0c}},
     {0x2d42fea5, 0xe71d, 0x4b76, {0x97, 0xcd, 0xc5, 0x0a, 0xfb, 0xb3, 0xae, 0x5d}}},
    {{0x0dd112ac, 0x7c24, 0x448c, {0xb9, 0x2b, 0x39, 0x60, 0xfb, 0x5b, 0xd3, 0x0c}},
     {0xb09d803c, 0x2414, 0x4a05, {0x82, 0xc6, 0x66, 0xdf, 0xdc, 0x9e, 0x9a, 0x44}}},
    {{0xf4faf4d4, 0x2d04, 0x4fce, {0xb3, 0xe0, 0x47, 0x4a, 0x71, 0x3a, 0x3e, 0x84}},
     {0x41a4e10c, 0x5a7e, 0x41d9, {0x8c, 0x37, 0x37, 0xbd, 0xe6, 0x2a, 0x07, 0xd6}}},
};
constexpr auto& user_pair = providers[0];
constexpr auto& gamertag_pair = providers[1];
constexpr auto& storage_fixture_pair = providers[8];
constexpr GUID fixture_iunknown{0, 0, 0, {0xc0, 0, 0, 0, 0, 0, 0, 0x46}};
constexpr GUID unsupported_iid{0xabcdef01};
constexpr GUID unknown_provider{0x12345678};
constexpr GUID async_provider{0x073b7dcb, 0x1fcf, 0x4030, {0x94,0xbe,0xe3,0xc9,0xeb,0x62,0x34,0x28}};
constexpr u64 init_failure = static_cast<unsigned>(E_ACCESSDENIED);
constexpr HRESULT query_failure = E_FAIL;
u64 init_result = init_failure;
unsigned init_calls[3]{}, provider_queries{}, async_queries{};
bool init_arguments_ok = true;
bool control = false;

template<int Index>
u64 WINAPI fixture_init(u64 a, u64 b, u64 c, u64 d)
{
    ++init_calls[Index];
    init_arguments_ok = init_arguments_ok && a == 11 && b == 22 && c == 33 && d == 44;
    return init_result;
}

const InitApi original_init[]{fixture_init<0>, fixture_init<1>, fixture_init<2>};
const InitApi wrapped_init[]{proxy_init<0>, proxy_init<1>, proxy_init<2>};

struct Block {
    void* queue{};
    void* context{};
    void (WINAPI* callback)(Block*){};
    unsigned char internal[32]{};
};
void* async_methods[11]{};
struct AsyncService { void** vtable; } async_service{async_methods};
Block* expected_block{};
bool pending{}, async_arguments_ok = true;
unsigned begins{}, completions{}, releases{};
HRESULT completion_status = E_PENDING;
size_t completion_size = 1;

HRESULT WINAPI fixture_begin(void* self, void* block, void*, const void* identity,
                             const char* name, decltype(&submit_provider) provider)
{
    ++begins;
    if (self != &async_service || !expected_block || block != expected_block || pending ||
        !identity || !name || !provider) {
        async_arguments_ok = false;
        return E_INVALIDARG;
    }
    const auto result = provider(0, nullptr);
    pending = SUCCEEDED(result);
    return result;
}

void WINAPI fixture_complete(void* self, void* block, HRESULT status, size_t size)
{
    ++completions;
    if (self != &async_service || !pending || block != expected_block) {
        async_arguments_ok = false;
        return;
    }
    pending = false;
    completion_status = status;
    completion_size = size;
    auto* completed = static_cast<Block*>(block);
    if (completed->callback) completed->callback(completed);
}

ULONG WINAPI fixture_release(void* self)
{
    ++releases;
    async_arguments_ok = async_arguments_ok && self == &async_service && !pending;
    return 1;
}

HRESULT WINAPI fixture_query(const GUID* provider, const GUID* iid, void** out)
{
    if ((out && !x2w_local::writable(out, sizeof(*out))) ||
        (provider && !x2w_local::readable(provider, sizeof(*provider))) ||
        (iid && !x2w_local::readable(iid, sizeof(*iid)))) return E_POINTER;
    // Leave a non-null output on failure to check whether each phase preserves or clears it.
    if (out) *out = &async_service;
    if (provider && *provider == async_provider) {
        ++async_queries;
        if (!out || !iid || *iid != async_provider) return query_failure;
        return S_OK;
    }
    ++provider_queries;
    return query_failure;
}

FARPROC WINAPI fixture_get_proc_address(HMODULE, LPCSTR name)
{
    for (int i = 0; i < 3; ++i)
        if (std::strcmp(name, exports[i]) == 0) return reinterpret_cast<FARPROC>(original_init[i]);
    if (std::strcmp(name, "QueryApiImpl") == 0) return reinterpret_cast<FARPROC>(&fixture_query);
    if (std::strcmp(name, "UnrelatedExport") == 0) return reinterpret_cast<FARPROC>(original_init[0]);
    return nullptr;
}

template<class T> u64 address(T* pointer) { return reinterpret_cast<u64>(pointer); }

u64 invoke(Proxy* proxy, int slot, u64 a = 0, u64 b = 0)
{
    return reinterpret_cast<Method>(proxy->vtable[slot])(proxy, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}

struct Completion { Proxy* user; unsigned callbacks{}; u64 result{init_failure}, handle{}; };

void WINAPI on_complete(Block* block)
{
    auto& seen = *static_cast<Completion*>(block->context);
    ++seen.callbacks;
    seen.result = invoke(seen.user, 8, address(block), address(&seen.handle));
}

bool set_switch(const char* name, bool enabled)
{
    const char* value = enabled ? "1" : "0";
    return _putenv_s(name, value) == 0 && SetEnvironmentVariableA(name, value) &&
           x2w_local::env_on(name) == enabled;
}

bool query_fails(QueryApi query, const GUID& provider, const GUID& iid, void* expected_output)
{
    void* object = nullptr;
    const unsigned previous = provider_queries;
    const int published = provider_count;
    return query(&provider, &iid, &object) == query_failure && object == expected_output &&
           provider_queries == previous + 1 && provider_count == published;
}

bool query_fails(QueryApi query, const ApiPair& pair, void* expected_output)
{
    return query_fails(query, pair.provider, pair.iid, expected_output);
}

const char* run()
{
    if (!set_switch("X2W_RUNTIME_PROXY", true) || !set_switch("X2W_TRACE_API", false) ||
        !set_switch("X2W_FAKE_USER", true) || !set_switch("X2W_RUNTIME_LOCAL", false))
        return "could not set the test environment";

    // Initialize the dispatch table as process attach does, without patching the test executable.
    fill_slots(std::make_index_sequence<max_slots>{});
    const auto query_addr = hooked_get_proc_address(nullptr, "QueryApiImpl");
    if (query_addr != reinterpret_cast<FARPROC>(&proxy_query)) return "switch off changed the QueryApiImpl hook";
    const auto query = reinterpret_cast<QueryApi>(query_addr);
    for (int i = 0; i < 3; ++i) {
        const auto entry = reinterpret_cast<InitApi>(hooked_get_proc_address(nullptr, exports[i]));
        if (entry != original_init[i] || entry(11, 22, 33, 44) != init_failure || init_calls[i] != 1)
            return "switch off changed an initialization export or its real failure";
    }
    if (local_active || activation_logs || !query_fails(query, user_pair, &async_service))
        return "switch off activated the fallback";

    if (!set_switch("X2W_RUNTIME_LOCAL", !control)) return "could not enable local runtime";
    if (hooked_get_proc_address(nullptr, "QueryApiImpl") != query_addr ||
        hooked_get_proc_address(nullptr, "UnrelatedExport") != fixture_get_proc_address(nullptr, "UnrelatedExport") ||
        hooked_get_proc_address(nullptr, "MissingExport"))
        return "local runtime changed a non-initialization export";

    for (int i = 0; i < 3; ++i) {
        // Give every export its own attempt to trigger the process-wide latch.
        InterlockedExchange(&local_active, 0);
        const auto entry = reinterpret_cast<InitApi>(hooked_get_proc_address(nullptr, exports[i]));
        if (entry != wrapped_init[i]) {
            std::printf("Initialization export: %s\n", exports[i]);
            return "local runtime did not return the expected initialization wrapper";
        }
        const unsigned previous = init_calls[i];
        init_result = S_FALSE;
        if (entry(11, 22, 33, 44) != S_FALSE || init_calls[i] != previous + 1 || local_active ||
            activation_logs != static_cast<unsigned>(i) || !query_fails(query, user_pair, &async_service))
            return "successful initialization did not pass through or activated the fallback";
        init_result = init_failure;
        if (entry(11, 22, 33, 44) != S_OK || init_calls[i] != previous + 2 || !local_active ||
            activation_logs != static_cast<unsigned>(i + 1))
            return "failed initialization did not activate and log the fallback";
        if (entry(11, 22, 33, 44) != S_OK || init_calls[i] != previous + 2 ||
            activation_logs != static_cast<unsigned>(i + 1) || !init_arguments_ok)
            return "fallback retried the real export, logged again, or changed initialization arguments";
    }

    for (int i = 0; i < 3; ++i) {
        const auto entry = reinterpret_cast<InitApi>(hooked_get_proc_address(nullptr, exports[i]));
        if (entry != wrapped_init[i] || entry(11, 22, 33, 44) != S_OK || init_calls[i] != 3 || activation_logs != 3)
            return "active fallback did not short-circuit all initialization exports";
    }

    // Keep the latch set so this control independently verifies the fake-user gate in QueryApiImpl.
    if (!set_switch("X2W_FAKE_USER", false)) return "could not disable fake user";
    for (int i = 0; i < 3; ++i)
        if (reinterpret_cast<InitApi>(hooked_get_proc_address(nullptr, exports[i])) != original_init[i])
            return "fake user off still wrapped an initialization export";
    for (const auto& pair : providers)
        if (!query_fails(query, pair, &async_service)) return "fake user off still returned a local object";
    if (!set_switch("X2W_FAKE_USER", true)) return "could not re-enable fake user";
    void* inaccessible = VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
    if (!inaccessible) return "could not allocate the inaccessible query fixture";
    auto* no_guid = static_cast<const GUID*>(inaccessible);
    void* rejected = &async_service;
    const int published = provider_count;
    const auto bad_provider = query(no_guid, &user_pair.iid, &rejected);
    const bool provider_cleared = rejected == nullptr;
    rejected = &async_service;
    const auto bad_iid = query(&storage_fixture_pair.provider, no_guid, &rejected);
    const auto bad_output = query(&storage_fixture_pair.provider, &storage_fixture_pair.iid,
                                  static_cast<void**>(inaccessible));
    VirtualFree(inaccessible, 0, MEM_RELEASE);
    if (bad_provider != E_POINTER || !provider_cleared || bad_iid != E_POINTER || rejected ||
        bad_output != E_POINTER || provider_count != published)
        return "invalid query arguments changed the real failure or published a local provider";
    if (!query_fails(query, unknown_provider, unknown_provider, nullptr)) return "unlisted provider did not keep the real failure";
    if (!query_fails(query, user_pair.provider, unsupported_iid, nullptr))
        return "known provider with an unknown IID returned a local object";
    if (!query_fails(query, user_pair.provider, providers[4].iid, nullptr))
        return "user provider accepted the title IID";
    GUID colliding_storage_provider = storage_fixture_pair.provider;
    colliding_storage_provider.Data4[7] ^= 1;
    if (!query_fails(query, colliding_storage_provider, storage_fixture_pair.iid, nullptr))
        return "storage provider with matching Data1 returned a local object";
    GUID colliding_provider = user_pair.provider;
    colliding_provider.Data4[7] ^= 1;
    if (!query_fails(query, colliding_provider, user_pair.iid, nullptr))
        return "provider with matching Data1 returned a local object";
    GUID colliding_iid = gamertag_pair.iid;
    colliding_iid.Data4[7] ^= 1;
    if (!query_fails(query, user_pair.provider, colliding_iid, nullptr) ||
        !query_fails(query, storage_fixture_pair.provider, unsupported_iid, nullptr))
        return "gamertag or storage accepted an unlisted pair";

    Proxy* user = nullptr;
    Proxy* first_store = nullptr;
    for (const auto& pair : providers) {
        void* object = nullptr;
        const unsigned previous = provider_queries;
        if (query(&pair.provider, &pair.iid, &object) != S_OK || !object || provider_queries != previous + 1)
            return "listed provider did not fall back after a real query failure";
        if (&pair == &storage_fixture_pair) {
            auto* storage = static_cast<StorageObject*>(object);
            const auto qi = reinterpret_cast<decltype(&storage_query)>(storage->vtable[0]);
            void* queried = &async_service;
            if (qi(storage, &pair.iid, &queried) != S_OK || queried != storage)
                return "storage QueryInterface did not return its interface";
            queried = &async_service;
            if (qi(storage, &fixture_iunknown, &queried) != S_OK || queried != storage)
                return "storage QueryInterface did not return its IUnknown identity";
            queried = &async_service;
            if (qi(storage, &unsupported_iid, &queried) != E_NOINTERFACE || queried)
                return "storage QueryInterface did not refuse an unsupported IID";
            if (qi(storage, &pair.iid, nullptr) != E_POINTER)
                return "storage QueryInterface accepted a null out pointer";
            continue;
        }
        auto* proxy = static_cast<Proxy*>(object);
        if (proxy->real || proxy->vtable != slot_table) return "local provider did not return a null-real proxy";
        if (&pair == &user_pair) user = proxy;
        if (&pair == &gamertag_pair) {
            if (proxy == user) return "gamertag reused the user object";
            void* queried = &async_service;
            if (invoke(proxy, 0, address(&user_pair.iid), address(&queried)) !=
                    static_cast<unsigned>(E_NOINTERFACE) || queried)
                return "gamertag QueryInterface did not refuse the user IID";
        }
        if (&pair == &providers[6]) first_store = proxy;
        if (&pair == &providers[7]) {
            void* queried = &async_service;
            if (proxy == first_store ||
                invoke(first_store, 0, address(&providers[6].iid), address(&queried)) != S_OK ||
                queried != first_store)
                return "the second store IID changed the first store object's interface";
            queried = &async_service;
            if (invoke(first_store, 0, address(&pair.iid), address(&queried)) !=
                    static_cast<unsigned>(E_NOINTERFACE) || queried)
                return "first store QueryInterface did not refuse the second store IID";
        }
        void* queried = &async_service;
        if (invoke(proxy, 0, address(&pair.iid), address(&queried)) != S_OK || queried != proxy)
            return "local QueryInterface did not return its interface";
        queried = &async_service;
        if (invoke(proxy, 0, address(&fixture_iunknown), address(&queried)) != S_OK || queried != proxy)
            return "local QueryInterface did not return its IUnknown identity";
        if (invoke(proxy, 1) != 1 || invoke(proxy, 2) != 1)
            return "local AddRef or Release result was wrong";
    }
    const unsigned previous_rejections = rejected_iid_logs;
    for (const GUID* iid : {&unsupported_iid, &gamertag_pair.iid}) {
        void* queried = &async_service;
        if (invoke(user, 0, address(iid), address(&queried)) != static_cast<unsigned>(E_NOINTERFACE) || queried ||
            rejected_iid_logs != previous_rejections + 1)
            return "user QueryInterface did not refuse an unsupported IID and log once";
    }
    constexpr int unanswered_slot = 95;
    if (invoke(user, unanswered_slot) != static_cast<unsigned>(E_NOTIMPL) || unanswered_logs != 1 ||
        invoke(user, unanswered_slot) != static_cast<unsigned>(E_NOTIMPL) || unanswered_logs != 1)
        return "unanswered local slot did not return E_NOTIMPL and log once";

    async_methods[2] = reinterpret_cast<void*>(&fixture_release);
    async_methods[7] = reinterpret_cast<void*>(&fixture_begin);
    async_methods[10] = reinterpret_cast<void*>(&fixture_complete);
    Completion seen{user};
    Block block{};
    block.context = &seen;
    block.callback = on_complete;
    expected_block = &block;
    if (invoke(user, 7, 0, address(&block)) != S_OK || async_queries != 1 ||
        begins != 1 || completions != 1 || releases != 1 || !async_arguments_ok || pending ||
        completion_status != S_OK || completion_size != 0 || seen.callbacks != 1)
        return "XUserAddAsync did not complete the caller's block through XAsync";
    if (seen.result != S_OK || seen.handle != local_user_handle)
        return "XUserAddResult did not return the local handle from the completion callback";

    // Blocks completed on behalf of a local answer (game save, package) go through the same service
    // when the user object is local, and are tracked so their result calls stay local.
    Block other{};
    expected_block = &other;
    if (complete_async(address(&other)) != S_OK || begins != 2 || completions != 2 || releases != 2 ||
        !async_arguments_ok || pending || completion_status != S_OK || !tracked(address(&other)))
        return "complete_async did not complete a block through XAsync for a local user object";
    if (invoke(user, 0, address(&user_pair.iid), 0) != static_cast<unsigned>(E_POINTER))
        return "local QueryInterface accepted a null out pointer";
    return nullptr;
}
}

int main(int argc, char** argv)
{
    // --control: the local runtime switch stays off where the test enables it, so the run must fail.
    control = argc > 1 && std::strcmp(argv[1], "--control") == 0;
    if (const char* reason = run()) {
        std::printf("FAIL: %s\n", reason);
        return 1;
    }
    std::puts("PASS");
    return 0;
}
