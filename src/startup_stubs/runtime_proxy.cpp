// Experimental (docs/startup-runtime.md): proxy between the title and the system gaming runtime.
//
// The title's static runtime library loads the system xgameruntime.dll and asks it for API objects
// with QueryApiImpl(provider GUID, interface GUID, &object). With X2W_RUNTIME_PROXY=1 this file (built
// into the GameInput stand-in, which the title imports) rewrites GetProcAddress and a direct
// xgameruntime.dll!QueryApiImpl import on the main EXE and on DLLs that share the EXE's directory
// (package helpers). Dynamic QueryApiImpl is wrapped only when the provider module's basename is
// xgameruntime.dll. The pre-link load notification only requests another scan. A worker scans
// and rewrites linked modules outside the notification, keeping each module live while it is patched.
// Enabled proxies remain loaded until process exit. Delay-import IAT slots are not
// rewritten. The wrapper hands out proxy objects whose methods log their first calls ("x2w-runtime")
// and forward to the real object.
//
// X2W_FAKE_USER=1 additionally answers the user API for one local user (see fake_user below)
// and, while the proxy is on, XStoreCreateContext for that user (see area_call).
#include "startup_stubs/api_trace.h"
#include "startup_stubs/console_mode.h"
#include "startup_stubs/local_log.h"
#include "startup_stubs/runtime_gpa_patch.h"
#include "startup_stubs/save_blob_pack.h"
#include "startup_stubs/save_store.h"
#include "startup_stubs/save_write_log.h"
#include "startup_stubs/storage_root.h"
#include "startup_stubs/storage_root_utf8.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <intrin.h>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <psapi.h>

namespace {
using u64 = std::uint64_t;
using Method = u64(__stdcall*)(void*, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);
constexpr int max_providers = 24, max_slots = 96;

struct Proxy {
    void** vtable;
    void* real;
    int provider;
    const GUID* local_iid = nullptr;
    LONG rejected_iid = 0;
    bool public_save_interface = false;
};
GUID provider_ids[max_providers];
Proxy* proxies[max_providers];
int provider_count;
LONG call_counts[max_providers][max_slots];
SRWLOCK proxy_lock = SRWLOCK_INIT;
void* slot_table[max_slots];

using QueryApi = HRESULT(__stdcall*)(const GUID*, const GUID*, void**);
// Written from the GPA hook and the post-link scan; first non-null wins.
std::atomic<QueryApi> real_query{nullptr};

bool keep_forward_target(FARPROC target) noexcept
{
#if defined(X2W_RUNTIME_PROXY_FIXTURE)
    return target != nullptr;
#else
    HMODULE module = nullptr;
    return target && GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                                        reinterpret_cast<LPCWSTR>(target), &module);
#endif
}

bool remember_real_query(QueryApi query) noexcept
{
    QueryApi expected = real_query.load(std::memory_order_acquire);
    if (expected) return expected == query;
    if (!keep_forward_target(reinterpret_cast<FARPROC>(query))) return false;
    return real_query.compare_exchange_strong(expected, query, std::memory_order_acq_rel) || expected == query;
}

QueryApi load_real_query() noexcept { return real_query.load(std::memory_order_acquire); }

// ---- local user (X2W_FAKE_USER=1) ----------------------------------------------------------
// The user API object is the provider whose GUID starts 01acd177 (seen: called at start-up and when
// the sign-in button is pressed). Slots are filled in from observed calls; see user_call below.
constexpr unsigned long user_provider_data1 = 0x01acd177;
constexpr u64 local_user_handle = 0x58325755'00000001ull;  // not a pointer the runtime knows
constexpr u64 local_user_id = 0x0009000012345678ull;
// XStoreImpl. Method 3 is XStoreCreateContext; the handle below is the offline context it writes.
constexpr GUID store_provider_id{0x0dd112ac, 0x7c24, 0x448c, {0xb9, 0x2b, 0x39, 0x60, 0xfb, 0x5b, 0xd3, 0x0c}};
constexpr u64 store_context_handle = 0x58325753'00000001ull;  // not a pointer the runtime knows
bool fake_user() noexcept { return x2w_local::env_on("X2W_FAKE_USER"); }
// X2W_RUNTIME_LOCAL=1 (with X2W_RUNTIME_PROXY and X2W_FAKE_USER): when the system runtime cannot
// initialise (no Gaming Services), report the initialisation as done and hand out local objects for
// the providers the fake layer answers. Unanswered methods of those objects fail with E_NOTIMPL.
bool runtime_local() noexcept { return x2w_local::env_on("X2W_RUNTIME_LOCAL"); }
volatile LONG local_active;  // set once a real initialisation failed and was answered locally
using InitApi = u64(__stdcall*)(u64, u64, u64, u64);
InitApi real_init[3];
constexpr const char* init_names[3] = {"InitializeApiImpl", "InitializeApiImplEx", "InitializeApiImplEx2"};
int user_provider = -1, save_provider = -1;
// The user provider also hands out a second interface (iid cef4fac0...) whose slot 3 is
// GetGamertag(handle, component, size, buffer, &used). It is proxied as an entry of its own (shown as
// provider cef4fac0), not as the main user object, whose slot 3 duplicates a handle.
int gamertag_provider = -1;
constexpr unsigned long gamertag_iid_data1 = 0xcef4fac0;

struct ApiPair { GUID provider, iid; };
constexpr ApiPair local_pairs[] = {
    {{0x01acd177, 0x91f9, 0x4763, {0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0}},
     {0x01acd177, 0x91f9, 0x4763, {0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0}}},
    {{0x01acd177, 0x91f9, 0x4763, {0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0}},
     {0xcef4fac0, 0x7676, 0x4a94, {0xa1, 0x19, 0x4c, 0x43, 0xf9, 0xeb, 0x5b, 0x74}}},
    {store_provider_id,
     {0x2d42fea5, 0xe71d, 0x4b76, {0x97, 0xcd, 0xc5, 0x0a, 0xfb, 0xb3, 0xae, 0x5d}}},
    {store_provider_id,
     {0xb09d803c, 0x2414, 0x4a05, {0x82, 0xc6, 0x66, 0xdf, 0xdc, 0x9e, 0x9a, 0x44}}},
    {{0x37e56907, 0x2f10, 0x41e8, {0xb7, 0x2f, 0x36, 0xed, 0xb1, 0x85, 0x33, 0x1a}},
     {0xbf2346b2, 0x39af, 0x4658, {0xb5, 0xea, 0x44, 0x71, 0x3c, 0x7e, 0x83, 0xb3}}},
    {{0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6f, 0xe2}},
     {0x1bfff3af, 0xf14a, 0x40a3, {0x8e, 0x35, 0x9a, 0xda, 0x90, 0x65, 0x93, 0xf9}}},
    {{0x973a344e, 0x24bf, 0x4d0f, {0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29}},
     {0x2549f142, 0x6419, 0x4a06, {0x97, 0xb5, 0x93, 0x1a, 0xab, 0x7c, 0x2f, 0x34}}},
    {{0xaf406016, 0xe850, 0x4aa8, {0xa8, 0x8d, 0x2f, 0x3d, 0xcb, 0x9d, 0xac, 0x7e}},
     {0x0b9d556e, 0xab70, 0x43f4, {0xba, 0xa0, 0x99, 0x99, 0xac, 0x4c, 0xb0, 0x40}}},
};
// Storage has its own object and remains available under the fake-user gate alone.
constexpr ApiPair storage_pair{
    {0xf4faf4d4, 0x2d04, 0x4fce, {0xb3, 0xe0, 0x47, 0x4a, 0x71, 0x3a, 0x3e, 0x84}},
    {0x41a4e10c, 0x5a7e, 0x41d9, {0x8c, 0x37, 0x37, 0xbd, 0xe6, 0x2a, 0x07, 0xd6}}
};
constexpr GUID iunknown_iid{0x00000000, 0x0000, 0x0000, {0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}};  // IID_IUnknown

bool matches_pair(const ApiPair& pair, const GUID* provider, const GUID* iid)
{
    return provider && iid && IsEqualGUID(pair.provider, *provider) && IsEqualGUID(pair.iid, *iid);
}

HRESULT local_query_interface(void* self, unsigned long provider, const GUID* implemented,
                              const GUID* iid, void** out, LONG& rejected)
{
    if (!out || !x2w_local::writable(out, sizeof(*out))) return E_POINTER;
    if (iid && x2w_local::readable(iid, sizeof(*iid)) &&
        (IsEqualGUID(*iid, iunknown_iid) || (implemented && IsEqualGUID(*iid, *implemented)))) {
        *out = self;
        return S_OK;
    }
    *out = nullptr;
    if (InterlockedCompareExchange(&rejected, 1, 0) == 0)
        x2w_local::logf("x2w-runtime local: %08lx::QueryInterface refused IID, E_NOINTERFACE", provider);
    return E_NOINTERFACE;
}
// Async blocks completed on behalf of a local answer (see complete_async): calls that mention them
// are always logged, and the borrowed operation's own failure code is not passed on for them.
u64 tracked_blocks[64];
volatile LONG tracked_count, tracked_logged;
constexpr u64 borrowed_failure = 0x89245110;
bool tracked(u64 value)
{
    if (!value) return false;
    for (LONG i = 0; i < tracked_count && i < 64; ++i)
        if (tracked_blocks[i] == value) return true;
    return false;
}
bool user_call(int slot, u64* args, u64* result);
bool save_call(int slot, u64* args, u64* result);
bool save_read_call(int slot, u64* args, u64* result);
bool area_call(const GUID& provider_id, int slot, u64* args, u64* result);
u64 complete_async(u64 async_block);
u64 local_user_add(u64 async_block);
void* storage_object();
const char* storage_path();
constexpr unsigned long save_provider_data1 = 0x704c3f58;  // hypothesis: the game-save API (first seen: slot 4 with the user handle)

// X2W_TRACE_API=1: one line per provider call with the return code, whether the answer came from the
// stand-in or the real runtime, and which declared output argument it wrote. Only declared outputs are
// read: arguments are not probed as pointers, because a handle or size that happens to look like an
// address would be read as if it were one. A slot with no declared contract reports "out=?" instead.
bool trace_api() noexcept
{
    static const bool on = x2w_local::env_on("X2W_TRACE_API");
    return on;
}

// The contract table and the decision about what was observed live in api_trace.h so the decision can
// be checked by a source-built control without a running title.
using x2w_api_trace::declared;
using x2w_api_trace::DeclaredOut;
using x2w_api_trace::max_outputs;
using x2w_api_trace::max_snapshot;
using x2w_api_trace::no_argument;


template <int N>
__declspec(noinline) u64 forward(Proxy* self, u64 a, u64 b, u64 c, u64 s0, u64 s1, u64 s2, u64 s3, u64 s4, u64 s5, u64 s6, u64 s7)
{
    // call_counts also feeds the periodic call-rate summary, so it keeps counting; a count that has
    // passed 2^31 is negative and must not log again.
    const LONG call = N >= 3 ? InterlockedIncrement(&call_counts[self->provider][N]) : 0;
    const bool log_it = call > 0 && call <= 6;
    u64 args[11] = {a, b, c, s0, s1, s2, s3, s4, s5, s6, s7};
    // Declared outputs are snapshotted so the write is seen rather than assumed; nothing else is read. The
    // snapshot is bounded by the declared width, so a one-byte flag, a DWORD and a caller-sized buffer
    // Declared outputs are snapshotted so a write can be observed rather than assumed; nothing else is
    // read. Each snapshot is bounded by the declared width, so a one-byte flag, a DWORD and a
    // caller-sized buffer are each compared within their own extent.
    bool compared[2] = {};
    std::size_t compared_bytes[2] = {};
    int compared_index[2] = {no_argument, no_argument};
    unsigned char before[2][max_snapshot] = {};
    const DeclaredOut* contract = trace_api() ? declared(provider_ids[self->provider].Data1, N) : nullptr;
    if (contract) {
        const auto take = [&](int slot, unsigned char index, unsigned char fixed_bytes, unsigned char size_index) {
            compared_index[slot] = index;
            if (index == no_argument || !args[index]) return;
            std::size_t bytes = fixed_bytes;
            if (!bytes) {
                if (size_index == no_argument) return;
                const u64 capacity = args[size_index];
                bytes = capacity < max_snapshot ? static_cast<std::size_t>(capacity) : max_snapshot;
            }
            if (!bytes || bytes > max_snapshot) return;
            if (!x2w_local::readable(reinterpret_cast<const void*>(args[index]), bytes)) return;
            std::memcpy(before[slot], reinterpret_cast<const void*>(args[index]), bytes);
            compared_bytes[slot] = bytes;
            compared[slot] = true;
        };
        take(0, contract->first, contract->first_bytes, contract->size_argument);
        take(1, contract->second, contract->second_bytes, no_argument);
    }
    u64 result = 0;
    bool handled = self->provider == user_provider && fake_user() && user_call(N, args, &result);
    if (!handled && self->provider == gamertag_provider && N == 3 && fake_user()) {
        // GetGamertag: the local user's name, NUL-terminated UTF-8; used = bytes including the NUL.
        static const char name[] = "Player";
        const u64 size = c, buffer = s0, used = s1;
        handled = true;
        if (a != local_user_handle) {
            result = 0x89245104;  // no such user
        } else if (size < sizeof(name) || !buffer || !x2w_local::writable(reinterpret_cast<void*>(buffer), sizeof(name))) {
            result = 0x8007007A;  // buffer too small
        } else {
            std::memcpy(reinterpret_cast<void*>(buffer), name, sizeof(name));
            if (used && x2w_local::writable(reinterpret_cast<void*>(used), 8)) *reinterpret_cast<u64*>(used) = sizeof(name);
            result = 0;
        }
    }
    if (!handled && self->provider == save_provider && N >= 3 && fake_user()) {
        if (self->public_save_interface) handled = save_read_call(N, args, &result);
        if (!handled) handled = save_call(N, args, &result);
    }
    if (!handled && N >= 3 && fake_user()) handled = area_call(provider_ids[self->provider], N, args, &result);
    if (!handled && N >= 3 && fake_user()) {
        // The offline context is our token, never a desktop runtime object. Unimplemented
        // methods must fail here rather than let the real runtime dereference it.
        for (int i = 0; i < 4 && !handled; ++i) {
            if (args[i] == store_context_handle) {
                handled = true;
                result = 0x80004005;
            }
        }
    }
    if (!handled && self->provider != user_provider && N >= 3 && fake_user()) {
        // Other API families cannot resolve the local user's handle: fail the call instead of passing it on.
        for (int i = 0; i < 4 && !handled; ++i) {
            if (args[i] == local_user_handle) {
                handled = true;
                result = 0x80004005;
            }
        }
    }
    if (!handled && !self->real) {
        // A local object (X2W_RUNTIME_LOCAL) has no runtime object behind it.
        handled = true;
        if (N == 0) {  // QueryInterface(iid, &object)
            result = static_cast<unsigned>(local_query_interface(self, provider_ids[self->provider].Data1,
                self->local_iid, reinterpret_cast<const GUID*>(a), reinterpret_cast<void**>(b), self->rejected_iid));
        } else if (N == 1 || N == 2) {  // AddRef, Release keep local objects for the process lifetime.
            result = 1;
        } else {
            result = 0x80004001;  // E_NOTIMPL
            static LONG unanswered[max_providers][max_slots];
            if (InterlockedIncrement(&unanswered[self->provider][N]) == 1)
                x2w_local::logf("x2w-runtime local: %08lx::slot%d has no answer, E_NOTIMPL", provider_ids[self->provider].Data1, N);
        }
    }
    if (!handled) {
        const Method method = reinterpret_cast<Method>((*reinterpret_cast<void***>(self->real))[N]);
        result = method(self->real, a, b, c, s0, s1, s2, s3, s4, s5, s6, s7);
    }
    const bool about_tracked = N >= 3 && (tracked(a) || tracked(b) || tracked(c));
    const u64 raw_result = result;
    if (about_tracked && !handled && fake_user() && (result & 0xffffffffu) == borrowed_failure) result = 0;
    if (trace_api()) {
        unsigned char after[2][max_snapshot] = {};
        bool current[2] = {};
        for (int i = 0; i < 2; ++i) {
            if (!compared[i] || !compared_bytes[i]) continue;
            if (!x2w_local::readable(reinterpret_cast<const void*>(args[compared_index[i]]), compared_bytes[i])) continue;
            std::memcpy(after[i], reinterpret_cast<const void*>(args[compared_index[i]]), compared_bytes[i]);
            current[i] = true;
        }
        const x2w_api_trace::Report report = x2w_api_trace::decide(contract, current,
                                                                  const_cast<const unsigned char (*)[max_snapshot]>(before),
                                                                  const_cast<const unsigned char (*)[max_snapshot]>(after),
                                                                  compared_bytes);
        char out[48] = "?";
        x2w_api_trace::format(report, contract, out, sizeof(out));
        x2w_local::logf("x2w-api %08lx::slot%d -> 0x%08x %s out=%s", provider_ids[self->provider].Data1, N,
                        static_cast<unsigned>(result & 0xffffffffu), handled ? "local" : "real", out);
    }
    if (log_it || (about_tracked && x2w_local::first_calls(tracked_logged, 120)))
        x2w_local::logf("x2w-runtime %08lx::slot%d(0x%llx, 0x%llx, 0x%llx | 0x%llx, 0x%llx) -> 0x%llx%s%s from +0x%llx", provider_ids[self->provider].Data1, N,
                        a, b, c, s0, s1, result, handled ? " (local user)" : "",
                        about_tracked ? (raw_result != result ? " (tracked async, failure hidden)" : " (tracked async)") : "",
                        reinterpret_cast<u64>(_ReturnAddress()) - reinterpret_cast<u64>(GetModuleHandleW(nullptr)));
    return result;
}
template <std::size_t... I>
void fill_slots(std::index_sequence<I...>)
{
    ((slot_table[I] = reinterpret_cast<void*>(&forward<static_cast<int>(I)>)), ...);
}

HRESULT __stdcall proxy_query(const GUID* provider, const GUID* iid, void** out)
{
    const QueryApi query = load_real_query();
    if (!query) return E_FAIL;
    const HRESULT hr = query(provider, iid, out);
    const bool output_writable = FAILED(hr) && fake_user() && x2w_local::writable(out, sizeof(*out));
    const bool pair_readable = output_writable && x2w_local::readable(provider, sizeof(*provider)) &&
                               x2w_local::readable(iid, sizeof(*iid));
    if (pair_readable && matches_pair(storage_pair, provider, iid)) {
        // Persistent local storage: the system runtime has none for an unpackaged process.
        static LONG logged;
        if (x2w_local::first_calls(logged, 2)) x2w_local::logf("x2w-runtime persistent local storage: local object instead of 0x%08lx", static_cast<unsigned long>(hr));
        *out = storage_object();
        return S_OK;
    }
    const ApiPair* local = nullptr;
    if (output_writable && InterlockedCompareExchange(&local_active, 0, 0)) {
        *out = nullptr;
        if (!pair_readable) return hr;
        for (const auto& pair : local_pairs) {
            if (matches_pair(pair, provider, iid)) {
                local = &pair;
                break;
            }
        }
        if (local) {
            static LONG logged;
            if (x2w_local::first_calls(logged, 24))
                x2w_local::logf("x2w-runtime local: provider %08lx real 0x%08lx -> local object", provider->Data1, static_cast<unsigned long>(hr));
        }
    }
    if (!out || !provider || (!local && (FAILED(hr) || !*out))) return hr;
    GUID key = *provider;
    const bool gamertag = provider->Data1 == user_provider_data1 && iid && iid->Data1 == gamertag_iid_data1;
    if (gamertag) key.Data1 = gamertag_iid_data1;
    provider = &key;
    AcquireSRWLockExclusive(&proxy_lock);
    int index = -1;
    for (int i = 0; i < provider_count; ++i)
        if (std::memcmp(&provider_ids[i], provider, sizeof(GUID)) == 0) index = i;
    if (index < 0 && provider_count < max_providers) {
        index = provider_count++;
        provider_ids[index] = *provider;
        if (provider->Data1 == user_provider_data1) user_provider = index;
        if (gamertag) gamertag_provider = index;
        if (provider->Data1 == save_provider_data1) save_provider = index;
        x2w_local::logf("x2w-runtime provider %08lx-%04x-... -> object %p (proxied as #%d)", provider->Data1, provider->Data2, *out, index);
    }
    Proxy* proxy = nullptr;
    if (index >= 0) {
        constexpr GUID save_class{0x704c3f58,0xe629,0x4cc2,{0xb1,0x97,0x30,0x51,0x1b,0x99,0x6f,0xe2}};
        constexpr GUID save_iid{0x1bfff3af,0xf14a,0x40a3,{0x8e,0x35,0x9a,0xda,0x90,0x65,0x93,0xf9}};
        const bool public_save = *provider == save_class && iid && *iid == save_iid;
        if (local) {
            // Each pair keeps its own interface identity even when two IIDs share a provider index.
            static Proxy* local_proxies[sizeof(local_pairs) / sizeof(*local_pairs)];
            auto& entry = local_proxies[local - local_pairs];
            if (!entry) entry = new Proxy{slot_table, nullptr, index, &local->iid, 0, public_save};
            proxies[index] = entry;
        } else {
            // One cached proxy per interface kind, so alternating queries reuse them instead of
            // allocating (and logging) a new one each time.
            static Proxy* by_kind[2][max_providers];
            Proxy*& cached = by_kind[public_save][index];
            if (!cached || cached->real != *out) {
                cached = new Proxy{slot_table, *out, index, nullptr, 0, public_save};
                if (index == save_provider)  // whether the synchronous queries of save_read.inl answer this interface
                    x2w_local::logf("x2w-runtime game save: interface %08lx -> synchronous reads %s", iid ? iid->Data1 : 0ul, public_save ? "on" : "off");
            }
            proxies[index] = cached;
        }
        proxy = proxies[index];
    }
    ReleaseSRWLockExclusive(&proxy_lock);
    if (proxy) *out = proxy;
    if (local) return proxy ? S_OK : hr;
    return hr;
}

// X2W_RUNTIME_LOCAL: the first failed InitializeApiImpl* call is reported as S_OK. Once latched,
// every later initialisation call returns S_OK without calling the real export.
template <int K>
u64 __stdcall proxy_init(u64 a, u64 b, u64 c, u64 d)
{
    if (InterlockedCompareExchange(&local_active, 0, 0)) return 0;
    const u64 real = real_init[K] ? real_init[K](a, b, c, d) : 0x80004005;
    if (static_cast<HRESULT>(real & 0xffffffffu) >= 0) return real;
    if (InterlockedExchange(&local_active, 1) == 0)
        x2w_local::logf("x2w-runtime local: %s(0x%llx, 0x%llx) real 0x%08lx -> S_OK (local runtime)", init_names[K], a, b,
                        static_cast<unsigned long>(real & 0xffffffffu));
    return 0;
}

HMODULE proxy_module;  // this DLL; its own GetProcAddress IAT must stay real or the hook recurses

bool wide_eq_ascii_ci(const wchar_t* wide, const char* ascii) noexcept
{
    if (!wide || !ascii) return false;
    for (;; ++wide, ++ascii) {
        const wchar_t w = *wide;
        const unsigned char a = static_cast<unsigned char>(*ascii);
        if (!w && !a) return true;
        if (!w || !a) return false;
        const wchar_t wl = (w >= L'A' && w <= L'Z') ? static_cast<wchar_t>(w - L'A' + L'a') : w;
        const wchar_t al = (a >= 'A' && a <= 'Z') ? static_cast<wchar_t>(a - 'A' + 'a') : static_cast<wchar_t>(a);
        if (wl != al) return false;
    }
}

const wchar_t* path_basename(const wchar_t* path) noexcept
{
    const wchar_t* base = path;
    for (const wchar_t* p = path; p && *p; ++p)
        if (*p == L'\\' || *p == L'/') base = p + 1;
    return base;
}

bool module_basename_is(HMODULE module, const char* expect) noexcept
{
    wchar_t path[MAX_PATH]{};
    if (!module) return false;
    const DWORD length = GetModuleFileNameW(module, path, MAX_PATH);
    if (!length || length >= MAX_PATH) return false;
    return wide_eq_ascii_ci(path_basename(path), expect);
}

#if defined(X2W_RUNTIME_PROXY_FIXTURE)
// startup_runtime_local stubs GetProcAddress and passes a null module; skip the basename gate there.
bool module_is_xgameruntime(HMODULE) noexcept { return true; }
#else
bool module_is_xgameruntime(HMODULE module) noexcept { return module_basename_is(module, "xgameruntime.dll"); }
#endif

// EXE and DLLs that sit next to it (the package/run folder). System and injector modules are out of scope.
bool module_is_exe_or_package_helper(HMODULE module) noexcept
{
    if (!module) return false;
    const HMODULE exe = GetModuleHandleW(nullptr);
    if (module == exe) return true;
    wchar_t exe_path[MAX_PATH]{}, mod_path[MAX_PATH]{};
    const DWORD exe_length = GetModuleFileNameW(exe, exe_path, MAX_PATH);
    const DWORD mod_length = GetModuleFileNameW(module, mod_path, MAX_PATH);
    if (!exe_length || exe_length >= MAX_PATH || !mod_length || mod_length >= MAX_PATH) return false;
    wchar_t* exe_slash = nullptr;
    for (wchar_t* p = exe_path; *p; ++p)
        if (*p == L'\\' || *p == L'/') exe_slash = p;
    wchar_t* mod_slash = nullptr;
    for (wchar_t* p = mod_path; *p; ++p)
        if (*p == L'\\' || *p == L'/') mod_slash = p;
    if (!exe_slash || !mod_slash) return false;
    const std::size_t exe_dir = static_cast<std::size_t>(exe_slash - exe_path);
    const std::size_t mod_dir = static_cast<std::size_t>(mod_slash - mod_path);
    if (exe_dir != mod_dir) return false;
    for (std::size_t i = 0; i < exe_dir; ++i) {
        wchar_t a = exe_path[i], b = mod_path[i];
        if (a >= L'A' && a <= L'Z') a = static_cast<wchar_t>(a - L'A' + L'a');
        if (b >= L'A' && b <= L'Z') b = static_cast<wchar_t>(b - L'A' + L'a');
        if (a != b) return false;
    }
    return true;
}

#include "startup_stubs/registry_console_mode.inl"

FARPROC WINAPI hooked_get_proc_address(HMODULE module, LPCSTR name)
{
    const FARPROC real = GetProcAddress(module, name);
    if (real && reinterpret_cast<u64>(name) > 0xffff) {
        static LONG logged;
        if (x2w_local::first_calls(logged, 40)) x2w_local::logf("x2w-runtime GetProcAddress(%p, \"%s\") -> %p", static_cast<void*>(module), name, static_cast<void*>(real));
        // Registry exports only when the module is a Windows registry provider (registry_console_mode.inl).
        if (const FARPROC hook = registry_hook_for(module, name, real)) return hook;
        // Wrap gaming-runtime exports only when the provider module is xgameruntime.dll.
        if (module_is_xgameruntime(module) && std::strcmp(name, "QueryApiImpl") == 0 &&
            remember_real_query(reinterpret_cast<QueryApi>(real)))
            return reinterpret_cast<FARPROC>(&proxy_query);
        if (runtime_local() && fake_user() && module_is_xgameruntime(module)) {
            static constexpr InitApi wrappers[3] = {&proxy_init<0>, &proxy_init<1>, &proxy_init<2>};
            for (int k = 0; k < 3; ++k) {
                if (std::strcmp(name, init_names[k]) != 0 || !keep_forward_target(real)) continue;
                real_init[k] = reinterpret_cast<InitApi>(real);
                return reinterpret_cast<FARPROC>(wrappers[k]);
            }
        }
    }
    return real;
}

// Resolve QueryApiImpl from a loaded xgameruntime.dll without reading a delay-IAT thunk.
void remember_real_query_from_export() noexcept
{
    const HMODULE runtime = GetModuleHandleW(L"xgameruntime.dll");
    if (!runtime || !module_is_xgameruntime(runtime)) return;
    // This stand-in's own GetProcAddress IAT is left unpatched, so this is the real export lookup.
    remember_real_query(reinterpret_cast<QueryApi>(GetProcAddress(runtime, "QueryApiImpl")));
}

// Rewrite one module's regular GetProcAddress / xgameruntime.dll!QueryApiImpl IAT slots.
// Delay-import hits are not rewritten (slot is a delay thunk until first use). Caller must hold the
// module live (pin) for the duration of the rewrite.
int patch_module(HMODULE module)
{
    if (!module || !module_is_exe_or_package_helper(module)) return 0;
    auto* base = reinterpret_cast<unsigned char*>(module);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return 0;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE || nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) return 0;
    const auto image_size = static_cast<std::size_t>(nt->OptionalHeader.SizeOfImage);
    const bool self = module == proxy_module;
    int patched = 0;
    x2w_runtime_patch::visit_imports(base, image_size, [&](const x2w_runtime_patch::Hit& hit) {
        // Never rewrite this DLL's GetProcAddress: hooked_get_proc_address calls it for the real lookup.
        if (self && hit.kind == x2w_runtime_patch::Kind::get_proc_address) return;
        if (hit.delay_import) {
            // Delay IAT holds a thunk, not the export. Do not rewrite; delay GetProcAddress is unhooked.
            if (hit.kind == x2w_runtime_patch::Kind::query_api_impl) remember_real_query_from_export();
            return;
        }
        auto* slot = reinterpret_cast<ULONGLONG*>(base + hit.first_thunk_rva);
        DWORD old = 0;
        if (!VirtualProtect(slot, 8, PAGE_READWRITE, &old)) return;
        if (hit.kind == x2w_runtime_patch::Kind::get_proc_address) {
            if (*slot != reinterpret_cast<ULONGLONG>(&hooked_get_proc_address)) {
                *slot = reinterpret_cast<ULONGLONG>(&hooked_get_proc_address);
                ++patched;
            }
        } else if (hit.kind == x2w_runtime_patch::Kind::query_api_impl) {
            // Regular import: slot holds the real export after the loader bound it.
            if (*slot != reinterpret_cast<ULONGLONG>(&proxy_query) &&
                remember_real_query(reinterpret_cast<QueryApi>(*slot))) {
                *slot = reinterpret_cast<ULONGLONG>(&proxy_query);
                ++patched;
            }
        }
        VirtualProtect(slot, 8, old, &old);
    });
    // Registry imports for the console mode (registry_console_mode.inl), counted on their own line.
    if (!self) {
        const int registry = patch_registry_imports(base, image_size);
        if (registry > 0)
            x2w_local::logf("x2w-runtime console mode: %d registry import(s) patched in the %s", registry,
                            module == GetModuleHandleW(nullptr) ? "main module" : "package module");
    }
    return patched;
}

// Pin the module, rewrite after linking, then drop the pin. Safe outside LdrDllNotification.
int patch_module_pinned(void* dll_base)
{
    if (!dll_base) return 0;
    HMODULE pinned = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, reinterpret_cast<LPCWSTR>(dll_base), &pinned) || !pinned)
        return 0;
    const int n = patch_module(pinned);
    if (n > 0) {
        char name[64] = "?";
        wchar_t path[MAX_PATH]{};
        const DWORD length = GetModuleFileNameW(pinned, path, MAX_PATH);
        if (length && length < MAX_PATH) {
            const wchar_t* base = path_basename(path);
            const int keep = 60;
            int i = 0;
            for (; i < keep && base[i]; ++i) {
                const wchar_t c = base[i];
                name[i] = (c > 0 && c < 128) ? static_cast<char>(c) : '?';
            }
            name[i] = 0;
        }
        x2w_local::logf("x2w-runtime proxy: %d import(s) patched in package module %s", n, name);
    }
    FreeLibrary(pinned);
    return n;
}

using LdrDllNotification = void(CALLBACK*)(ULONG reason, const void* data, void* context);
using LdrRegisterDllNotificationFn = LONG(NTAPI*)(ULONG flags, LdrDllNotification callback, void* context, void** cookie);
constexpr ULONG ldr_dll_notification_reason_loaded = 1;
// Keep the pre-link notification inside this module. One scan covers all still-loaded helpers,
// so a burst needs no storage for module addresses and cannot overwrite pending work.
std::atomic<bool> rescan_package_modules{false};
static_assert(std::atomic<bool>::is_always_lock_free);

void CALLBACK dll_loaded(ULONG reason, const void*, void*)
{
    if (reason == ldr_dll_notification_reason_loaded)
        rescan_package_modules.store(true, std::memory_order_release);
}

void patch_package_modules()
{
    std::vector<HMODULE> modules(512);
    DWORD needed = 0;
    if (!EnumProcessModules(GetCurrentProcess(), modules.data(), static_cast<DWORD>(modules.size() * sizeof(HMODULE)), &needed) || !needed)
        return;
    if (needed > modules.size() * sizeof(HMODULE)) {
        x2w_local::logf("x2w-runtime proxy: EnumProcessModules truncated (%u bytes needed, grew from %zu)",
                        static_cast<unsigned>(needed), modules.size() * sizeof(HMODULE));
        modules.resize(needed / sizeof(HMODULE) + 8);
        needed = 0;
        if (!EnumProcessModules(GetCurrentProcess(), modules.data(), static_cast<DWORD>(modules.size() * sizeof(HMODULE)), &needed) ||
            !needed)
            return;
        if (needed > modules.size() * sizeof(HMODULE)) {
            x2w_local::logf("x2w-runtime proxy: EnumProcessModules still truncated (%u bytes needed for %zu slots); scanning first %zu",
                            static_cast<unsigned>(needed), modules.size(), modules.size());
            needed = static_cast<DWORD>(modules.size() * sizeof(HMODULE));
        }
    }
    const unsigned count = needed / sizeof(HMODULE);
    int modules_patched = 0, slots = 0, considered = 0;
    for (unsigned i = 0; i < count; ++i) {
        if (!module_is_exe_or_package_helper(modules[i])) continue;
        ++considered;
        const int n = patch_module_pinned(modules[i]);
        if (n > 0) {
            ++modules_patched;
            slots += n;
        }
    }
    if (slots > 0 || considered > 0)
        x2w_local::logf("x2w-runtime proxy: %d import(s) in %u of %u package module(s) (%u loaded scanned)", slots, modules_patched,
                        considered, count);
}

// Patch the main EXE immediately. The load notice only requests another worker scan.
// The worker pins and patches modules after linking. Package-directory helpers are in scope; other modules are not.
void patch_imports()
{
    const HMODULE exe = GetModuleHandleW(nullptr);
    HMODULE pinned = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, reinterpret_cast<LPCWSTR>(exe), &pinned) && pinned) {
        const int main_slots = patch_module(pinned);
        x2w_local::logf("x2w-runtime proxy: %d import(s) of the main module patched", main_slots);
        FreeLibrary(pinned);
    }
    static void* dll_cookie = nullptr;
    static const LdrRegisterDllNotificationFn register_dll = [] {
        const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
        return ntdll ? reinterpret_cast<LdrRegisterDllNotificationFn>(GetProcAddress(ntdll, "LdrRegisterDllNotification"))
                     : nullptr;
    }();
    if (register_dll && !dll_cookie) {
        const LONG status = register_dll(0, &dll_loaded, nullptr, &dll_cookie);
        if (status != 0) x2w_local::logf("x2w-runtime proxy: LdrRegisterDllNotification failed 0x%08lx", static_cast<unsigned long>(status));
    }
    if (HANDLE worker = CreateThread(nullptr, 0, [](void*) -> DWORD {
        // Outside DllMain and the pre-link notification. Clear the request before each scan so
        // loads that arrive during the scan are picked up on the next pass.
        rescan_package_modules.store(true, std::memory_order_release);
        for (;;) {
            if (rescan_package_modules.exchange(false, std::memory_order_acq_rel))
                patch_package_modules();
            Sleep(25);
        }
    }, nullptr, 0, nullptr)) {
        CloseHandle(worker);
    } else {
        x2w_local::logf("x2w-runtime proxy: package-module worker unavailable (%lu)", GetLastError());
    }
}

// ---- the local user -------------------------------------------------------------------------
// Filled in from observed calls (slot numbers are positions in the user API object's vtable).
// Returning false forwards the call to the real runtime.
bool user_call(int slot, u64* args, u64* result)
{
    const auto out = [](u64 pointer, std::size_t bytes) { return pointer && x2w_local::writable(reinterpret_cast<void*>(pointer), bytes); };
    constexpr u64 ok = 0, fail = 0x80004005, not_found = 0x80070490;
    switch (slot) {
    case 3:  // XUserDuplicateHandle(handle, &duplicate)
        if (args[0] != local_user_handle) return false;
        if (out(args[1], 8)) *reinterpret_cast<u64*>(args[1]) = local_user_handle;
        *result = ok;
        return true;
    case 4:  // XUserCloseHandle(handle)
        if (args[0] != local_user_handle) return false;
        *result = 0;
        return true;
    case 5:  // XUserCompare(a, b)
        if (args[0] != local_user_handle && args[1] != local_user_handle) return false;
        *result = args[0] == args[1] ? 0 : 1;
        return true;
    case 7:  // XUserAddAsync(options, async): with a local runtime there is no real operation to borrow
        if (!InterlockedCompareExchange(&local_active, 0, 0)) break;  // unchanged without a local runtime
        *result = local_user_add(args[1]);
        return true;
    case 8:  // XUserAddResult(async, &handle): the real operation ends with "no default user"; answer with the local user
        if (out(args[1], 8)) *reinterpret_cast<u64*>(args[1]) = local_user_handle;
        *result = ok;
        return true;
    case 9:  // XUserGetLocalId(handle, &local id)
        if (args[0] != local_user_handle) return false;
        if (out(args[1], 8)) *reinterpret_cast<u64*>(args[1]) = 1;
        *result = ok;
        return true;
    case 10:  // XUserFindUserByLocalId(local id, &handle)
    case 12:  // XUserFindUserById(id, &handle)
        if (args[0] != (slot == 10 ? 1 : local_user_id)) return false;
        if (out(args[1], 8)) *reinterpret_cast<u64*>(args[1]) = local_user_handle;
        *result = ok;
        return true;
    case 11:  // XUserGetId(handle, &id)
        if (args[0] != local_user_handle) return false;
        if (out(args[1], 8)) *reinterpret_cast<u64*>(args[1]) = local_user_id;
        *result = ok;
        return true;
    case 13:  // XUserGetIsGuest(handle, &is guest)
        if (args[0] != local_user_handle) return false;
        if (out(args[1], 1)) *reinterpret_cast<bool*>(args[1]) = false;
        *result = ok;
        return true;
    case 14:  // XUserGetState(handle, &state): 0 = signed in
        if (args[0] != local_user_handle) return false;
        if (out(args[1], 4)) *reinterpret_cast<std::uint32_t*>(args[1]) = 0;
        *result = ok;
        return true;
    case 15: {  // XUserGetGamertag(handle, component, size, buffer, &used)
        if (args[0] != local_user_handle) return false;
        static const char name[] = "Player";
        const u64 size = args[2];
        if (size >= sizeof(name) && out(args[3], sizeof(name))) std::memcpy(reinterpret_cast<void*>(args[3]), name, sizeof(name));
        if (out(args[4], 8)) *reinterpret_cast<u64*>(args[4]) = sizeof(name);
        *result = size >= sizeof(name) ? ok : 0x8007007A;  // ERROR_INSUFFICIENT_BUFFER
        return true;
    }
    case 19:  // XUserGetAgeGroup(handle, &group): 3 = adult
        if (args[0] != local_user_handle) return false;
        if (out(args[1], 4)) *reinterpret_cast<std::uint32_t*>(args[1]) = 3;
        *result = ok;
        return true;
    case 20:  // XUserCheckPrivilege(handle, options, privilege, &has, &deny reason)
        if (args[0] != local_user_handle) return false;
        if (out(args[3], 1)) *reinterpret_cast<bool*>(args[3]) = true;
        if (out(args[4], 4)) *reinterpret_cast<std::uint32_t*>(args[4]) = 0;
        *result = ok;
        return true;
    case 40:  // XUserFindForDevice(device id, &handle): the virtual gamepad belongs to the local user
        if (out(args[1], 8)) *reinterpret_cast<u64*>(args[1]) = local_user_handle;
        *result = ok;
        return true;
    default:
        break;
    }
    // Anything else that names the local user cannot go to the real runtime (the handle is not its own).
    for (int i = 0; i < 4; ++i) {
        if (args[i] == local_user_handle) {
            *result = slot == 35 ? not_found : fail;
            return true;
        }
    }
    return false;
}

// ---- local game saves -----------------------------------------------------------------------
// Slot numbers follow the order of the public XGameSave.h (hypothesis, checked against the logged
// argument shapes). Containers and blob writes are kept on disk (save_store.h, below
// storage_path()/x2w-gamesave). For the verified public interface (see proxy_query), the
// synchronous queries 13, 14, 18, 19 and 20 return committed data (save_read.inl); other interface
// revisions keep the empty results in save_call. Slots 21 and 22 read through the system XAsync
// service (see async_read). The other asynchronous calls are completed by starting the real "add
// default user silently" operation on the caller's async block: it fails quickly inside the
// runtime and runs the caller's completion callback on its task queue; the matching Result call is
// then answered here.
constexpr u64 save_provider_handle = 0x58325753'00000001ull, save_container_handle = 0x58325743'00000001ull,
              save_update_handle = 0x58325755'50000001ull;
// Container and update handles keep their fixed base; the low bits count from 1 and index these tables.
// Reopening a container name returns the same handle, so the table stays small.
constexpr std::size_t max_save_handles = x2w_save::ContainerTable::capacity;
constexpr u64 e_invalidarg = 0x80070057, e_outofmemory = 0x8007000E, e_fail = 0x80004005;
x2w_save::ContainerTable save_table;  // guarded by save_lock; closed containers give their entry back
SRWLOCK save_lock = SRWLOCK_INIT;
x2w_save::Store save_store() { return x2w_save::Store(std::filesystem::path(storage_path()) / "x2w-gamesave"); }
// A NUL-terminated name of at most x2w_save::max_name_bytes in readable memory.
bool read_save_name(u64 pointer, std::string& name)
{
    const std::size_t readable = x2w_local::probe_readable(static_cast<std::uintptr_t>(pointer), x2w_save::max_name_bytes + 1);
    const char* text = reinterpret_cast<const char*>(pointer);
    const std::size_t length = readable ? strnlen(text, readable) : 0;
    if (!readable || length == readable) return false;
    name.assign(text, length);
    return true;
}
u64 save_status(x2w_save::Status status)
{
    using x2w_save::Status;
    switch (status) {
    case Status::ok: return 0;
    case Status::bad_name: case Status::too_large: return e_invalidarg;
    case Status::not_found: return 0x80830008;
    default: return e_fail;
    }
}
u64 open_container(const std::string& name, u64* handle)
{
    const x2w_save::Status status = save_store().create_container(name);
    if (status != x2w_save::Status::ok) return save_status(status);
    AcquireSRWLockExclusive(&save_lock);
    const long index = save_table.open(name);
    ReleaseSRWLockExclusive(&save_lock);
    if (index < 0) return e_outofmemory;
    *handle = save_container_handle + static_cast<u64>(index);
    return 0;
}
// The container name behind an update handle, or empty when it is not an open update.
std::string update_container(u64 update)
{
    std::string name;
    AcquireSRWLockShared(&save_lock);
    struct Unlock { ~Unlock() { ReleaseSRWLockShared(&save_lock); } } unlock;
    if (update >= save_update_handle) name = save_table.update_container(static_cast<std::size_t>(update - save_update_handle));
    return name;
}
#include "startup_stubs/save_read.inl"
// Publishes what an update has staged, all or nothing. The update itself stays open.
u64 submit_update(u64 update)
{
    if (update < save_update_handle || update - save_update_handle >= max_save_handles) return 0;
    const std::size_t index = static_cast<std::size_t>(update - save_update_handle);
    try {
        std::string container;
        std::vector<x2w_save::Action> actions;
        {
            AcquireSRWLockExclusive(&save_lock);
            struct Unlock { ~Unlock() { ReleaseSRWLockExclusive(&save_lock); } } guard;
            container = save_table.update_container(index);
            actions = save_table.take(index);
        }
        if (container.empty() || actions.empty()) return 0;
        SetLastError(0);
        const auto status = save_store().commit(container, actions);
        const DWORD last_error = GetLastError();
        {
            // Every game-save write result (save_write_log.h): the first 20 in full, failures, totals every 25 writes.
            static x2w_save::WriteLog write_log;
            u64 staged = 0;
            for (const auto& action : actions) staged += action.data.size();
            AcquireSRWLockExclusive(&save_lock);
            const std::string text = write_log.record(container, actions.size(), staged, status == x2w_save::Status::ok,
                                                      static_cast<int>(status), last_error);
            ReleaseSRWLockExclusive(&save_lock);
            if (!text.empty()) x2w_local::logf("%s", text.c_str());
        }
        if (status != x2w_save::Status::ok) x2w_local::logf("x2w-runtime game save: update refused (status %d)", static_cast<int>(status));
        return save_status(status);
    } catch (const std::bad_alloc&) {
        return e_outofmemory;
    } catch (const std::filesystem::filesystem_error&) {
        return e_fail;
    }
}
// XAsyncBegin reserves the caller's block and completion queue. Disk publication
// starts only after acceptance. Complete carries the result for this block;
// it can invoke user code immediately, so no save lock may span that call.
// The verified public PC GDK async interface uses slots 7, 10 and 11.
constexpr GUID submit_async_family{0x073b7dcb,0x1fcf,0x4030,{0x94,0xbe,0xe3,0xc9,0xeb,0x62,0x34,0x28}};
constexpr char submit_identity[] = "local XGameSaveSubmitUpdate";
HRESULT WINAPI submit_provider(unsigned, const void*) { return S_OK; }
u64 async_submit(int slot, u64 update, u64 block)
{
    const QueryApi ask = load_real_query();
    if (!block || !ask) return e_fail;
    void* service = nullptr;
    const auto query = ask(&submit_async_family, &submit_async_family, &service);
    if (FAILED(query)) return static_cast<unsigned>(query);
    if (!service) return e_fail;
    auto** methods = *static_cast<void***>(service);
    struct ApiRef {
        void* object;
        ULONG (WINAPI* release)(void*);
        ~ApiRef() { release(object); }
    } api{service,reinterpret_cast<ULONG(WINAPI*)(void*)>(methods[2])};
    auto* async = reinterpret_cast<void*>(block);
    if (slot == 29) {
        using Result = HRESULT(WINAPI*)(void*,void*,const void*,size_t,void*,size_t*);
        return static_cast<unsigned>(reinterpret_cast<Result>(methods[11])(service, async, submit_identity, 0, nullptr, nullptr));
    }
    using Begin = HRESULT(WINAPI*)(void*,void*,void*,const void*,const char*,decltype(&submit_provider));
    const auto hr = reinterpret_cast<Begin>(methods[7])(service, async, nullptr, submit_identity, submit_identity, submit_provider);
    if (FAILED(hr)) return static_cast<unsigned>(hr);
    const auto status = static_cast<HRESULT>(submit_update(update));
    using Complete = void(WINAPI*)(void*,void*,HRESULT,size_t);
    reinterpret_cast<Complete>(methods[10])(service, async, status, 0);
    return S_OK;
}

// X2W_RUNTIME_LOCAL: XUserAddAsync completes at once with S_OK through the same XAsync service;
// XUserAddResult (user slot 8) then hands out the local user.
constexpr char user_add_identity[] = "local XUserAddAsync";
u64 local_user_add(u64 block)
{
    const QueryApi ask = load_real_query();
    if (!block || !ask) return e_fail;
    void* service = nullptr;
    const auto query = ask(&submit_async_family, &submit_async_family, &service);
    if (FAILED(query) || !service) {
        x2w_local::logf("x2w-runtime local: XUserAddAsync, async service unavailable 0x%08lx", static_cast<unsigned long>(query));
        return FAILED(query) ? static_cast<unsigned>(query) : e_fail;
    }
    auto** methods = *static_cast<void***>(service);
    struct ApiRef {
        void* object;
        ULONG (WINAPI* release)(void*);
        ~ApiRef() { release(object); }
    } api{service, reinterpret_cast<ULONG(WINAPI*)(void*)>(methods[2])};
    auto* async = reinterpret_cast<void*>(block);
    using Begin = HRESULT(WINAPI*)(void*, void*, void*, const void*, const char*, decltype(&submit_provider));
    const auto hr = reinterpret_cast<Begin>(methods[7])(service, async, nullptr, user_add_identity, user_add_identity, submit_provider);
    if (FAILED(hr)) return static_cast<unsigned>(hr);
    using Complete = void(WINAPI*)(void*, void*, HRESULT, size_t);
    reinterpret_cast<Complete>(methods[10])(service, async, S_OK, 0);
    static LONG logged;
    if (x2w_local::first_calls(logged, 4)) x2w_local::logf("x2w-runtime local: XUserAddAsync completed with the local user");
    return S_OK;
}

// XGameSaveReadBlobDataAsync / Result. Same XAsync service as submit, different identity, so a
// read block is not a submit block. Begin is attempted before any read: a rejection leaves the
// store untouched. The snapshot is packed only after acceptance. Complete can run the caller's
// callback before it returns, so no save lock is held across it. A failed read completes with no
// payload (the runtime then drops the context); the bytes are copied into the caller's buffer only
// when slot 22 asks, and only if that buffer is large enough.
constexpr char read_identity[] = "local XGameSaveReadBlobData";
constexpr u64 small_buffer = 0x80830007;
struct AsyncProviderData {
    void* async;
    std::size_t buffer_size;
    void* buffer;
    void* context;
};
static_assert(sizeof(AsyncProviderData) == 32 && offsetof(AsyncProviderData, context) == 24, "XAsyncProviderData");
struct ReadAsync {
    void* block = nullptr;
    std::uint32_t count = 0;
    std::size_t bytes = 0;
    std::vector<x2w_save_pack::Piece> pieces;
};
struct ReadRecord { void* block; std::uint32_t count; std::size_t bytes; };
std::vector<ReadRecord> read_records;  // accepted reads whose payload is still outstanding
SRWLOCK read_records_lock = SRWLOCK_INIT;
void remember_read(const ReadAsync& state)
{
    AcquireSRWLockExclusive(&read_records_lock);
    struct Unlock { ~Unlock() { ReleaseSRWLockExclusive(&read_records_lock); } } unlock;
    read_records.push_back({state.block, state.count, state.bytes});
}
void forget_read(void* block)
{
    AcquireSRWLockExclusive(&read_records_lock);
    struct Unlock { ~Unlock() { ReleaseSRWLockExclusive(&read_records_lock); } } unlock;
    for (std::size_t i = 0; i < read_records.size(); ++i)
        if (read_records[i].block == block) {
            read_records.erase(read_records.begin() + static_cast<std::ptrdiff_t>(i));
            break;
        }
}
bool pending_read(void* block, std::uint32_t& count, std::size_t& bytes)
{
    AcquireSRWLockShared(&read_records_lock);
    struct Unlock { ~Unlock() { ReleaseSRWLockShared(&read_records_lock); } } unlock;
    for (const ReadRecord& record : read_records)
        if (record.block == block) {
            count = record.count;
            bytes = record.bytes;
            return true;
        }
    return false;
}
HRESULT WINAPI read_provider(unsigned op, const void* raw)
{
    if (op == 0) return S_OK;  // Begin: the read runs after XAsyncBegin accepts the block
    if (!raw) return S_OK;
    const auto* data = static_cast<const AsyncProviderData*>(raw);
    auto* state = static_cast<ReadAsync*>(data->context);
    if (op == 2 && state && data->buffer && state->bytes <= data->buffer_size)
        x2w_save_pack::write(state->pieces, data->buffer);  // GetResult
    if (op == 4 && state) {  // Cleanup, including the runtime's cleanup of a failed read
        forget_read(state->block);
        delete state;
    }
    return S_OK;
}
// Names and bytes of one accepted read. The save lock is not held on return.
u64 snapshot_read(ReadAsync& state, u64 container, u64 names_pointer, std::uint32_t count)
{
    std::string container_name;
    {
        AcquireSRWLockShared(&save_lock);
        struct Unlock { ~Unlock() { ReleaseSRWLockShared(&save_lock); } } unlock;
        if (container >= save_container_handle && container - save_container_handle < max_save_handles)
            container_name = save_table.container_name(static_cast<std::size_t>(container - save_container_handle));
    }
    if (container_name.empty()) return e_invalidarg;
    if (count > SIZE_MAX / sizeof(const char*)) return e_invalidarg;
    if (count && x2w_local::probe_readable(static_cast<std::uintptr_t>(names_pointer), sizeof(const char*) * count) != sizeof(const char*) * count)
        return e_invalidarg;
    const auto* names = reinterpret_cast<const char* const*>(names_pointer);
    std::vector<std::string> wanted;
    wanted.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        std::string name;
        if (!names[i] || !read_save_name(reinterpret_cast<u64>(names[i]), name) || !x2w_save::valid_name(name))
            return e_invalidarg;
        wanted.push_back(std::move(name));
    }
    std::vector<std::vector<unsigned char>> data;
    const x2w_save::Status status = save_store().read_blobs(container_name, wanted, data);
    if (status != x2w_save::Status::ok) return save_status(status);
    if (data.size() != wanted.size() || data.size() > 0xffffffffu) return e_fail;
    state.pieces.resize(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        state.pieces[i].name = std::move(wanted[i]);
        state.pieces[i].bytes = std::move(data[i]);
    }
    state.count = static_cast<std::uint32_t>(state.pieces.size());
    if (!x2w_save_pack::bytes_needed(state.pieces, state.bytes)) return e_fail;
    return 0;
}
u64 async_read(int slot, u64 container, u64 names, u64 count_or_size, u64 block_or_buffer, u64 count_pointer)
{
    const QueryApi ask = load_real_query();
    if (!ask) return e_fail;
    void* service = nullptr;
    const auto query = ask(&submit_async_family, &submit_async_family, &service);
    if (FAILED(query)) return static_cast<unsigned>(query);
    if (!service) return e_fail;
    auto** methods = *static_cast<void***>(service);
    struct ApiRef {
        void* object;
        ULONG (WINAPI* release)(void*);
        ~ApiRef() { release(object); }
    } api{service, reinterpret_cast<ULONG (WINAPI*)(void*)>(methods[2])};
    if (slot == 22) {
        auto* async = reinterpret_cast<void*>(container);  // XGameSaveReadBlobDataResult(async, size, data, &count)
        if (!async) return e_fail;
        const auto out = [](u64 pointer, std::size_t bytes) {
            return pointer && x2w_local::writable(reinterpret_cast<void*>(pointer), bytes);
        };
        std::uint32_t count = 0;
        std::size_t bytes = 0;
        if (pending_read(async, count, bytes)) {
            // A short or unusable buffer must not consume the result or change the caller's memory.
            if (count_or_size < bytes) return small_buffer;
            if (!out(count_pointer, sizeof(std::uint32_t)) || !out(block_or_buffer, bytes) ||
                block_or_buffer % alignof(x2w_save_pack::Blob)) return e_invalidarg;
            using Result = HRESULT(WINAPI*)(void*, void*, const void*, size_t, void*, size_t*);
            const auto hr = reinterpret_cast<Result>(methods[11])(
                service, async, read_identity, static_cast<std::size_t>(count_or_size),
                reinterpret_cast<void*>(block_or_buffer), nullptr);
            if (FAILED(hr)) return static_cast<unsigned>(hr);
            *reinterpret_cast<std::uint32_t*>(count_pointer) = count;
            return 0;
        }
        // No payload was kept: a missing blob or other failure, or a read of no blobs. The caller's
        // buffer is not passed through, so a failure cannot change it. A success with no blobs
        // reports a count of zero.
        using Result = HRESULT(WINAPI*)(void*, void*, const void*, size_t, void*, size_t*);
        const auto hr = reinterpret_cast<Result>(methods[11])(service, async, read_identity, 0, nullptr, nullptr);
        if (FAILED(hr)) return static_cast<unsigned>(hr);
        if (!out(count_pointer, sizeof(std::uint32_t))) return e_invalidarg;
        *reinterpret_cast<std::uint32_t*>(count_pointer) = 0;
        return 0;
    }
    auto* async = reinterpret_cast<void*>(block_or_buffer);
    if (!async) return e_fail;
    ReadAsync* state;
    try {
        state = new ReadAsync;
    } catch (const std::bad_alloc&) {
        return e_outofmemory;
    }
    using Begin = HRESULT(WINAPI*)(void*, void*, void*, const void*, const char*, decltype(&read_provider));
    const auto begun = reinterpret_cast<Begin>(methods[7])(service, async, state, read_identity, read_identity, read_provider);
    if (FAILED(begun)) {
        delete state;  // rejection: the runtime has not taken the context, and nothing was read
        return static_cast<unsigned>(begun);
    }
    using Complete = void(WINAPI*)(void*, void*, HRESULT, size_t);
    const auto complete = reinterpret_cast<Complete>(methods[10]);
    try {
        state->block = async;
        const u64 status = snapshot_read(*state, container, names, static_cast<std::uint32_t>(count_or_size));
        if (status) {
            complete(service, async, static_cast<HRESULT>(status), 0);  // no payload; context is released by Cleanup
            return S_OK;
        }
        if (state->bytes) remember_read(*state);
        complete(service, async, S_OK, state->bytes);
        return S_OK;
    } catch (const std::bad_alloc&) {
        complete(service, async, static_cast<HRESULT>(e_outofmemory), 0);
        return S_OK;
    } catch (const std::filesystem::filesystem_error&) {
        complete(service, async, static_cast<HRESULT>(e_fail), 0);
        return S_OK;
    }
}
u64 complete_async(u64 async_block)
{
    Proxy* user = user_provider >= 0 ? proxies[user_provider] : nullptr;
    if (!user || !async_block) return 0x80004005;
    const LONG index = InterlockedIncrement(&tracked_count) - 1;
    if (index < 64) tracked_blocks[index] = async_block;
    // A local user object (X2W_RUNTIME_LOCAL) has no runtime operation to borrow: complete the block
    // through the XAsync service directly, as the local XUserAddAsync does.
    if (!user->real) return local_user_add(async_block);
    const Method add = reinterpret_cast<Method>((*reinterpret_cast<void***>(user->real))[7]);
    return add(user->real, 1 /* add default user silently */, async_block, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}
bool save_call(int slot, u64* args, u64* result)
{
    const auto out = [](u64 pointer, std::size_t bytes) { return pointer && x2w_local::writable(reinterpret_cast<void*>(pointer), bytes); };
    constexpr u64 ok = 0, blob_not_found = 0x80830008;
    *result = ok;
    switch (slot) {
    case 3:  // XGameSaveInitializeProvider(user, configuration id, sync on demand, &provider)
        if (out(args[3], 8)) *reinterpret_cast<u64*>(args[3]) = save_provider_handle;
        return true;
    case 4:  // XGameSaveInitializeProviderAsync(user, configuration id, sync on demand, async)
        *result = complete_async(args[3]);
        return true;
    case 5:  // XGameSaveInitializeProviderResult(async, &provider)
        if (out(args[1], 8)) *reinterpret_cast<u64*>(args[1]) = save_provider_handle;
        return true;
    case 6:  // close provider
        return true;
    case 17:  // XGameSaveCloseContainer(container): the table entry is reused once nothing needs it
        if (args[0] >= save_container_handle && args[0] - save_container_handle < max_save_handles) {
            AcquireSRWLockExclusive(&save_lock);
            save_table.close(static_cast<std::size_t>(args[0] - save_container_handle));
            ReleaseSRWLockExclusive(&save_lock);
        }
        return true;
    case 24:  // XGameSaveCloseUpdate(update)
        if (args[0] >= save_update_handle && args[0] - save_update_handle < max_save_handles) {
            AcquireSRWLockExclusive(&save_lock);
            save_table.end_update(static_cast<std::size_t>(args[0] - save_update_handle));
            ReleaseSRWLockExclusive(&save_lock);
        }
        return true;
    case 7:  // XGameSaveGetRemainingQuota(provider, &bytes)
        if (out(args[1], 8)) *reinterpret_cast<std::int64_t*>(args[1]) = 256ll << 20;
        return true;
    case 8:  // XGameSaveGetRemainingQuotaAsync(provider, async)
        *result = complete_async(args[1]);
        return true;
    case 9:  // XGameSaveGetRemainingQuotaResult(async, &bytes)
        if (out(args[1], 8)) *reinterpret_cast<std::int64_t*>(args[1]) = 256ll << 20;
        return true;
    case 12: case 13: case 14: case 15: case 18: case 19:  // container/blob queries: layout unknown, nothing reported
        return true;
    case 10: case 11: {  // XGameSaveDeleteContainer(provider, name), ...Async(provider, name, async)
        std::string name;
        if (read_save_name(args[1], name)) save_store().delete_container(name);
        if (slot == 11) *result = complete_async(args[2]);
        return true;
    }
    case 16: {  // XGameSaveCreateContainer(provider, name, &container)
        std::string name;
        u64 handle = 0;
        if (!read_save_name(args[1], name)) *result = e_invalidarg;
        else *result = open_container(name, &handle);
        if (!*result && out(args[2], 8)) *reinterpret_cast<u64*>(args[2]) = handle;
        return true;
    }
    case 20:  // XGameSaveReadBlobData(container, blobs, &count, size, data)
        *result = blob_not_found;
        return true;
    case 21:  // XGameSaveReadBlobDataAsync(container, names, count, async)
        *result = async_read(slot, args[0], args[1], args[2], args[3], 0);
        return true;
    case 22:  // XGameSaveReadBlobDataResult(async, size, data, &count)
        *result = async_read(slot, args[0], 0, args[1], args[2], args[3]);
        return true;
    case 23: {  // XGameSaveCreateUpdate(container, display name, &update)
        u64 handle = 0;
        if (args[0] >= save_container_handle && args[0] - save_container_handle < max_save_handles) {
            AcquireSRWLockExclusive(&save_lock);
            const long update = save_table.begin_update(static_cast<std::size_t>(args[0] - save_container_handle));
            ReleaseSRWLockExclusive(&save_lock);
            if (update >= 0) handle = save_update_handle + static_cast<u64>(update);
        }
        if (!handle) *result = e_outofmemory;
        else if (out(args[2], 8)) *reinterpret_cast<u64*>(args[2]) = handle;
        return true;
    }
    case 25: case 26: {
        // Hypothesis (public header order): SubmitBlobWrite(update, blob name, data, size) and
        // SubmitBlobDelete(update, blob name). The action is only kept with the update; nothing
        // reaches the disk before the update is submitted, and an update closed without a submit
        // leaves the save as it was. Only an open update is acted on; any other argument is
        // accepted and dropped as before.
        x2w_save::Action action;
        try {
            if (update_container(args[0]).empty() || !read_save_name(args[1], action.blob)) return true;
            action.remove = slot == 26;
            if (!x2w_save::valid_name(action.blob)) {
                *result = e_invalidarg;
                return true;
            }
            if (!action.remove) {
                const std::size_t size = static_cast<std::size_t>(args[3]);
                if (size > x2w_save::max_blob_bytes || (size && x2w_local::probe_readable(static_cast<std::uintptr_t>(args[2]), size) != size)) {
                    *result = e_invalidarg;
                    return true;
                }
                const auto* bytes = reinterpret_cast<const unsigned char*>(args[2]);
                if (size) action.data.assign(bytes, bytes + size);
            }
        } catch (const std::bad_alloc&) {
            *result = e_outofmemory;
            return true;
        }
        AcquireSRWLockExclusive(&save_lock);
        const bool staged = args[0] >= save_update_handle && save_table.stage(static_cast<std::size_t>(args[0] - save_update_handle), std::move(action));
        ReleaseSRWLockExclusive(&save_lock);
        if (!staged) *result = e_outofmemory;
        return true;
    }
    case 27:  // XGameSaveSubmitUpdate(update): publishes the staged actions, all or none
        *result = submit_update(args[0]);
        return true;
    case 28:  // XGameSaveSubmitUpdateAsync(update, async)
        *result = async_submit(slot, args[0], args[1]);
        return true;
    case 29:  // XGameSaveSubmitUpdateResult(async): consume this block's completion
        *result = async_submit(slot, 0, args[0]);
        return true;
    default:
        *result = 0x80004001;  // E_NOTIMPL
        return true;
    }
}

// ---- other areas, answered as an offline console with one installed package --------------------
// Which provider is which area, and what an offline console answers, follow the public VIctory
// project's runtime notes (github.com/Marvisak/VIctory, used as a reference for facts only).
unsigned env_number(const char* name, unsigned fallback, int base)
{
    char text[32]{};
    return GetEnvironmentVariableA(name, text, sizeof(text)) ? static_cast<unsigned>(std::strtoul(text, nullptr, base)) : fallback;
}
bool area_call(const GUID& provider_id, int slot, u64* args, u64* result)
{
    const unsigned long provider = provider_id.Data1;
    const auto out = [](u64 pointer, std::size_t bytes) { return pointer && x2w_local::writable(reinterpret_cast<void*>(pointer), bytes); };
    if (provider == 0x973a344e && slot == 3) {  // title id (a 32-bit value from the title's manifest: X2W_TITLE_ID, hex)
        const unsigned id = env_number("X2W_TITLE_ID", 0, 16);
        if (!id) return false;
        if (out(args[0], 4)) *reinterpret_cast<std::uint32_t*>(args[0]) = id;
        *result = 0;
        return true;
    }
    if (provider == 0xaf406016 && slot == 3) {  // the running package's identifier (size, buffer): the title id in hex
        char text[16];
        const int length = std::snprintf(text, sizeof(text), "%08X", env_number("X2W_TITLE_ID", 0, 16));
        if (args[0] <= static_cast<u64>(length) || !out(args[1], static_cast<std::size_t>(length) + 1)) {
            *result = 0x8007007A;
            return true;
        }
        std::memcpy(reinterpret_cast<void*>(args[1]), text, static_cast<std::size_t>(length) + 1);
        *result = 0;
        return true;
    }
    if (provider == 0xaf406016 && slot >= 5 && slot <= 38) {
        // The rest of the package area, for one fully installed package (the title itself).
        static const unsigned char monitor = 0;
        const u64 monitor_handle = reinterpret_cast<u64>(&monitor);
        constexpr u64 no_package = 0x89245208;
        *result = 0;
        switch (slot) {
        case 5:  // create an installation monitor (id, selector count, selectors, interval, queue, &monitor)
            if (out(args[5], 8)) *reinterpret_cast<u64*>(args[5]) = monitor_handle;
            return true;
        case 6: case 14: case 19:  // close monitor, change install order, uninstall chunks: accepted
            return true;
        case 7:  // progress (monitor, &progress): everything installed
            if (out(args[1], 32)) {
                const u64 bytes = u64{200} << 30;
                u64 words[4] = {bytes, bytes, bytes, 0x0101 /* launchable, completed */};
                std::memcpy(reinterpret_cast<void*>(args[1]), words, sizeof(words));
            }
            return true;
        case 9:  // register for progress changes (monitor, context, callback, &token): never changes
            if (out(args[3], 8)) *reinterpret_cast<u64*>(args[3]) = 1;
            return true;
        case 10:  // unregister
            *result = 1;
            return true;
        case 12:  // chunk availability (id, count, selectors, &availability): 0 = ready
            if (out(args[3], 4)) *reinterpret_cast<std::uint32_t*>(args[3]) = 0;
            return true;
        case 16:  // install chunks (id, count, selectors, interval, suppress confirmation, async)
            *result = complete_async(args[5]);
            return true;
        case 17:  // result of that: a monitor
            if (out(args[1], 8)) *reinterpret_cast<u64*>(args[1]) = monitor_handle;
            return true;
        case 18:  // download estimate (id, count, selectors, &bytes, &needs confirmation): nothing to download
            if (out(args[3], 8)) *reinterpret_cast<u64*>(args[3]) = 0;
            if (out(args[4], 1)) *reinterpret_cast<bool*>(args[4]) = false;
            return true;
        case 34:  // uninstall another package: refused
            return true;
        case 37:  // mount another package (id, async): none is installed
            *result = complete_async(args[1]);
            return true;
        case 38:
            *result = no_package;
            return true;
        default:
            return false;
        }
    }
    if (provider == 0xaf406016 && slot == 4) {  // is the title a package: yes
        *result = 1;
        return true;
    }
    if (provider == 0xaf406016 && slot == 39) {  // enumerate packages: none besides the title
        *result = 0;
        return true;
    }
    if (provider == 0x37e56907 && slot == 15) {  // connectivity hint: stack initialized, no connectivity
        if (out(args[0], 16)) {
            const std::uint32_t words[4] = {1 /* level: none */, 0 /* cost: unknown */, 0, 1 /* initialized; other flags clear */};
            std::memcpy(reinterpret_cast<void*>(args[0]), words, sizeof(words));
        }
        *result = 0;
        return true;
    }
    if (slot == 3 && IsEqualGUID(provider_id, store_provider_id)) {
        // XStoreCreateContext(user, &context). Vtable slot 3 of XStoreImpl
        // (0dd112ac-7c24-448c-b92b-3960fb5bd30c): QueryInterface, AddRef, Release, then this.
        // The public GDK takes the console user and writes an XStoreContextHandle. The VIctory
        // runtime notes (github.com/Marvisak/VIctory, facts only) answer it offline by writing one
        // context and returning success; catalog, licence and purchase methods stay unanswered.
        // Only the local user is answered. A different user is refused and the output is left
        // unchanged. A buffer that cannot be written is E_INVALIDARG.
        if (args[0] != local_user_handle) {
            *result = 0x80004005;
            return true;
        }
        if (!out(args[1], 8)) {
            *result = 0x80070057;
            return true;
        }
        *reinterpret_cast<u64*>(args[1]) = store_context_handle;
        *result = 0;
        return true;
    }
    return false;
}

// Source-built check (tests/startup_stub_consumer.cpp). Both switches are required, as they are
// for a title: the proxy is installed only for X2W_RUNTIME_PROXY, and this answer only for the
// fake user. Returns 0 when the local user receives the context, a mutated provider still
// fails with E_FAIL without writing, and a non-local user is refused the same way.
int store_method3_check()
{
    if (!x2w_local::env_on("X2W_RUNTIME_PROXY") || !fake_user()) return 2;
    if (provider_count >= max_providers) return 3;
    const int index = provider_count++;
    provider_ids[index] = store_provider_id;
    Proxy proxy{nullptr, nullptr, index};

    u64 context = 0;
    const u64 answered = forward<3>(&proxy, local_user_handle, reinterpret_cast<u64>(&context), 0, 0, 0, 0, 0, 0, 0, 0, 0);
    if (answered != 0 || context != store_context_handle) return 4;

    u64 untouched = 0x1111111111111111ull;
    provider_ids[index].Data4[7] ^= 1;  // same first word, different provider
    const u64 refused = forward<3>(&proxy, local_user_handle, reinterpret_cast<u64>(&untouched), 0, 0, 0, 0, 0, 0, 0, 0, 0);
    if (refused != 0x80004005 || untouched != 0x1111111111111111ull) return 5;
    provider_ids[index] = store_provider_id;

    u64 foreign_out = 0x1111111111111111ull;
    const u64 foreign_user = local_user_handle ^ 1ull;
    const u64 foreign = forward<3>(&proxy, foreign_user, reinterpret_cast<u64>(&foreign_out), 0, 0, 0, 0, 0, 0, 0, 0, 0);
    if (foreign != 0x80004005 || foreign_out != 0x1111111111111111ull) return 7;

    const u64 missing = forward<3>(&proxy, local_user_handle, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    if (missing != 0x80070057) return 6;

    const auto desktop = +[](void*, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64) -> u64 {
        return 0x1234;
    };
    void* methods[5] = {};
    methods[4] = reinterpret_cast<void*>(desktop);
    void** real = methods;
    proxy.real = &real;
    for (int position = 0; position < 4; ++position) {
        u64 arguments[4] = {};
        arguments[position] = context;
        const u64 blocked = forward<4>(&proxy, arguments[0], arguments[1], arguments[2], arguments[3], 0, 0, 0, 0, 0, 0, 0);
        if (blocked != 0x80004005) return 8;
    }
    // A foreign value still reaches the desktop implementation: the guard is token-specific.
    if (forward<4>(&proxy, context ^ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) != 0x1234) return 9;

    std::printf("store context %016llx\n", static_cast<unsigned long long>(context));
    std::printf("store refused %08llx\n", static_cast<unsigned long long>(refused & 0xffffffffu));
    std::printf("store user refused %08llx\n", static_cast<unsigned long long>(foreign & 0xffffffffu));
    return 0;
}

// Persistent local storage: the root of a drive (X2W_STORAGE_PATH, e.g. "S:\\", mapped by the run
// script onto a folder outside the game) with the quota from the manifest (X2W_STORAGE_MB).
struct StorageObject { void* const* vtable; };
// The value is read once. It gets a trailing separator when it has none (storage_root.h). A value
// that does not fit is refused with a log line and the default is used: handing the executable a
// cut-off or empty root would send its files somewhere else without a word.
const char* storage_path()
{
    static char path[x2w_startup::storage_root_capacity];
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    InitOnceExecuteOnce(&once, [](PINIT_ONCE, PVOID, PVOID*) -> BOOL {
        wchar_t value[x2w_startup::storage_root_capacity + 1];
        const DWORD capacity = static_cast<DWORD>(sizeof(value) / sizeof(*value));
        const DWORD length = GetEnvironmentVariableW(L"X2W_STORAGE_PATH", value, capacity);
        if (length >= capacity)
            x2w_local::logf("x2w-runtime X2W_STORAGE_PATH has %lu characters, more than the %zu supported: using S:\\",
                            static_cast<unsigned long>(length - 1), x2w_startup::storage_root_capacity - 2);
        else if (length && !x2w_startup::storage_root_utf8(value, path))
            x2w_local::logf("x2w-runtime X2W_STORAGE_PATH is invalid UTF-16 or does not fit as UTF-8 with its trailing separator: using S:\\");
        if (!path[0]) std::strcpy(path, "S:\\");
        return TRUE;
    }, nullptr, nullptr);
    return path;
}
HRESULT __stdcall storage_query(StorageObject* self, const GUID* iid, void** out)
{
    static LONG rejected;
    return local_query_interface(self, storage_pair.provider.Data1, &storage_pair.iid, iid, out, rejected);
}
u64 __stdcall storage_ref(StorageObject*) { return 1; }  // Process-lifetime object.
HRESULT __stdcall storage_path_size(StorageObject*, std::size_t* out)
{
    if (out) *out = std::strlen(storage_path()) + 1;
    x2w_local::logf("x2w-runtime storage path size -> %zu", std::strlen(storage_path()) + 1);
    return S_OK;
}
HRESULT __stdcall storage_get_path(StorageObject*, std::size_t size, char* buffer, std::size_t* used)
{
    const std::size_t need = std::strlen(storage_path()) + 1;
    if (used) *used = need;
    if (!buffer || size < need) return static_cast<HRESULT>(0x8007007A);
    std::memcpy(buffer, storage_path(), need);
    {
        // Whether the root exists on this PC: the title writes its settings below it itself, and a missing drive makes those writes fail.
        const DWORD attributes = GetFileAttributesA(storage_path());
        x2w_local::logf("x2w-runtime storage path -> %s (%s)", storage_path(),
                        attributes == INVALID_FILE_ATTRIBUTES ? "DOES NOT EXIST on this PC: the title's writes below it fail" : "exists");
    }
    return S_OK;
}
HRESULT __stdcall storage_space(StorageObject*, u64* out)
{
    const u64 total = u64{env_number("X2W_STORAGE_MB", 1024, 10)} << 20;
    if (out) {
        out[0] = total;  // writable now
        out[1] = total;  // left in the quota
        out[2] = 0;      // used
        out[3] = total;  // quota
    }
    return S_OK;
}
HRESULT __stdcall storage_prompt(StorageObject*, u64, u64 block) { return static_cast<HRESULT>(complete_async(block)); }
HRESULT __stdcall storage_prompt_result(StorageObject*, u64) { return S_OK; }
void* storage_object()
{
    static void* const table[8] = {
        reinterpret_cast<void*>(&storage_query), reinterpret_cast<void*>(&storage_ref), reinterpret_cast<void*>(&storage_ref),
        reinterpret_cast<void*>(&storage_path_size), reinterpret_cast<void*>(&storage_get_path), reinterpret_cast<void*>(&storage_space),
        reinterpret_cast<void*>(&storage_prompt), reinterpret_cast<void*>(&storage_prompt_result)};
    static StorageObject object{table};
    return &object;
}
}  // namespace

int x2w_store_method3_check() { return store_method3_check(); }

BOOL WINAPI DllMain(HINSTANCE self, DWORD reason, void*)
{
    if (reason == DLL_PROCESS_ATTACH && x2w_local::env_on("X2W_RUNTIME_PROXY")) {
        // Published IAT entries, the loader callback and both workers reference this module.
        HMODULE pinned = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                                reinterpret_cast<LPCWSTR>(self), &pinned))
            return FALSE;
        proxy_module = self;
        fill_slots(std::make_index_sequence<max_slots>{});
        patch_imports();
        // Heartbeat: every 10 s, the runtime methods called since the last report and how often, so
        // a title that keeps polling one call (waiting for something) shows it after the first-call
        // logs have gone quiet.
        if (HANDLE heartbeat = CreateThread(nullptr, 0, [](void*) -> DWORD {
            static LONG seen[max_providers][max_slots];
            for (int second = 10;; second += 10) {
                Sleep(10000);
                char line[1024];
                int n = std::snprintf(line, sizeof(line), "x2w-runtime calls %d s:", second);
                bool any = false;
                for (int p = 0; p < provider_count && p < max_providers; ++p)
                    for (int slot = 3; slot < max_slots; ++slot) {
                        const LONG now = call_counts[p][slot], delta = now - seen[p][slot];
                        seen[p][slot] = now;
                        if (delta > 0 && n < static_cast<int>(sizeof(line)) - 40) {
                            n += std::snprintf(line + n, sizeof(line) - n, " %08lx:%d x%ld", provider_ids[p].Data1, slot, delta);
                            any = true;
                        }
                    }
                if (any) x2w_local::logf("%s", line);
            }
        }, nullptr, 0, nullptr)) CloseHandle(heartbeat);
    }
    return TRUE;
}
