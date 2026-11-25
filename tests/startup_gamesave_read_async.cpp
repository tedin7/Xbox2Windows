// A blob committed in one process is read in a second through slots 21 and 22.
// The controlled XAsync stand-in can reject Begin and hold one block open. The system
// variant uses the installed gaming runtime and an immediate queue, and exits 77 when
// that runtime cannot be loaded.
#include "../src/startup_stubs/runtime_proxy.cpp"

#include <array>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

namespace {
using Provider = decltype(&read_provider);
struct Block {
    void* queue{};
    void* context{};
    void (WINAPI* callback)(Block*){};
    unsigned char internal[32]{};
};
struct ProviderData {
    void* async;
    std::size_t buffer_size;
    void* buffer;
    void* context;
};
struct Operations { void* methods[12]{}; } operations;
void** api = operations.methods;
bool refuse = false;
struct Operation {
    Block* block{};
    const void* id{};
    void* context{};
    Provider provider{};
    HRESULT result{E_PENDING};
    std::size_t bytes{};
};
std::array<Operation, 8> pending;
std::mutex pending_lock;
std::condition_variable progress;
Block* paused_block = nullptr;
bool entered = false;

ULONG WINAPI drop(void*) { return 1; }
HRESULT WINAPI begin(void*, Block* block, void* context, const void* id, const char*, Provider provider)
{
    if (id && std::strcmp(static_cast<const char*>(id), "local XGameSaveSubmitUpdate") == 0) return E_INVALIDARG;
    std::unique_lock lock(pending_lock);
    if (refuse) return E_ACCESSDENIED;
    for (const auto& op : pending)
        if (op.block == block) return E_INVALIDARG;
    for (auto& op : pending)
        if (!op.block) {
            op = {block, id, context, provider, E_PENDING, 0};
            if (block == paused_block) {
                entered = true;
                progress.notify_one();
                progress.wait(lock, [] { return paused_block == nullptr; });
            }
            lock.unlock();
            return provider(0, nullptr);
        }
    return E_OUTOFMEMORY;
}
void WINAPI complete(void*, Block* block, HRESULT status, std::size_t bytes)
{
    Provider provider{};
    void* context{};
    bool cleanup = false;
    {
        const std::lock_guard lock(pending_lock);
        for (auto& op : pending)
            if (op.block == block) {
                op.result = status;
                op.bytes = bytes;
                if (FAILED(status) || bytes == 0) {
                    cleanup = true;
                    provider = op.provider;
                    context = op.context;
                    op.context = nullptr;
                }
                break;
            }
    }
    if (block->callback) block->callback(block);
    if (cleanup && provider && context) {
        const ProviderData data{block, 0, nullptr, context};
        provider(4, &data);
    }
}
HRESULT WINAPI get_result_size(void*, Block* block, std::size_t* out)
{
    const std::lock_guard lock(pending_lock);
    for (const auto& op : pending)
        if (op.block == block) {
            if (out) *out = FAILED(op.result) ? 0 : op.bytes;
            return op.result;
        }
    return E_INVALIDARG;
}
HRESULT WINAPI get_result(void*, Block* block, const void* id, std::size_t size, void* buffer, std::size_t* used)
{
    Provider provider{};
    void* context{};
    HRESULT status = E_INVALIDARG;
    std::size_t bytes = 0;
    {
        const std::lock_guard lock(pending_lock);
        for (auto& op : pending)
            if (op.block == block) {
                if (op.id != id) return E_INVALIDARG;
                status = op.result;
                if (FAILED(status) || op.bytes == 0) return status;
                if (!op.context) return E_INVALIDARG;
                if (!buffer || size < op.bytes) {
                    provider = op.provider;
                    context = op.context;
                    op.context = nullptr;
                    status = E_NOT_SUFFICIENT_BUFFER;
                    break;
                }
                provider = op.provider;
                context = op.context;
                bytes = op.bytes;
                op.context = nullptr;
                break;
            }
        if (!provider) return status;
    }
    const ProviderData data{block, size, buffer, context};
    if (status != E_NOT_SUFFICIENT_BUFFER) {
        const HRESULT produced = provider(2, &data);
        if (FAILED(produced)) status = produced;
        else if (used) *used = bytes;
    }
    provider(4, &data);
    return status;
}
HRESULT WINAPI service(const GUID* cls, const GUID* iid, void** out)
{
    if (*cls != submit_async_family || *iid != submit_async_family) return E_NOINTERFACE;
    *out = &api;
    return S_OK;
}

template<class T> u64 ptr(T* p) { return reinterpret_cast<u64>(p); }
u64 call(int slot, u64 a = 0, u64 b = 0, u64 c = 0, u64 d = 0)
{
    u64 args[]{a, b, c, d, 0, 0, 0, 0, 0, 0, 0}, result{};
    if (!save_call(slot, args, &result)) std::abort();
    return result;
}
std::size_t packed_size(const char* name, std::size_t payload)
{
    return sizeof(x2w_save_pack::Blob) + std::strlen(name) + 1 + payload;
}
bool packed_ok(const unsigned char* buffer, std::uint32_t count, const char* name, const unsigned char* payload, std::size_t payload_size)
{
    if (count != 1) return false;
    const auto* blob = reinterpret_cast<const x2w_save_pack::Blob*>(buffer);
    const auto* base = reinterpret_cast<const unsigned char*>(blob);
    if (std::strcmp(blob->info.name, name) != 0 || blob->info.size != payload_size) return false;
    if (std::memcmp(blob->data, payload, payload_size) != 0) return false;
    return reinterpret_cast<const unsigned char*>(blob->info.name) == base + sizeof(x2w_save_pack::Blob) &&
        blob->data == reinterpret_cast<const unsigned char*>(blob->info.name) + std::strlen(name) + 1;
}
bool unchanged(const unsigned char* buffer, std::size_t bytes, unsigned char fill, std::uint32_t count, std::uint32_t expected)
{
    for (std::size_t i = 0; i < bytes; ++i)
        if (buffer[i] != fill) return false;
    return count == expected;
}

HRESULT (WINAPI* result_size)(void*, void*, std::size_t*) = nullptr;
void* result_service = nullptr;
struct Seen { unsigned callbacks{}; HRESULT size_hr{E_FAIL}; std::size_t bytes{}; u64 status{1}; std::uint32_t count{}; bool match{}; };
Seen seen;
constexpr unsigned char preference_bytes[]{0x41, 0x42, 0x43};
void WINAPI on_preference(Block* block)
{
    std::size_t bytes = 0;
    seen.size_hr = result_size(result_service, block, &bytes);
    seen.bytes = bytes;
    alignas(x2w_save_pack::Blob) unsigned char buffer[64];
    std::memset(buffer, 0xA5, sizeof buffer);
    std::uint32_t count = 0;
    seen.status = call(22, ptr(block), bytes, ptr(buffer), ptr(&count));
    seen.count = count;
    seen.match = bytes == packed_size("preference", 3) && packed_ok(buffer, count, "preference", preference_bytes, 3);
    for (std::size_t i = bytes; i < sizeof buffer && seen.match; ++i) seen.match = buffer[i] == 0xA5;
    ++seen.callbacks;
}

const char* preference_name = "preference";
const char* other_name = "other";
constexpr unsigned char other_bytes[]{0x10, 0x20};

int commit()
{
    u64 provider_handle{}, container{}, update{};
    const char* names[]{preference_name, other_name};
    if (call(3, local_user_handle, ptr("fixture"), 0, ptr(&provider_handle)) ||
        call(16, provider_handle, ptr("profile"), ptr(&container)) ||
        call(23, container, 0, ptr(&update)) ||
        call(25, update, ptr(names[0]), ptr(preference_bytes), 3) ||
        call(25, update, ptr(names[1]), ptr(other_bytes), 2) ||
        call(27, update)) return 1;
    std::puts("committed preference and other");
    return 0;
}

int read_phase(bool system)
{
    void* runtime = nullptr;
    void** methods = nullptr;
    struct Runtime {
        void* service{};
        void* queue{};
        void** methods{};
        ~Runtime()
        {
            if (methods && queue) reinterpret_cast<void(WINAPI*)(void*, void*)>(methods[17])(service, queue);
            if (methods && service) reinterpret_cast<ULONG(WINAPI*)(void*)>(methods[2])(service);
        }
    } held;
    if (system) {
        const auto dll = LoadLibraryExW(L"xgameruntime.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!dll) return 77;
        real_query = reinterpret_cast<QueryApi>(GetProcAddress(dll, "QueryApiImpl"));
        if (!real_query) return 77;
        if (FAILED(load_real_query()(&submit_async_family, &submit_async_family, &runtime)) || !runtime) return 77;
        methods = *static_cast<void***>(runtime);
        held.service = runtime;
        held.methods = methods;
        using CreateQueue = HRESULT(WINAPI*)(void*, unsigned, unsigned, void**);
        if (FAILED(reinterpret_cast<CreateQueue>(methods[12])(runtime, 3, 3, &held.queue))) return 77;
        result_size = reinterpret_cast<decltype(result_size)>(methods[4]);
        result_service = runtime;
    } else {
        operations.methods[2] = reinterpret_cast<void*>(&drop);
        operations.methods[4] = reinterpret_cast<void*>(&get_result_size);
        operations.methods[7] = reinterpret_cast<void*>(&begin);
        operations.methods[10] = reinterpret_cast<void*>(&complete);
        operations.methods[11] = reinterpret_cast<void*>(&get_result);
        real_query = service;
        result_size = reinterpret_cast<decltype(result_size)>(&get_result_size);
        result_service = &api;
    }

    u64 provider_handle{}, container{};
    if (call(3, local_user_handle, ptr("fixture"), 0, ptr(&provider_handle)) ||
        call(16, provider_handle, ptr("profile"), ptr(&container))) return 2;
    if (call(20, container) != 0x80830008) return 3;

    if (!system) {
        refuse = true;
        Block rejected{};
        const char* names[]{preference_name};
        if (call(21, container, ptr(names), 1, ptr(&rejected)) != static_cast<unsigned>(E_ACCESSDENIED)) return 4;
        if (call(22, ptr(&rejected), 64, 0, 0) != static_cast<unsigned>(E_INVALIDARG)) return 4;
        refuse = false;
    }

    seen = {};
    Block callback_block{};
    callback_block.queue = held.queue;
    callback_block.context = &seen;
    callback_block.callback = on_preference;
    const char* names[]{preference_name};
    if (call(21, container, ptr(names), 1, ptr(&callback_block)) || seen.callbacks != 1 || FAILED(seen.size_hr) ||
        seen.bytes != packed_size("preference", 3) || seen.status || seen.count != 1 || !seen.match) {
        std::printf("callback callbacks=%u size=%08lx bytes=%zu status=%08llx count=%u match=%d\n",
            seen.callbacks, static_cast<unsigned long>(seen.size_hr), seen.bytes, seen.status, seen.count, seen.match);
        return 5;
    }

    unsigned char missing_buffer[64];
    std::memset(missing_buffer, 0x3C, sizeof missing_buffer);
    std::uint32_t missing_count = 0x22222222;
    Block missing{};
    missing.queue = held.queue;
    const char* absent = "absent";
    const char* absent_names[]{absent};
    if (call(21, container, ptr(absent_names), 1, ptr(&missing)) ||
        call(22, ptr(&missing), sizeof missing_buffer, ptr(missing_buffer), ptr(&missing_count)) != 0x80830008 ||
        !unchanged(missing_buffer, sizeof missing_buffer, 0x3C, missing_count, 0x22222222)) return 6;

    const std::size_t need = packed_size("preference", 3);
    alignas(x2w_save_pack::Blob) unsigned char short_buffer[64];
    std::memset(short_buffer, 0x5C, sizeof short_buffer);
    std::uint32_t short_count = 0xA5A5A5A5;
    Block retry{};
    retry.queue = held.queue;
    if (call(21, container, ptr(names), 1, ptr(&retry))) return 7;
    if (call(22, ptr(&retry), need - 1, ptr(short_buffer), ptr(&short_count)) != 0x80830007 ||
        !unchanged(short_buffer, need - 1, 0x5C, short_count, 0xA5A5A5A5)) return 7;
    std::memset(short_buffer, 0xA5, sizeof short_buffer);
    short_count = 0;
    if (call(22, ptr(&retry), need, ptr(short_buffer), ptr(&short_count)) ||
        !packed_ok(short_buffer, short_count, "preference", preference_bytes, 3)) return 7;

    if (system) {
        Block first{}, second{};
        first.queue = second.queue = held.queue;
        const char* other_names[]{other_name};
        if (call(21, container, ptr(names), 1, ptr(&first)) || call(21, container, ptr(other_names), 1, ptr(&second))) return 8;
        alignas(x2w_save_pack::Blob) unsigned char second_buffer[64]{};
        alignas(x2w_save_pack::Blob) unsigned char first_buffer[64]{};
        std::uint32_t second_count = 0, first_count = 0;
        if (call(22, ptr(&second), packed_size("other", 2), ptr(second_buffer), ptr(&second_count)) ||
            call(22, ptr(&first), need, ptr(first_buffer), ptr(&first_count)) ||
            !packed_ok(second_buffer, second_count, "other", other_bytes, 2) ||
            !packed_ok(first_buffer, first_count, "preference", preference_bytes, 3)) return 8;
    } else {
        Block first{}, second{};
        paused_block = &first;
        u64 first_start = 1;
        std::thread worker([&] { first_start = call(21, container, ptr(names), 1, ptr(&first)); });
        {
            std::unique_lock lock(pending_lock);
            progress.wait(lock, [] { return entered; });
        }
        alignas(x2w_save_pack::Blob) unsigned char other_buffer[64];
        std::memset(other_buffer, 0x11, sizeof other_buffer);
        std::uint32_t other_count = 0x01010101;
        const char* other_names[]{other_name};
        const bool second_ok = call(21, container, ptr(other_names), 1, ptr(&second)) == 0 &&
            call(22, ptr(&second), packed_size("other", 2), ptr(other_buffer), ptr(&other_count)) == 0 &&
            packed_ok(other_buffer, other_count, "other", other_bytes, 2);
        unsigned char pending_buffer[8];
        std::memset(pending_buffer, 0x22, sizeof pending_buffer);
        std::uint32_t pending_count = 0x33333333;
        const bool still_pending = call(22, ptr(&first), need, ptr(pending_buffer), ptr(&pending_count)) == static_cast<unsigned>(E_PENDING) &&
            unchanged(pending_buffer, sizeof pending_buffer, 0x22, pending_count, 0x33333333);
        {
            const std::lock_guard lock(pending_lock);
            paused_block = nullptr;
        }
        progress.notify_one();
        worker.join();
        alignas(x2w_save_pack::Blob) unsigned char first_buffer[64]{};
        std::uint32_t first_count = 0;
        if (!second_ok || !still_pending || first_start ||
            call(22, ptr(&first), need, ptr(first_buffer), ptr(&first_count)) ||
            !packed_ok(first_buffer, first_count, "preference", preference_bytes, 3)) return 8;
    }
    std::puts(system ? "system read callback, size, bytes, missing, short, overlap"
                     : "controlled read refused Begin; callback, size, bytes, missing, short, overlap");
    return 0;
}

int run_child(const std::string& self, const std::string& arguments)
{
    std::string command = "\"" + self + "\" " + arguments;
#ifdef _WIN32
    command = "\"" + command + "\"";
#endif
    const int code = std::system(command.c_str());
#ifdef _WIN32
    return code;
#else
    return code == -1 ? -1 : (code >> 8) & 0xff;
#endif
}
}
int main(int argc, char** argv)
{
    const bool system = argc >= 2 && std::strcmp(argv[1], "system") == 0;
    const bool controlled = argc >= 2 && std::strcmp(argv[1], "controlled") == 0;
    if (!system && !controlled) return 2;
    if (argc >= 3 && std::strcmp(argv[2], "read") == 0) return read_phase(system);
    if (commit()) return 1;
    return run_child(argv[0], std::string(argv[1]) + " read");
}
