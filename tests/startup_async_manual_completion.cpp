// One synthetic XAsync operation whose completion port is dispatched by hand (#870).
//
// The test reports four observations separately: the Begin return, the non-waiting status, the completion
// callback count and the result the consumer retrieves in its callback. Each is printed for three
// phases:
//   pending   accepted, not yet completed;
//   unpumped  the provider has completed on another thread. The work port has been dispatched once,
//             and the completion port has not;
//   dispatched  one dispatch of the completion port.
// Before dispatch, the callback count must be 0 and nothing is consumed. One dispatch must run exactly one
// callback, and that callback must retrieve the synthetic payload. A second dispatch must find nothing. A
// second block on an immediate queue is the contrast: its callback runs without any dispatch.
//
// system: the installed Windows gaming runtime's XAsync service (073b7dcb), on a queue whose work and
//   completion ports are both manual. The test exits 77 when that runtime or the queue is unavailable.
//   It prints the runtime's status and result-size answers but does not assert them.
// controlled: a stand-in service in this file that keeps a completed block's callback on the manual
//   completion port until that port is dispatched.
// inline-control: the same stand-in runs the callback inside Complete, as an immediate port would. The
//   unpumped check must then print FAIL.
//
// No account, title or application input is used.
// XAsyncGetStatus      https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xasync/functions/xasyncgetstatus
// XAsyncComplete       https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xasyncprovider/functions/xasynccomplete
// XTaskQueueDispatch   https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xtaskqueue/functions/xtaskqueuedispatch
// XTaskQueueCreate     https://learn.microsoft.com/en-us/gaming/gdk/docs/reference/system/xtaskqueue/functions/xtaskqueuecreate
// Async interface slot numbers: https://github.com/ivall/xcl/blob/64b3f35599bb7102f55d640e1a791e7d6dd3b175/crates/xcl/src/hle/xgameruntime/gdk_table.rs#L57-L70
#include <windows.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>

namespace {
constexpr GUID async_family{0x073b7dcb, 0x1fcf, 0x4030, {0x94, 0xbe, 0xe3, 0xc9, 0xeb, 0x62, 0x34, 0x28}};
enum Slot : int {
    slot_release = 2, slot_get_status = 3, slot_get_result_size = 4, slot_begin = 7, slot_complete = 10,
    slot_get_result = 11, slot_create_queue = 12, slot_dispatch = 16, slot_close_queue = 17, slot_count = 31
};
enum : std::uint32_t { mode_manual = 0, mode_immediate = 3 };
enum : std::uint32_t { port_work = 0, port_completion = 1 };
enum : std::uint32_t { op_begin = 0, op_do_work = 1, op_get_result = 2, op_cancel = 3, op_cleanup = 4 };

struct Block {
    void* queue{};
    void* context{};
    void (WINAPI* callback)(Block*){};
    unsigned char internal[32]{};
};
struct ProviderData {
    Block* async;
    std::size_t buffer_size;
    void* buffer;
    void* context;
};
using Provider = HRESULT (WINAPI*)(std::uint32_t, const ProviderData*);

using QueryApi = HRESULT (WINAPI*)(const GUID*, const GUID*, void**);
using ReleaseFn = ULONG (WINAPI*)(void*);
using GetStatusFn = HRESULT (WINAPI*)(void*, Block*, bool);
using ResultSizeFn = HRESULT (WINAPI*)(void*, Block*, std::size_t*);
using BeginFn = HRESULT (WINAPI*)(void*, Block*, void*, const void*, const char*, Provider);
using CompleteFn = void (WINAPI*)(void*, Block*, HRESULT, std::size_t);
using ResultFn = HRESULT (WINAPI*)(void*, Block*, const void*, std::size_t, void*, std::size_t*);
using CreateQueueFn = HRESULT (WINAPI*)(void*, std::uint32_t, std::uint32_t, void**);
using DispatchFn = bool (WINAPI*)(void*, void*, std::uint32_t, std::uint32_t);
using CloseQueueFn = void (WINAPI*)(void*, void*);

// The service every call goes through: the installed runtime's interface or the stand-in below.
struct Service {
    void* self{};
    void** methods{};
    template<class F> F at(int slot) const { return reinterpret_cast<F>(methods[slot]); }
};
Service service;

constexpr char identity[] = "x2w manual completion";
constexpr unsigned char payload[8]{0x58, 0x32, 0x57, 0x2d, 0x61, 0x64, 0x64, 0x21};
std::atomic<unsigned> provider_begins{0}, provider_results{0}, provider_cleanups{0};

HRESULT WINAPI provider(std::uint32_t op, const ProviderData* data)
{
    switch (op) {
    case op_begin:
        ++provider_begins;
        return S_OK;
    case op_get_result:
        if (!data || !data->buffer || data->buffer_size < sizeof payload) return E_NOT_SUFFICIENT_BUFFER;
        std::memcpy(data->buffer, payload, sizeof payload);
        ++provider_results;
        return S_OK;
    case op_cleanup:
        ++provider_cleanups;
        return S_OK;
    default:
        return S_OK;
    }
}

struct Seen {
    std::atomic<unsigned> callbacks{0};
    HRESULT result{E_FAIL};
    std::size_t used{};
    bool match{};
};

void WINAPI on_complete(Block* block)
{
    auto* seen = static_cast<Seen*>(block->context);
    unsigned char buffer[16];
    std::memset(buffer, 0xA5, sizeof buffer);
    std::size_t used = 0;
    seen->result = service.at<ResultFn>(slot_get_result)(service.self, block, identity, sizeof buffer, buffer, &used);
    seen->used = used;
    bool match = SUCCEEDED(seen->result) && used == sizeof payload && std::memcmp(buffer, payload, sizeof payload) == 0;
    for (std::size_t i = sizeof payload; i < sizeof buffer && match; ++i) match = buffer[i] == 0xA5;
    seen->match = match;
    ++seen->callbacks;
}

// Stand-in XAsync service. A block's status stays E_PENDING until Complete. Complete then records the
// status and size. On a manual completion port it queues the callback, and on an immediate port, or under
// inline-control, it runs the callback at once. Get-result copies through the provider and cleans up once.
namespace standin {
struct Queue {
    std::uint32_t work{}, completion{};
    std::deque<Block*> ready;
};
struct Operation {
    Block* block{};
    const void* id{};
    void* context{};
    Provider provider{};
    HRESULT status{E_PENDING};
    std::size_t bytes{};
    bool consumed{};
};
std::mutex lock;
std::array<Operation, 4> operations;
std::array<Queue, 2> queues;
std::size_t queues_used = 0;
bool run_inline = false;
void* methods[slot_count]{};
struct Object { void** vtable; } object{methods};

Operation* find(Block* block)
{
    for (auto& op : operations)
        if (op.block == block) return &op;
    return nullptr;
}
ULONG WINAPI release(void*) { return 1; }
HRESULT WINAPI get_status(void*, Block* block, bool)
{
    const std::lock_guard guard(lock);
    const auto* op = find(block);
    return op ? op->status : E_INVALIDARG;
}
HRESULT WINAPI get_result_size(void*, Block* block, std::size_t* out)
{
    const std::lock_guard guard(lock);
    const auto* op = find(block);
    if (!op) return E_INVALIDARG;
    if (op->status == E_PENDING) return E_PENDING;
    // The installed runtime reports 0 bytes once the result has been retrieved.
    if (out) *out = SUCCEEDED(op->status) && !op->consumed ? op->bytes : 0;
    return op->status;
}
HRESULT WINAPI begin(void*, Block* block, void* context, const void* id, const char*, Provider p)
{
    {
        const std::lock_guard guard(lock);
        if (find(block)) return E_INVALIDARG;
        Operation* slot = find(nullptr);
        if (!slot) return E_OUTOFMEMORY;
        *slot = {block, id, context, p, E_PENDING, 0, false};
    }
    const ProviderData data{block, 0, nullptr, context};
    return p(op_begin, &data);
}
void WINAPI complete(void*, Block* block, HRESULT status, std::size_t bytes)
{
    bool now = false;
    {
        const std::lock_guard guard(lock);
        auto* op = find(block);
        if (!op || op->status != E_PENDING) return;
        op->status = status;
        op->bytes = bytes;
        auto* queue = static_cast<Queue*>(block->queue);
        now = run_inline || !queue || queue->completion != mode_manual;
        if (!now && block->callback) queue->ready.push_back(block);
    }
    if (now && block->callback) block->callback(block);
}
HRESULT WINAPI get_result(void*, Block* block, const void* id, std::size_t size, void* buffer, std::size_t* used)
{
    Provider p{};
    void* context{};
    std::size_t bytes{};
    {
        const std::lock_guard guard(lock);
        auto* op = find(block);
        if (!op) return E_INVALIDARG;
        if (op->status == E_PENDING) return E_PENDING;
        if (op->id != id) return E_INVALIDARG;
        if (FAILED(op->status)) return op->status;
        if (op->consumed) return E_INVALIDARG;
        if (!buffer || size < op->bytes) return E_NOT_SUFFICIENT_BUFFER;
        op->consumed = true;
        p = op->provider;
        context = op->context;
        bytes = op->bytes;
    }
    const ProviderData data{block, size, buffer, context};
    const HRESULT produced = p(op_get_result, &data);
    p(op_cleanup, &data);
    if (SUCCEEDED(produced) && used) *used = bytes;
    return produced;
}
HRESULT WINAPI create_queue(void*, std::uint32_t work, std::uint32_t completion, void** out)
{
    const std::lock_guard guard(lock);
    if (!out || queues_used == queues.size()) return E_OUTOFMEMORY;
    queues[queues_used] = {work, completion, {}};
    *out = &queues[queues_used++];
    return S_OK;
}
bool WINAPI dispatch(void*, void* handle, std::uint32_t port, std::uint32_t)
{
    Block* block{};
    {
        const std::lock_guard guard(lock);
        auto* queue = static_cast<Queue*>(handle);
        if (!queue || port != port_completion || queue->ready.empty()) return false;
        block = queue->ready.front();
        queue->ready.pop_front();
    }
    block->callback(block);
    return true;
}
void WINAPI close_queue(void*, void*) {}

void install(bool inline_completion)
{
    run_inline = inline_completion;
    methods[slot_release] = reinterpret_cast<void*>(&release);
    methods[slot_get_status] = reinterpret_cast<void*>(&get_status);
    methods[slot_get_result_size] = reinterpret_cast<void*>(&get_result_size);
    methods[slot_begin] = reinterpret_cast<void*>(&begin);
    methods[slot_complete] = reinterpret_cast<void*>(&complete);
    methods[slot_get_result] = reinterpret_cast<void*>(&get_result);
    methods[slot_create_queue] = reinterpret_cast<void*>(&create_queue);
    methods[slot_dispatch] = reinterpret_cast<void*>(&dispatch);
    methods[slot_close_queue] = reinterpret_cast<void*>(&close_queue);
    service = {&object, methods};
}
}

struct Phase {
    HRESULT status{};
    HRESULT size_hr{};
    std::size_t size{};
    unsigned callbacks{};
};
Phase observe(Block* block, const Seen& seen)
{
    Phase phase;
    phase.status = service.at<GetStatusFn>(slot_get_status)(service.self, block, false);
    phase.size_hr = service.at<ResultSizeFn>(slot_get_result_size)(service.self, block, &phase.size);
    phase.callbacks = seen.callbacks.load();
    return phase;
}
void print(const char* name, const Phase& phase, const Seen& seen)
{
    std::printf("%-10s status=0x%08lx result_size=0x%08lx bytes=%zu callbacks=%u consumed=%s\n", name,
        static_cast<unsigned long>(phase.status), static_cast<unsigned long>(phase.size_hr),
        SUCCEEDED(phase.size_hr) ? phase.size : 0, phase.callbacks,
        seen.callbacks ? (seen.match ? "expected" : "mismatch") : "no");
}

int fail(const char* what)
{
    std::printf("FAIL: %s\n", what);
    return 1;
}

int run(bool system)
{
    const auto create = service.at<CreateQueueFn>(slot_create_queue);
    const auto begin = service.at<BeginFn>(slot_begin);
    const auto complete = service.at<CompleteFn>(slot_complete);
    const auto dispatch = service.at<DispatchFn>(slot_dispatch);

    struct Queues {
        void* manual{};
        void* immediate{};
        ~Queues()
        {
            const auto close = service.at<CloseQueueFn>(slot_close_queue);
            if (manual) close(service.self, manual);
            if (immediate) close(service.self, immediate);
        }
    } queues;
    const HRESULT manual_hr = create(service.self, mode_manual, mode_manual, &queues.manual);
    if (FAILED(manual_hr) || !queues.manual) {
        std::printf("manual queue creation returned 0x%08lx\n", static_cast<unsigned long>(manual_hr));
        return fail("manual completion queue creation");
    }

    Seen seen;
    Block block{};
    block.queue = queues.manual;
    block.context = &seen;
    block.callback = on_complete;
    const HRESULT started = begin(service.self, &block, nullptr, identity, identity, provider);
    std::printf("start      begin=0x%08lx provider_begin=%u\n", static_cast<unsigned long>(started), provider_begins.load());
    if (started != S_OK) return fail("Begin did not accept the block");

    const Phase pending = observe(&block, seen);
    print("pending", pending, seen);

    // Complete on a provider thread. The work port is empty because Begin did not schedule DoWork.
    std::thread([&] { complete(service.self, &block, S_OK, sizeof payload); }).join();
    const bool work_dispatched = dispatch(service.self, queues.manual, port_work, 0);
    const Phase unpumped = observe(&block, seen);
    print("unpumped", unpumped, seen);
    std::printf("unpumped   work_port_dispatched=%d provider_result=%u cleanup=%u\n", work_dispatched,
        provider_results.load(), provider_cleanups.load());

    if (work_dispatched) return fail("an unexpected work-port callback ran");
    if (unpumped.callbacks != 0) return fail("callback ran before the completion port was dispatched");
    if (provider_results != 0) return fail("provider result was read before completion dispatch");
    const bool dispatched = dispatch(service.self, queues.manual, port_completion, 0);
    const unsigned after_one = seen.callbacks.load();
    const bool again = dispatch(service.self, queues.manual, port_completion, 0);
    const Phase final_phase = observe(&block, seen);
    print("dispatched", final_phase, seen);
    std::printf("dispatched completion_dispatch=%d second_dispatch=%d result=0x%08lx used=%zu provider_result=%u cleanup=%u\n",
        dispatched, again, static_cast<unsigned long>(seen.result), seen.used, provider_results.load(),
        provider_cleanups.load());

    // Contrast: the same operation on an immediate completion port runs its callback without dispatch.
    Seen contrast;
    Block immediate_block{};
    const HRESULT immediate_hr = create(service.self, mode_immediate, mode_immediate, &queues.immediate);
    if (FAILED(immediate_hr) || !queues.immediate) {
        std::printf("immediate queue creation returned 0x%08lx\n", static_cast<unsigned long>(immediate_hr));
        return fail("immediate completion queue creation");
    }
    immediate_block.queue = queues.immediate;
    immediate_block.context = &contrast;
    immediate_block.callback = on_complete;
    if (begin(service.self, &immediate_block, nullptr, identity, identity, provider) != S_OK)
        return fail("Begin did not accept the immediate-port block");
    std::thread([&] { complete(service.self, &immediate_block, S_OK, sizeof payload); }).join();
    std::printf("immediate  callbacks=%u consumed=%s\n", contrast.callbacks.load(),
        contrast.callbacks ? (contrast.match ? "expected" : "mismatch") : "no");
    if (contrast.callbacks != 1 || !contrast.match)
        return fail("the immediate-port contrast did not run its callback with the payload");

    if (pending.status != E_PENDING || pending.callbacks != 0)
        return fail("status before Complete was not E_PENDING with no callback");
    if (!dispatched || after_one != 1)
        return fail("one completion-port dispatch did not run exactly one callback");
    if (seen.result != S_OK || !seen.match)
        return fail("the callback did not retrieve the synthetic payload");
    if (again || seen.callbacks != 1)
        return fail("a second completion-port dispatch found more work");
    if (unpumped.status != S_OK && (!system || unpumped.status != E_PENDING))
        return fail("unexpected status after Complete");
    if (final_phase.status != S_OK)
        return fail("completion was dispatched but the operation did not succeed");
    if (!system && (unpumped.size != sizeof payload || final_phase.size != 0))
        return fail("stand-in result size was not 8 before and 0 after retrieval");
    std::puts(system ? "PASS: system manual completion port" : "PASS: controlled manual completion port");
    return 0;
}
}

int main(int argc, char** argv)
{
    const char* mode = argc >= 2 ? argv[1] : "";
    const bool system = std::strcmp(mode, "system") == 0;
    if (!system && std::strcmp(mode, "controlled") != 0 && std::strcmp(mode, "inline-control") != 0) {
        std::fputs("usage: startup_async_manual_completion system|controlled|inline-control\n", stderr);
        return 2;
    }
    if (!system) {
        standin::install(std::strcmp(mode, "inline-control") == 0);
        return run(false);
    }
    const auto dll = LoadLibraryExW(L"xgameruntime.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!dll) {
        std::puts("Windows gaming runtime unavailable");
        return 77;
    }
    const auto query = reinterpret_cast<QueryApi>(GetProcAddress(dll, "QueryApiImpl"));
    void* object = nullptr;
    if (!query || FAILED(query(&async_family, &async_family, &object)) || !object) {
        std::puts("Windows gaming runtime XAsync service unavailable");
        return 77;
    }
    service = {object, *static_cast<void***>(object)};
    const int code = run(true);
    service.at<ReleaseFn>(slot_release)(object);
    return code;
}
