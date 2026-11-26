// X2W_D3D12_RESIDENCY=1: desktop texture residency under memory pressure. Included by d3d12_draw.inl right after
// texture_uploads(). Decisions live in texture_residency.h and texture_flights.h (unit tested without a GPU).
//
// Safety: a texture is a candidate only when it is in the retire class (X2W_D3D12_RELEASE_RETIRED=1: its desktop object
// dies under draw_lock together with its texture_uploads() entry, so the pointer is never stale here), was never
// written by the GPU, was never named by the title's own Evict / MakeResident, and no recording or submission that
// names it is unfinished (the fence rule). Evict and MakeResident are the public ID3D12Device methods.
// X2W_D3D12_RESIDENCY_LIMIT_MB=<n> replaces the DXGI local budget and counts only the textures this code knows about
// (a drill for WARP, which reports no useful budget).
#include "startup_stubs/texture_flights.h"
#include "startup_stubs/texture_residency.h"

bool residency_observe()
{
    static const bool on = std::getenv("X2W_D3D12_RESIDENCY") != nullptr && std::getenv("X2W_D3D12_RESIDENCY")[0] == '1';
    return on;
}

SRWLOCK residency_lock = SRWLOCK_INIT;  // a leaf: nothing is taken inside it
x2w_residency::Flights residency_flights;
std::unordered_set<const void*> residency_title_touched_set;
struct ResidencyQueue {
    ID3D12CommandQueue* queue;
    x2w_startup::ComPtr<ID3D12Fence> fence;
    UINT64 value;
};
std::vector<ResidencyQueue> residency_queues;
LONG residency_evicted, residency_made_resident, residency_refused, residency_pins;
LONG64 residency_evicted_bytes;

void residency_title_touched(ID3D12Resource* desktop)
{
    if (!residency_observe()) return;
    AcquireSRWLockExclusive(&residency_lock);
    try { residency_title_touched_set.insert(desktop); } catch (const std::bad_alloc&) {}
    ReleaseSRWLockExclusive(&residency_lock);
}

void residency_discard_list(Wrapper* list)
{
    if (!residency_observe()) return;
    AcquireSRWLockExclusive(&residency_lock);
    residency_flights.discard(list);
    ReleaseSRWLockExclusive(&residency_lock);
}

// After ExecuteCommandLists: one fence value per queue behind the lists' recorded textures.
void residency_note_submission(ID3D12CommandQueue* queue, const std::vector<Wrapper*>& lists)
{
    if (!residency_observe()) return;
    std::vector<const void*> keys(lists.begin(), lists.end());
    AcquireSRWLockExclusive(&residency_lock);
    std::size_t index = 0;
    while (index < residency_queues.size() && residency_queues[index].queue != queue) ++index;
    bool ok = false;
    try {
        if (index == residency_queues.size()) {
            ResidencyQueue entry{queue, {}, 0};
            x2w_startup::ComPtr<ID3D12Device> device;
            if (SUCCEEDED(queue->GetDevice(IID_PPV_ARGS(&device)))) device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&entry.fence));
            residency_queues.push_back(std::move(entry));
        }
        ResidencyQueue& q = residency_queues[index];
        if (q.fence && SUCCEEDED(queue->Signal(q.fence.Get(), q.value + 1))) {
            ++q.value;
            ok = true;
        }
        residency_flights.submit(keys, index, ok ? q.value : 0, ok);
    } catch (const std::bad_alloc&) {
    }
    ReleaseSRWLockExclusive(&residency_lock);
}

// upload_texture (draw_lock held): the recording names the texture; bring it back first if it was evicted.
// Returns false when MakeResident failed (the texture cannot be used now).
bool residency_use(Wrapper* list, ID3D12Resource* texture, bool retire_class, TextureUpload& state, LONG frame)
{
    state.evictable = retire_class;
    x2w_residency::Texture t;
    t.last_use = state.last_use;
    t.resident = state.resident;
    t.pinned = state.pinned;
    t.evicted_at = state.evicted_at;
    const bool bring_back = x2w_residency::note_use(t, frame, x2w_residency::Config{});
    if (t.pinned && !state.pinned) InterlockedIncrement(&residency_pins);
    state.pinned = t.pinned;
    state.resident = t.resident;
    AcquireSRWLockExclusive(&residency_lock);
    try { residency_flights.note_use(list, reinterpret_cast<std::uint64_t>(texture)); } catch (const std::bad_alloc&) { state.evictable = false; }
    ReleaseSRWLockExclusive(&residency_lock);
    if (!bring_back) return true;
    ID3D12Pageable* pageable = texture;
    if (FAILED(device_of(the_device)->MakeResident(1, &pageable))) {
        state.resident = false;
        return false;
    }
    InterlockedIncrement(&residency_made_resident);
    return true;
}

bool residency_local_memory(unsigned long long& used, unsigned long long& budget)
{
    x2w_startup::ComPtr<IDXGIFactory4> factory;
    x2w_startup::ComPtr<IDXGIAdapter3> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return false;
    if (FAILED(factory->EnumAdapterByLuid(device_of(the_device)->GetAdapterLuid(), IID_PPV_ARGS(&adapter)))) return false;
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) return false;
    used = info.CurrentUsage;
    budget = info.Budget;
    return true;
}

// Every 60 frames (render mode): evict the idle textures the plan names while the local group is over its high mark.
void residency_tick(LONG frame)
{
    if (!residency_observe()) return;
    static x2w_residency::Pressure pressure;
    const x2w_residency::Config config;
    AcquireSRWLockExclusive(&draw_lock);
    std::vector<x2w_residency::Texture> known;
    unsigned long long used = 0, budget = 0;
    const char* limit = std::getenv("X2W_D3D12_RESIDENCY_LIMIT_MB");
    AcquireSRWLockExclusive(&residency_lock);
    for (std::size_t i = 0; i < residency_queues.size(); ++i)
        if (residency_queues[i].fence) residency_flights.complete(i, residency_queues[i].fence->GetCompletedValue());
    try {
        for (const auto& entry : texture_uploads()) {
            const TextureUpload& s = entry.second;
            if (!s.uploads || !s.bytes) continue;
            x2w_residency::Texture t;
            t.id = reinterpret_cast<std::uint64_t>(entry.first);
            t.bytes = s.bytes;
            t.last_use = s.last_use;
            t.resident = s.resident;
            t.pinned = s.pinned;
            t.evicted_at = s.evicted_at;
            t.in_flight = residency_flights.busy(t.id);
            t.unsafe = !s.evictable || s.gpu_written || residency_title_touched_set.count(entry.first) != 0;
            if (limit && t.resident) used += t.bytes;
            known.push_back(t);
        }
    } catch (const std::bad_alloc&) {
        known.clear();
    }
    ReleaseSRWLockExclusive(&residency_lock);
    bool have = false;
    if (limit) {
        budget = static_cast<unsigned long long>(std::strtoull(limit, nullptr, 10)) << 20;
        have = budget != 0;
    } else {
        have = residency_local_memory(used, budget);
    }
    if (have && !known.empty()) {
        const bool active = pressure.update(used, budget, config);
        for (const std::uint64_t id : x2w_residency::plan_evictions(known, used, budget, frame, config, active)) {
            auto* texture = reinterpret_cast<ID3D12Resource*>(id);
            D3D12_HEAP_PROPERTIES heap{};
            D3D12_HEAP_FLAGS flags{};
            ID3D12Pageable* pageable = texture;
            auto found = texture_uploads().find(texture);
            if (found == texture_uploads().end() || FAILED(texture->GetHeapProperties(&heap, &flags)) ||  // placed resources fail here
                FAILED(device_of(the_device)->Evict(1, &pageable))) {
                InterlockedIncrement(&residency_refused);
                continue;
            }
            found->second.resident = false;
            found->second.evicted_at = frame;
            InterlockedIncrement(&residency_evicted);
            InterlockedAdd64(&residency_evicted_bytes, static_cast<LONG64>(found->second.bytes));
        }
    }
    ReleaseSRWLockExclusive(&draw_lock);
}
