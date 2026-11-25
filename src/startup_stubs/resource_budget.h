#pragma once

#include <d3d12.h>
#include <cstdint>
#include <new>

namespace x2w_budget {

enum Category { placed, texture, buffer, upload, readback, heap, category_count };
inline constexpr const char* names[] = {"placed", "texture", "buffer", "upload", "readback", "heap"};

struct Counts {
    std::uint64_t bytes, objects;
};
struct Snapshot {
    Counts live[category_count], retained[category_count];
    std::uint64_t unmeasured;
};

inline bool enabled()
{
    static const bool on = [] {
        char value[2]{};
        return GetEnvironmentVariableA("X2W_D3D12_BUDGET", value, sizeof(value)) == 1 && value[0] == '1';
    }();
    return on;
}

struct Counter {
    LONG64 bytes{}, objects{};
};
inline SRWLOCK lock = SRWLOCK_INIT;
inline Counter live[category_count], retained[category_count];
inline LONG64 unmeasured{};

inline Snapshot snapshot()
{
    Snapshot result{};
    AcquireSRWLockShared(&lock);
    for (int i = 0; i < category_count; ++i) {
        result.live[i] = {static_cast<std::uint64_t>(live[i].bytes), static_cast<std::uint64_t>(live[i].objects)};
        result.retained[i] = {static_cast<std::uint64_t>(retained[i].bytes), static_cast<std::uint64_t>(retained[i].objects)};
    }
    result.unmeasured = static_cast<std::uint64_t>(InterlockedCompareExchange64(&unmeasured, 0, 0));
    ReleaseSRWLockShared(&lock);
    return result;
}

// The desktop object's private-data reference owns this ticket. Guest Release may move it
// to retained while the shim still owns the desktop object. The ticket holds no object reference.
class Ticket final : public IUnknown {
public:
    Ticket(Category category, LONG64 bytes, UINT users) : category_(category), bytes_(bytes), users_(users)
    {
        AcquireSRWLockExclusive(&lock);
        adjust(users_ ? live : retained, 1);
        ReleaseSRWLockExclusive(&lock);
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override
    {
        if (!out) return E_POINTER;
        *out = nullptr;
        if (iid != __uuidof(IUnknown)) return E_NOINTERFACE;
        *out = static_cast<IUnknown*>(this);
        AddRef();
        return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return InterlockedIncrement(&refs_); }
    ULONG STDMETHODCALLTYPE Release() override
    {
        const ULONG refs = InterlockedDecrement(&refs_);
        if (!refs) delete this;
        return refs;
    }

    void begin_use()
    {
        AcquireSRWLockExclusive(&lock);
        if (!users_++) {
            adjust(retained, -1);
            adjust(live, 1);
        }
        ReleaseSRWLockExclusive(&lock);
    }
    void end_use()
    {
        AcquireSRWLockExclusive(&lock);
        if (users_ && !--users_) {
            adjust(live, -1);
            adjust(retained, 1);
        }
        ReleaseSRWLockExclusive(&lock);
    }

private:
    ~Ticket()
    {
        AcquireSRWLockExclusive(&lock);
        adjust(users_ ? live : retained, -1);
        ReleaseSRWLockExclusive(&lock);
    }
    void adjust(Counter* counters, LONG64 delta) const
    {
        InterlockedAdd64(&counters[category_].bytes, bytes_ * delta);
        InterlockedAdd64(&counters[category_].objects, delta);
    }
    LONG refs_{1};
    Category category_;
    LONG64 bytes_;
    UINT users_;
};

inline Ticket* attach(ID3D12Object* object, Category category, UINT64 bytes, UINT users = 1)
{
    if (!bytes || bytes > INT64_MAX) {
        InterlockedIncrement64(&unmeasured);
        return nullptr;
    }
    constexpr GUID key{0x28213cf4, 0x25cf, 0x42be, {0x80, 0xf1, 0xc5, 0x9e, 0x97, 0xf7, 0x20, 0xd1}};
    auto* ticket = new (std::nothrow) Ticket(category, static_cast<LONG64>(bytes), users);
    if (ticket) {
        const HRESULT hr = object->SetPrivateDataInterface(key, ticket);
        ticket->Release();
        if (SUCCEEDED(hr)) return ticket;
    }
    InterlockedIncrement64(&unmeasured);
    return nullptr;
}

inline Ticket* resource(ID3D12Device* device, ID3D12Resource* resource, Category category)
{
    if (!enabled()) return nullptr;
    const auto desc = resource->GetDesc();
    return attach(resource, category, device->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes);
}

inline Ticket* committed(ID3D12Device* device, ID3D12Resource* resource, D3D12_HEAP_TYPE type)
{
    if (!enabled()) return nullptr;
    const auto desc = resource->GetDesc();
    const auto category = type == D3D12_HEAP_TYPE_UPLOAD ? upload : type == D3D12_HEAP_TYPE_READBACK ? readback :
        desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? buffer : texture;
    return attach(resource, category, device->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes);
}

inline Ticket* backing_heap(ID3D12Heap* object, UINT users = 1)
{
    return enabled() ? attach(object, heap, object->GetDesc().SizeInBytes, users) : nullptr;
}

}
