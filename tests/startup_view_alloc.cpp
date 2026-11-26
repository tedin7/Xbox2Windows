// A view is published into three containers (view_records, view_materialized, view_ids). When one of them cannot
// allocate, the stand-in must record nothing, untrack the destination descriptor and keep running (#1199,
// docs/startup-runtime.md#d3d12-view-allocation-failure). The fixture-only export x2w_d3d12_fail_view_allocations
// makes the allocations at one site fail.
//
//   startup_view_alloc <warp|hardware> <records|materialized|ids> [control]
//     (X2W_D3D12_BUDGET=1 X2W_D3D12_RELEASE_RETIRED=1 X2W_D3D12_PERMISSIVE=1 X2W_D3D12_RENDER=1 X2W_TRAP_LOG=<file>)
//     "control" never arms the failure, so the armed expectations must fail and the fixture must print FAIL.
#include <windows.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

using Microsoft::WRL::ComPtr;

struct ViewStoreProbe {
    unsigned long long records, materialized, ids, records_capacity, materialized_capacity;
    unsigned shadow, view_refs, retire_when_idle;
};
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) void x2w_d3d12_fail_view_allocations(int, LONG);
extern "C" __declspec(dllimport) void x2w_d3d12_view_probe(unsigned long long, void*, ViewStoreProbe*);

namespace {

constexpr std::uint32_t token_magic = 0x56573258;  // "X2WV"
constexpr LONG failing_calls = 3;
constexpr UINT fill_limit = 100000;

template<class Result = HRESULT, class... Args>
Result invoke(IUnknown* object, unsigned slot, Args... args)
{
    using Call = Result (WINAPI*)(IUnknown*, Args...);
    return reinterpret_cast<Call>((*reinterpret_cast<void***>(object))[slot])(object, args...);
}

int failures;

void expect(bool ok, const char* what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

ComPtr<IDXGIAdapter1> pick_adapter(IDXGIFactory4* factory, bool warp)
{
    ComPtr<IDXGIAdapter1> adapter;
    if (warp) {
        factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
        return adapter;
    }
    wchar_t name[128]{};
    GetEnvironmentVariableW(L"XBOX2WINDOWS_ADAPTER", name, static_cast<DWORD>(std::size(name)));
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i, adapter.Reset()) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            (!name[0] || std::wcsstr(desc.Description, name)) &&
            SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr)))
            return adapter;
    }
    return nullptr;
}

std::string read_log()
{
    wchar_t path[MAX_PATH];
    const DWORD length = GetEnvironmentVariableW(L"X2W_TRAP_LOG", path, MAX_PATH);
    std::string text;
    if (!length || length >= MAX_PATH) return text;
    const HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return text;
    char chunk[4096];
    DWORD got = 0;
    while (ReadFile(file, chunk, sizeof(chunk), &got, nullptr) && got) text.append(chunk, got);
    CloseHandle(file);
    return text;
}

// The view id + 1 in the token the stand-in writes into title descriptor memory, or 0 when it holds no token.
std::uint32_t token(UINT64 handle)
{
    std::uint32_t words[2];
    std::memcpy(words, reinterpret_cast<const void*>(handle), sizeof(words));
    return words[0] == token_magic ? words[1] + 1 : 0;
}

ViewStoreProbe probe(UINT64 handle, void* resource = nullptr)
{
    ViewStoreProbe p{};
    x2w_d3d12_view_probe(handle, resource, &p);
    return p;
}

}  // namespace

int main(int argc, char** argv)
{
    const char* const sites[] = {"records", "materialized", "ids"};
    int site = -1;
    for (int i = 0; argc >= 3 && i < 3; ++i)
        if (!std::strcmp(argv[2], sites[i])) site = i;
    if (argc < 3 || argc > 4 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "hardware")) || site < 0 ||
        (argc == 4 && std::strcmp(argv[3], "control")))
        return 2;
    const bool control = argc == 4;
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // the lines before a crash must reach the test log
    wchar_t log_path[MAX_PATH];
    const DWORD log_length = GetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path, MAX_PATH);
    if (!log_length || log_length >= MAX_PATH) {
        std::puts("FAIL: X2W_TRAP_LOG is not set");
        return 2;
    }
    DeleteFileW(log_path);  // before the stand-in writes its first line

    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    const auto adapter = pick_adapter(factory.Get(), std::strcmp(argv[1], "warp") == 0);
    if (!adapter) return 77;
    DXGI_ADAPTER_DESC1 adapter_desc{};
    if (SUCCEEDED(adapter->GetDesc1(&adapter_desc))) std::printf("adapter=%ls\n", adapter_desc.Description);
    ComPtr<IUnknown> guest, heap;
    std::array<unsigned char, 64> parameters{};
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                     reinterpret_cast<void**>(guest.GetAddressOf()))))
        return 3;
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 backing, bytes;
    } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3}, 0, 0};
    if (FAILED(invoke(guest.Get(), 14, &heap_desc, &__uuidof(ID3D12DescriptorHeap), heap.GetAddressOf()))) return 3;
    const UINT64 first = invoke<UINT64>(heap.Get(), 9);
    const UINT stride = invoke<UINT>(guest.Get(), 15, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT64 earlier = first, filler = first + stride, target = first + 2 * UINT64{stride};

    D3D12_HEAP_PROPERTIES default_heap{};
    default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = texture.Height = 16;
    texture.DepthOrArraySize = texture.MipLevels = 1;
    texture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texture.SampleDesc.Count = 1;
    ComPtr<IUnknown> textures[3];  // replaced view, failed view, later view
    const IID iid = __uuidof(ID3D12Resource);
    for (auto& t : textures)
        if (FAILED(invoke(guest.Get(), 27, &default_heap, D3D12_HEAP_FLAG_NONE, &texture, D3D12_RESOURCE_STATE_COMMON,
                          static_cast<const D3D12_CLEAR_VALUE*>(nullptr), &iid,
                          reinterpret_cast<void**>(t.GetAddressOf()))))
            return 3;

    // The stand-in identifies a view by every byte of its description, padding included, so each description is zeroed
    // with memset before its fields are set: brace initialization need not clear the padding.
    D3D12_SHADER_RESOURCE_VIEW_DESC srv;
    std::memset(&srv, 0, sizeof(srv));
    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    const auto texture_view = [&](int index, UINT64 handle) {
        invoke<void>(guest.Get(), 18, static_cast<void*>(textures[index].Get()), &srv, handle);
    };
    const auto constant_view = [&](UINT index, UINT64 handle) {
        D3D12_CONSTANT_BUFFER_VIEW_DESC desc;
        std::memset(&desc, 0, sizeof(desc));
        desc.BufferLocation = 0x100000ull + 256ull * index;
        desc.SizeInBytes = 256;
        invoke<void>(guest.Get(), 17, &desc, handle);
    };

    // Two views published before any failure.
    constant_view(0, earlier);
    const std::uint32_t earlier_id = token(earlier);
    texture_view(0, target);
    const std::uint32_t replaced_id = token(target);
    if (!earlier_id || !replaced_id || probe(earlier).shadow != earlier_id || probe(target).shadow != replaced_id) {
        std::puts("FAIL: the views published before the failure are not tracked");
        return 3;
    }
    const auto first_texture = probe(target, textures[0].Get());
    if (first_texture.view_refs != 1 || !first_texture.retire_when_idle) {
        std::puts("FAIL: the first texture is not in the retire class with one tracked view");
        return 3;
    }

    // A vector allocates only when it is full: distinct views until the next one has to grow it.
    UINT filled = 1;
    const auto fill = [&] {
        if (site == 2) return true;
        for (; filled < fill_limit; ++filled) {
            const auto p = probe(target);
            if (site == 0 ? p.records == p.records_capacity : p.materialized == p.materialized_capacity) return true;
            constant_view(filled, filler);
        }
        std::puts("FAIL: the vector never filled up");
        return false;
    };
    if (!fill()) return 3;

    const auto before = probe(target);
    std::printf("site=%s records=%llu capacity=%llu: arming %ld failures\n", sites[site], before.records,
                before.records_capacity, control ? 0 : failing_calls);
    if (!control) x2w_d3d12_fail_view_allocations(site, failing_calls);
    for (LONG call = 0; call < failing_calls; ++call) {
        texture_view(1, target);
        const auto p = probe(target, textures[1].Get());
        expect(p.records == before.records && p.materialized == before.materialized && p.ids == before.ids,
               "a failed publication changed a view container");
        expect(!p.shadow, "the destination slot still names a view after a failed publication");
        expect(!token(target), "title memory still holds a view token after a failed publication");
        expect(!p.view_refs && !p.retire_when_idle, "the resource of the failed view can still be released early");
    }
    x2w_d3d12_fail_view_allocations(site, 0);
    const auto replaced = probe(target, textures[0].Get());
    expect(!replaced.view_refs && replaced.retire_when_idle, "the replaced view kept its reference");

    // Views published before the failure keep their ids and add no record.
    const unsigned long long held = probe(target).records;
    constant_view(0, earlier);
    expect(token(earlier) == earlier_id && probe(earlier).shadow == earlier_id, "an earlier view changed its id");
    texture_view(0, target);
    expect(token(target) == replaced_id && probe(target).shadow == replaced_id, "the replaced view changed its id");
    expect(probe(target).records == held, "an earlier view added a record");

    // The view that failed, and then a view never seen before, are published normally.
    for (int index = 1; index < 3; ++index) {
        const auto start = probe(target);
        texture_view(index, target);
        const std::uint32_t id = token(target);
        const auto p = probe(target, textures[index].Get());
        expect(id && p.shadow == id, "a later view is not tracked");
        // In "control" the second texture view was already published by the unarmed calls.
        expect(p.records == start.records + (control && index == 1 ? 0 : 1) && p.materialized == p.records &&
                   p.ids == p.records,
               "a later view did not add one record to each container");
        expect(p.view_refs == 1, "a later view holds no reference");
        texture_view(index, target);
        expect(token(target) == id && probe(target).records == p.records, "a later view did not keep its id");
    }

    // Slot 19 names two resources, the view's and its counter. A failed UAV publication over a tracked UAV drops both
    // references of the earlier one and adds none. UAV-capable textures and buffers are never in the retire class, so
    // only the reference counts are observable here.
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = 4096;
    buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buffer.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<IUnknown> buffers[4];  // earlier resource and counter, then the failed view's resource and counter
    for (auto& b : buffers)
        if (FAILED(invoke(guest.Get(), 27, &default_heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COMMON,
                          static_cast<const D3D12_CLEAR_VALUE*>(nullptr), &iid,
                          reinterpret_cast<void**>(b.GetAddressOf()))))
            return 3;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav;
    std::memset(&uav, 0, sizeof(uav));
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.NumElements = 16;
    uav.Buffer.StructureByteStride = 16;
    const auto counted_view = [&](int index) {
        invoke<void>(guest.Get(), 19, static_cast<void*>(buffers[index].Get()),
                     static_cast<void*>(buffers[index + 1].Get()), &uav, target);
    };
    const auto refs = [&](int index) { return probe(target, buffers[index].Get()).view_refs; };
    counted_view(0);
    const std::uint32_t counted_id = token(target);
    if (!counted_id || probe(target).shadow != counted_id || refs(0) != 1 || refs(1) != 1) {
        std::puts("FAIL: the slot 19 view and its counter are not tracked");
        return 3;
    }
    if (!fill()) return 3;
    const auto counted = probe(target);
    if (!control) x2w_d3d12_fail_view_allocations(site, 1);
    counted_view(2);
    x2w_d3d12_fail_view_allocations(site, 0);
    const auto dropped = probe(target);
    expect(dropped.records == counted.records && dropped.materialized == counted.materialized && dropped.ids == counted.ids,
           "a failed slot 19 publication changed a view container");
    expect(!dropped.shadow && !token(target), "the destination still names a view after a failed slot 19 publication");
    expect(!refs(0), "the replaced slot 19 view kept its resource reference");
    expect(!refs(1), "the replaced slot 19 view kept its counter reference");
    expect(control || (!refs(2) && !refs(3)), "the failed slot 19 view holds a reference");
    counted_view(2);
    expect(token(target) && probe(target).shadow == token(target) && refs(2) == 1 && refs(3) == 1,
           "a later slot 19 view and its counter are not tracked");
    counted_view(0);
    expect(token(target) == counted_id && refs(0) == 1 && refs(1) == 1 && !refs(2) && !refs(3),
           "the earlier slot 19 view did not take its slot back");

    const std::string line = std::string("x2w-d3d12 note_view: out of memory in view_") + sites[site];
    expect(read_log().find(line) != std::string::npos, "the log does not name the failed allocation");

    if (failures) return 5;
    std::printf("view allocation failure checks passed (%s)\n", sites[site]);
    return 0;
}
