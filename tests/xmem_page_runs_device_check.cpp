// Placed buffers over physical pages at the boundary between two views (src/xmem_mapped_pages.cpp).
// Adjacent pages of one mapping call share a view, so a buffer can lie inside one view or across
// two: two mapping calls next to each other, the boundary between two backing sections inside one
// call, or pages that each have a view of their own. The device wrapper's placed-resource entry
// (slot 133) must accept all of them alike and leave the memory as it was. With `join`, X2W_XMEM_JOIN_CALLS=1 as
// well: a mapping call may then join the view that ends at its address, and the mapping order below keeps every
// boundary in place all the same. Without it the switch is cleared.
// usage: xmem_page_runs_device_check warp|hardware [join]
#include "xmem_test_device.h"

#include <cstring>
#include <vector>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) void* WINAPI XMemVirtualAlloc(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
extern "C" __declspec(dllimport) BOOL WINAPI XMemAllocatePhysicalPages(ULONG_PTR, PULONG_PTR, PULONG_PTR);
extern "C" __declspec(dllimport) void* WINAPI XMemMapPhysicalPages(void*, ULONG_PTR, const ULONG_PTR*);

namespace {
constexpr SIZE_T page = 65536;
int failures;

HRESULT create_wrapper(IUnknown* adapter, REFIID iid, void** out)
{
    unsigned char parameters[64]{};
    return D3D12XboxCreateDevice(adapter, parameters, iid, out);
}

using Place133 = HRESULT (*)(void*, void*, const D3D12_RESOURCE_DESC*, UINT, const void*, const GUID*, void**);

void* allocation_base(const void* address)
{
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(address, &info, sizeof(info)) == sizeof(info) && info.State == MEM_COMMIT ? info.AllocationBase : nullptr;
}

std::vector<ULONG_PTR> pages(ULONG_PTR count)
{
    std::vector<ULONG_PTR> tokens(count);
    ULONG_PTR got = count;
    if (!XMemAllocatePhysicalPages(0x20400000, &got, tokens.data()) || got != count) tokens.clear();
    return tokens;
}

// A two-page buffer that starts one page before `boundary`. views_expected is how many views it
// has to touch (1: the boundary is inside a view; 2: it is between two).
void place(const char* name, ID3D12Device* wrapper, unsigned char* boundary, int views_expected)
{
    unsigned char* const address = boundary - page;
    const int views = allocation_base(address) == allocation_base(boundary) ? 1 : 2;
    if (!allocation_base(address) || !allocation_base(boundary) || views != views_expected) {
        std::printf("FAIL %s: the buffer touches %d view(s), expected %d\n", name, views, views_expected);
        ++failures;
        return;
    }
    for (SIZE_T i = 0; i < 2 * page; i += 4096) address[i] = static_cast<unsigned char>(0x40 + i / 4096);
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 2 * page;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    void* placed = nullptr;
    const GUID iid = __uuidof(ID3D12Resource);
    const auto slot = reinterpret_cast<Place133>((*reinterpret_cast<void***>(wrapper))[133]);
    const HRESULT hr = slot(wrapper, address, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, &iid, &placed);
    const HRESULT removed = wrapper->GetDeviceRemovedReason();
    bool intact = true;
    for (SIZE_T i = 0; i < 2 * page; i += 4096) intact &= address[i] == static_cast<unsigned char>(0x40 + i / 4096);
    std::printf("%s: %d view(s), slot 133 -> 0x%08lx, object %s, memory %s\n", name, views, static_cast<unsigned long>(hr),
                placed ? "created" : "missing", intact ? "unchanged" : "changed");
    if (FAILED(hr) || !placed || FAILED(removed) || !intact) ++failures;
    if (placed) static_cast<IUnknown*>(placed)->Release();
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2 && !(argc == 3 && std::strcmp(argv[2], "join") == 0)) return 1;
    const bool warp = std::strcmp(argv[1], "warp") == 0;
    const bool join = argc == 3;
    bool unmatched = false;
    SetEnvironmentVariableW(L"X2W_D3D12_PERMISSIVE", L"1");
    SetEnvironmentVariableW(L"X2W_XMEM_SHARED_VIEWS", L"1");  // the mode under test; read by the first mapping call
    SetEnvironmentVariableW(L"X2W_XMEM_JOIN_CALLS", join ? L"1" : nullptr);
    auto wrapper = create_device(warp, unmatched, create_wrapper);
    if (!wrapper) return warp || unmatched ? 1 : 77;

    auto* const whole = static_cast<unsigned char*>(XMemVirtualAlloc(nullptr, SIZE_T{16} << 30, 0x20402000, 0x21, PAGE_READWRITE));
    if (!whole) return 1;
    unsigned char* const first = whole + page;

    // Two mapping calls next to each other: two views that meet at first + 8 pages. The upper one is mapped first: with
    // X2W_XMEM_JOIN_CALLS=1 a call joins only a view that ends at its address (merge_with_previous), never one that
    // starts after it.
    const auto a = pages(8), b = pages(8);
    if (a.empty() || b.empty() || XMemMapPhysicalPages(first + 8 * page, 8, b.data()) != first + 8 * page ||
        XMemMapPhysicalPages(first, 8, a.data()) != first) {
        std::printf("check_failed: allocate or map the two 8-page calls (error %lu)\n", GetLastError());
        return 1;
    }
    place("inside one view", wrapper.Get(), first + 4 * page, 1);
    place("across two mapping calls", wrapper.Get(), first + 8 * page, 2);

    // One call that runs past the end of a 256 MiB backing section: the view ends where the section does.
    unsigned char* const second = first + 64 * page;
    const auto many = pages(4200);
    if (many.empty() || XMemMapPhysicalPages(second, many.size(), many.data()) != second) return 1;
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(second, &info, sizeof(info)) != sizeof(info) || info.RegionSize >= many.size() * page) {
        std::printf("FAIL section boundary: the 4200-page mapping is one view\n");
        return 1;
    }
    place("across two backing sections", wrapper.Get(), second + info.RegionSize, 2);

    // Pages mapped one call at a time get a view each. They are mapped from high to low addresses: with
    // X2W_XMEM_JOIN_CALLS=1 no call then has an adjacent view below it to join, regardless of the order in which the
    // allocator assigned the backing slots.
    unsigned char* const third = second + 8192 * page;
    const auto single = pages(8);
    if (single.empty()) {
        std::printf("check_failed: allocate 8 single pages (error %lu)\n", GetLastError());
        return 1;
    }
    for (int i = 7; i >= 0; --i)
        if (XMemMapPhysicalPages(third + i * page, 1, &single[7 - i]) != third + i * page) {
            std::printf("check_failed: single-page call %d (error %lu)\n", i, GetLastError());
            return 1;
        }
    place("across two single-page views", wrapper.Get(), third + 4 * page, 2);

    if (failures) {
        std::printf("xmem_page_runs_device: %d FAILED\n", failures);
        return 1;
    }
    std::printf("xmem_page_runs_device: PASS (a buffer inside one view and across three kinds of view boundary)\n");
    return 0;
}
