// Checks src/startup_stubs/title_heap_memory.h.
//   (no arguments)   the rule, without a device
//   warp | hardware  through the startup device wrapper: every heap start the executable is given
//                    is pointer-aligned, and samplers can be created and copied through such starts
//   --mutate tags    the rule with the alignment test removed; must fail (negative control)
#include "xmem_test_device.h"
#include "../src/startup_stubs/title_heap_memory.h"
#include "../src/startup_stubs/logical_view_table.h"

#include <cstring>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
int failures;

void expect(const char* name, bool got, bool wanted)
{
    if (got != wanted) {
        std::printf("FAIL %s: %d, expected %d\n", name, got, wanted);
        ++failures;
    }
}

int rule(bool mutate)
{
    const auto needs = [mutate](D3D12_DESCRIPTOR_HEAP_TYPE type, std::uint64_t start) {
        return mutate ? type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV : x2w_startup::heap_needs_title_memory(type, start);
    };
    constexpr auto views = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, samplers = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                   targets = D3D12_DESCRIPTOR_HEAP_TYPE_RTV, depth = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    // The four starts measured through the translation layer, and aligned starts of each type.
    expect("view heap, tagged start", needs(views, 0x7c000014), true);
    expect("view heap, aligned start", needs(views, 0x7c000000), true);
    expect("sampler heap, tagged start", needs(samplers, 0x74adb007), true);
    expect("sampler heap, start with one tag bit", needs(samplers, 0x74adb004), true);
    expect("sampler heap, aligned start", needs(samplers, 0x74adb000), false);
    expect("sampler heap, 8-aligned start", needs(samplers, 0x74adb008), false);
    expect("render-target heap, aligned start", needs(targets, 0x74bdf260), false);
    expect("depth heap, aligned start", needs(depth, 0x74bff760), false);
    expect("render-target heap, tagged start", needs(targets, 0x74bdf261), true);
    std::uint64_t memory = 0;
    unsigned allocations = 0;
    const auto fail_allocation = [&](std::uint64_t) -> std::uint64_t {
        ++allocations;
        return 0;
    };
    expect("tagged sampler without backing rejects OOM",
           x2w_startup::ensure_title_heap_memory(samplers, 0x74adb007, 64, memory, fail_allocation), false);
    expect("failure leaves backing empty", memory == 0 && allocations == 1, true);
    expect("aligned sampler has no backing requirement",
           x2w_startup::ensure_title_heap_memory(samplers, 0x74adb000, 64, memory, fail_allocation), true);
    memory = 0x10000;
    expect("keeps caller-provided CPU memory",
           x2w_startup::ensure_title_heap_memory(samplers, 0x74adb007, 64, memory, fail_allocation), true);
    expect("no alternate allocation after validation", memory == 0x10000 && allocations == 1, true);
    // A successful allocation becomes the heap's memory, sized for the whole heap: 1024 view
    // descriptors of 32 bytes are 0x8000 bytes, 128 samplers of 32 bytes are 0x1000.
    std::uint64_t asked = 0;
    const auto allocate = [&](std::uint64_t size) -> std::uint64_t {
        ++allocations;
        asked = size;
        return 0x20000;
    };
    memory = 0;
    expect("view heap without backing gets the allocation",
           x2w_startup::ensure_title_heap_memory(views, 0x7c000000, 0x8000, memory, allocate), true);
    expect("view heap backing is the allocated address and size",
           memory == 0x20000 && asked == 0x8000 && allocations == 2, true);
    memory = 0;
    expect("tagged sampler without backing gets the allocation",
           x2w_startup::ensure_title_heap_memory(samplers, 0x74adb007, 0x1000, memory, allocate), true);
    expect("sampler backing is the allocated address and size",
           memory == 0x20000 && asked == 0x1000 && allocations == 3, true);
    constexpr std::uint64_t cpu = 0x20000, gpu = 0x40000;
    constexpr unsigned desktop_stride = 64, title_stride = x2w_startup::title_view_stride;
    expect("console view size selects 32 bytes",
           x2w_startup::view_heap_title_stride(views, 4, cpu, 4 * title_stride, desktop_stride) == title_stride, true);
    expect("desktop view size keeps desktop stride",
           x2w_startup::view_heap_title_stride(views, 4, cpu, 4 * desktop_stride, desktop_stride) == desktop_stride, true);
    expect("missing backing keeps desktop stride",
           x2w_startup::view_heap_title_stride(views, 4, 0, 4 * title_stride, desktop_stride) == desktop_stride, true);
    expect("other heap keeps desktop stride",
           x2w_startup::view_heap_title_stride(targets, 4, cpu, 4 * title_stride, desktop_stride) == desktop_stride, true);
    for (unsigned i = 0; i < 4; ++i) {
        expect("raw view CPU maps by index",
               x2w_startup::desktop_descriptor_offset(i * title_stride, title_stride, desktop_stride) == i * desktop_stride, true);
        expect("desktop-sized CPU keeps offset",
               x2w_startup::desktop_descriptor_offset(i * desktop_stride, desktop_stride, desktop_stride) == i * desktop_stride, true);
    }
    const auto raw_table = x2w_startup::resolve_view_table(gpu, cpu, 4, title_stride, gpu + 3 * title_stride);
    expect("raw view table at slot 3", raw_table.address == cpu + 96 && raw_table.bytes == 32, true);
    const auto old_table = x2w_startup::resolve_view_table(gpu, cpu, 4, desktop_stride, gpu + 3 * desktop_stride);
    expect("desktop-sized view table keeps offset", old_table.address == cpu + 192 && old_table.bytes == 64, true);
    if (failures) {
        std::printf("startup_title_heap: %d FAILED\n", failures);
        return 1;
    }
    std::printf("startup_title_heap: PASS (9 starts and backing allocation controls)\n");
    return 0;
}

HRESULT create_wrapper(IUnknown* adapter, REFIID iid, void** out)
{
    unsigned char parameters[64]{};
    return D3D12XboxCreateDevice(adapter, parameters, iid, out);
}

using Word = std::uint64_t;
Word call(void* object, int index, Word a = 0, Word b = 0, Word c = 0, Word d = 0)
{
    using Slot = Word (*)(void*, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d);
}

struct Heap {
    const char* name;
    D3D12_DESCRIPTOR_HEAP_TYPE type;
    D3D12_DESCRIPTOR_HEAP_FLAGS flags;
    UINT count;
    void* object = nullptr;
    Word start = 0, stride = 0;
};

int device(bool warp)
{
    bool unmatched = false;
    auto desktop = create_device(warp, unmatched);
    if (!desktop) return warp || unmatched ? 1 : 77;
    SetEnvironmentVariableW(L"X2W_D3D12_PERMISSIVE", L"1");
    auto wrapper = create_device(warp, unmatched, create_wrapper);
    if (!wrapper) return 1;
    void* const guest = wrapper.Get();

    // The four heaps a console executable creates without memory of its own (measured counts).
    Heap heaps[] = {{"shader-visible sampler heap", D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 128},
                    {"staging sampler heap", D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 128},
                    {"render-target heap", D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 4096},
                    {"depth heap", D3D12_DESCRIPTOR_HEAP_TYPE_DSV, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 2048}};
    bool ours = false, theirs = false;
    for (Heap& heap : heaps) {
        // What the adapter gives a desktop caller for the same heap, and what the rule makes of it.
        const D3D12_DESCRIPTOR_HEAP_DESC plain{heap.type, heap.count, heap.flags, 0};
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> reference;
        if (FAILED(desktop->CreateDescriptorHeap(&plain, IID_PPV_ARGS(&reference)))) return 1;
        const Word desktop_start = reference->GetCPUDescriptorHandleForHeapStart().ptr;
        const bool expected = x2w_startup::heap_needs_title_memory(heap.type, desktop_start);
        // The console description: the desktop one, then an address and a size the caller may supply (none here).
        const struct {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            Word memory, bytes;
        } described{plain, 0, 0};
        const GUID iid = __uuidof(ID3D12DescriptorHeap);
        if (FAILED(static_cast<HRESULT>(call(guest, 14, reinterpret_cast<Word>(&described), reinterpret_cast<Word>(&iid),
                                             reinterpret_cast<Word>(&heap.object)))) || !heap.object) {
            std::printf("FAIL %s: not created\n", heap.name);
            return 1;
        }
        heap.start = call(heap.object, 9);
        heap.stride = call(guest, 15, heap.type);
        // Memory of ours is the base of a committed private allocation; a desktop handle that is an address is not.
        MEMORY_BASIC_INFORMATION info{};
        const bool plain_memory = VirtualQuery(reinterpret_cast<void*>(heap.start), &info, sizeof(info)) == sizeof(info) &&
                                  info.State == MEM_COMMIT && info.Type == MEM_PRIVATE &&
                                  reinterpret_cast<Word>(info.AllocationBase) == heap.start;
        std::printf("%s: desktop start 0x%llx (low bits 0x%llx), wrapper start 0x%llx, stride %llu, %s\n", heap.name,
                    static_cast<unsigned long long>(desktop_start), static_cast<unsigned long long>(desktop_start & 7),
                    static_cast<unsigned long long>(heap.start), static_cast<unsigned long long>(heap.stride),
                    plain_memory ? "plain memory" : "desktop handle");
        if (!heap.start || (heap.start & (alignof(void*) - 1)) || !heap.stride) {
            std::printf("FAIL %s: the start given to the executable is not an aligned address\n", heap.name);
            return 1;
        }
        if (expected && !plain_memory) {
            std::printf("FAIL %s: the desktop start is not an address, and the heap has no memory of its own\n", heap.name);
            return 1;
        }
        (expected ? ours : theirs) = true;
    }
    // Samplers created in the staging heap and copied to the shader-visible one, through handles
    // computed from the starts as an address plus an index times the stride.
    D3D12_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    for (UINT i = 0; i < 3; ++i) {
        sampler.MaxAnisotropy = i;
        call(guest, 22, reinterpret_cast<Word>(&sampler), heaps[1].start + i * heaps[1].stride);
        call(guest, 24, 1, heaps[0].start + (100 + i) * heaps[0].stride, heaps[1].start + i * heaps[1].stride, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    }
    const HRESULT removed = wrapper->GetDeviceRemovedReason();
    if (FAILED(removed)) {
        std::printf("FAIL: device removed after the sampler copies (0x%08lx)\n", static_cast<unsigned long>(removed));
        return 1;
    }
    std::printf("startup_title_heap_device: PASS (4 heaps with aligned starts%s%s; 3 samplers created and copied)\n",
                ours ? ", plain memory where the desktop start is tagged" : "", theirs ? ", desktop handles where it is an address" : "");
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc == 1) return rule(false);
    if (argc == 3 && !std::strcmp(argv[1], "--mutate") && !std::strcmp(argv[2], "tags")) return rule(true);
    if (argc == 2 && (!std::strcmp(argv[1], "warp") || !std::strcmp(argv[1], "hardware"))) return device(!std::strcmp(argv[1], "warp"));
    return 2;
}
