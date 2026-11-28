#include <d3d12.h>

// A source-built module whose basename matches the console BVH builder. The gate resolves the
// caller of device slot 62; a call through this export must be tagged as a builder pipeline.
extern "C" __declspec(dllexport) HRESULT WINAPI CreateBuilderPipeline(
    IUnknown* device, const D3D12_COMPUTE_PIPELINE_STATE_DESC* desc, REFIID iid, void** out)
{
    if (!device || !desc || !out) return E_POINTER;
    using Slot62 = HRESULT(WINAPI*)(IUnknown*, const D3D12_COMPUTE_PIPELINE_STATE_DESC*, UINT,
                                    const void*, REFIID, void**);
    const auto fn = reinterpret_cast<Slot62>((*reinterpret_cast<void***>(device))[62]);
    return fn(device, desc, 0, nullptr, iid, out);
}
