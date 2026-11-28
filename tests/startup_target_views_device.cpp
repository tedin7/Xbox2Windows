// Synthetic target views through the shim, checked by the D3D12 debug layer.
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
namespace {
using Word = std::uint64_t;
template<class T> Word arg(T* p) { return reinterpret_cast<Word>(p); }
Word invoke(void* p, unsigned slot, Word a=0, Word b=0, Word c=0, Word d=0, Word e=0, Word f=0, Word g=0) {
    using Call = Word (WINAPI*)(void*, Word, Word, Word, Word, Word, Word, Word);
    return reinterpret_cast<Call>((*static_cast<void***>(p))[slot])(p,a,b,c,d,e,f,g);
}
std::atomic<unsigned> errors{0};
void WINAPI message(D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID, LPCSTR text, void*) {
    if (severity <= D3D12_MESSAGE_SEVERITY_ERROR && ++errors <= 12) std::printf("debug layer: %.300s\n", text);
}
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    ComPtr<IDXGIFactory6> factory; ComPtr<IDXGIAdapter> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    if (!std::strcmp(argv[1], "warp")) {
        if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) return 3;
    } else if (!std::strcmp(argv[1], "hardware")) {
        for (UINT i=0;; ++i) {
            ComPtr<IDXGIAdapter1> candidate;
            HRESULT hr=factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
            if (hr==DXGI_ERROR_NOT_FOUND) return 77;
            if (FAILED(hr)) return 3;
            DXGI_ADAPTER_DESC1 desc{}; if (FAILED(candidate->GetDesc1(&desc))) return 3;
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) { adapter=candidate; break; }
        }
    } else return 2;
    std::array<unsigned char,64> parameters{}; void* device=nullptr;
    ComPtr<ID3D12Device> desktop; ComPtr<ID3D12InfoQueue1> queue;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(),parameters.data(),__uuidof(ID3D12Device),&device)) ||
        FAILED(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&desktop)))) return 4;
    DWORD cookie=0;
    if (FAILED(desktop.As(&queue)) || FAILED(queue->RegisterMessageCallback(message,D3D12_MESSAGE_CALLBACK_FLAG_NONE,nullptr,&cookie))) { std::puts("no debug layer"); return 5; }
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC tex{D3D12_RESOURCE_DIMENSION_TEXTURE2D,0,32,32,2,4,DXGI_FORMAT_R8G8B8A8_UNORM,{1,0},D3D12_TEXTURE_LAYOUT_UNKNOWN,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    D3D12_RESOURCE_DESC array_tex=tex; tex.DepthOrArraySize=1;
    D3D12_RESOURCE_DESC depth=tex; depth.DepthOrArraySize=1; depth.MipLevels=1; depth.Format=DXGI_FORMAT_D32_FLOAT; depth.Flags=D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER,0,65536,1,1,1,DXGI_FORMAT_UNKNOWN,{1,0},D3D12_TEXTURE_LAYOUT_ROW_MAJOR,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    void* texture=nullptr; void* array_resource=nullptr; void* z=nullptr; void* buf=nullptr;
    auto make = [&](D3D12_RESOURCE_DESC* desc, void** output) {
        return SUCCEEDED(static_cast<HRESULT>(invoke(device,27,arg(&heap),0,arg(desc),D3D12_RESOURCE_STATE_COMMON,0,arg(&__uuidof(ID3D12Resource)),arg(output))));
    };
    if (!make(&tex,&texture) || !make(&array_tex,&array_resource) || !make(&depth,&z) || !make(&buffer,&buf)) return 6;
    struct HeapRequest { D3D12_DESCRIPTOR_HEAP_DESC desc; UINT64 backing, bytes; };
    auto make_heap = [&](D3D12_DESCRIPTOR_HEAP_TYPE type, void** output) {
        HeapRequest req{{type,4,D3D12_DESCRIPTOR_HEAP_FLAG_NONE,0},0,0};
        return SUCCEEDED(static_cast<HRESULT>(invoke(device,14,arg(&req),arg(&__uuidof(ID3D12DescriptorHeap)),arg(output))));
    };
    void *rh=nullptr,*dh=nullptr,*uh=nullptr;
    if (!make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV,&rh) || !make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_DSV,&dh) || !make_heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,&uh)) return 7;
    const Word rt=invoke(rh,9), ds=invoke(dh,9), ua=invoke(uh,9);
    D3D12_RENDER_TARGET_VIEW_DESC r{}; r.Format=tex.Format; r.ViewDimension=D3D12_RTV_DIMENSION_TEXTURE2D; r.Texture2D.MipSlice=9;
    invoke(device,20,arg(texture),arg(&r),rt);
    r.ViewDimension=D3D12_RTV_DIMENSION_TEXTURE2DARRAY; r.Texture2DArray={0,0,5,0}; invoke(device,20,arg(array_resource),arg(&r),rt);
    r.ViewDimension=D3D12_RTV_DIMENSION_TEXTURE2D; r.Texture2D={0,7}; invoke(device,20,arg(texture),arg(&r),rt);
    D3D12_DEPTH_STENCIL_VIEW_DESC d{}; d.Format=depth.Format; d.ViewDimension=D3D12_DSV_DIMENSION_TEXTURE2D; d.Flags=D3D12_DSV_FLAG_READ_ONLY_STENCIL; invoke(device,21,arg(z),arg(&d),ds);
    D3D12_UNORDERED_ACCESS_VIEW_DESC u{}; u.ViewDimension=D3D12_UAV_DIMENSION_BUFFER; u.Buffer.NumElements=1; u.Buffer.StructureByteStride=4; u.Buffer.CounterOffsetInBytes=100; invoke(device,19,arg(buf),0,arg(&u),ua);
    u={}; u.Format=tex.Format; u.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D; u.Texture2D.MipSlice=4; invoke(device,19,arg(texture),0,arg(&u),ua);
    invoke(device,20,0,0,rt); invoke(device,19,0,0,0,ua); d.ViewDimension=D3D12_DSV_DIMENSION_UNKNOWN; invoke(device,21,0,arg(&d),ds);
    r.Texture2D={0,0}; invoke(device,20,arg(texture),arg(&r),rt);
    u.Texture2D={0,0}; invoke(device,19,arg(texture),0,arg(&u),ua);
    d.ViewDimension=D3D12_DSV_DIMENSION_TEXTURE2D; d.Flags=D3D12_DSV_FLAG_NONE; invoke(device,21,arg(z),arg(&d),ds);
    const HRESULT removed=desktop->GetDeviceRemovedReason();
    std::printf("errors=%u removed=0x%08lx\n",errors.load(),static_cast<unsigned long>(removed));
    queue->UnregisterMessageCallback(cookie);
    return errors || removed!=S_OK ? 8 : 0;
}
