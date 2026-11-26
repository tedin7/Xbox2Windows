#pragma once
#include <d3d12.h>
#include <dxgi.h>
#include <wrl/client.h>

// Keeps the producer, argument resource and signature alive until the consumer's queue fence.
struct Slot67Indirect {
    Microsoft::WRL::ComPtr<ID3D12RootSignature> root;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> writer;
    Microsoft::WRL::ComPtr<ID3D12Resource> placed, counts, upload, readback;
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> signature;
    Microsoft::WRL::ComPtr<ID3D12Resource> batch_output;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> output_allocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> output_list;

    void prepare(IDXGIAdapter* adapter, void* guest_device, void* guest_list, void* backing, UINT groups, void* count_backing = nullptr);
    void execute(void* guest_list, UINT commands, bool count_buffer, bool outside) const;
    void execute_batch(void* guest_list, bool counted, void* count_alias, bool first_only, bool short_range) const;
    void copy_batch_output(IDXGIAdapter* adapter, void* guest_device, void* guest_queue, void* output, UINT bytes);
    void read_batch_output(void* destination, UINT bytes) const;
    bool check_batch(UINT count) const;
    bool check_record(UINT groups) const;
};
