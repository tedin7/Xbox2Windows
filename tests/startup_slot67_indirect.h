#pragma once
#include <d3d12.h>
#include <dxgi.h>
#include <wrl/client.h>

// Keeps the producer, argument resource and signature alive until the consumer's queue fence.
struct Slot67Indirect {
    Microsoft::WRL::ComPtr<ID3D12RootSignature> root;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> writer;
    Microsoft::WRL::ComPtr<ID3D12Resource> placed, upload, readback, before_consumer, native_args;
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> signature;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> native_queue;
    Microsoft::WRL::ComPtr<ID3D12Resource> native_output;

    void prepare(IDXGIAdapter* adapter, void* guest_device, void* guest_list, void* backing, UINT groups, bool write_zero = false,
                 void* guest_queue = nullptr, void* guest_output = nullptr);
    void execute(void* guest_list, UINT commands, bool count_buffer, bool outside) const;
    void capture_before_consumer(IDXGIAdapter* adapter, void* guest_device, void* guest_list);
    bool check_record(UINT groups, bool write_zero = false, bool consumer_boundary = false) const;
    // The guest list has closed and its fence has completed; the output rests in COMMON.
    void draw_pixels(IDXGIAdapter* adapter, void* guest_device, void* guest_queue, void* guest_output, bool stale) const;
};
