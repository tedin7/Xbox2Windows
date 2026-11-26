#pragma once
#include <d3d12.h>
#include <dxgi.h>
#include <wrl/client.h>

// Keeps the producer, argument resource and signature alive until the consumer's queue fence.
struct Slot67Indirect {
    Microsoft::WRL::ComPtr<ID3D12RootSignature> root;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> writer;
    Microsoft::WRL::ComPtr<ID3D12Resource> placed, upload, readback;
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> signature;

    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> native_list;
    Microsoft::WRL::ComPtr<ID3D12Resource> output, output_upload, output_readback;
    UINT constant_count = 0;

    void prepare(IDXGIAdapter* adapter, void* guest_device, void* guest_list, void* backing, UINT groups,
                 void* title_root = nullptr, UINT constants = 0, void* guest_output = nullptr);
    void capture_output(bool dispatched) const;
    void read_output(UINT* words) const;
    void execute(void* guest_list, UINT commands, bool count_buffer, bool outside) const;
    bool check_record(UINT groups) const;
};
