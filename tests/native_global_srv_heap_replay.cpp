// Native compute fixture whose shader picks a global (descriptor heap) SRV slot from a GPU input.
//
// root    OUT.root                                  writes the root signature the request container carries.
// record  SDK CACHE REQ.cdx ADAPTER SELECTOR EXPECT [--no-guard]
//         Native recording path: reads the GPU selector back before the dispatch, captures {select, view types}
//         into REQ.views, refuses a slot of another view type before the shader runs, otherwise records the
//         dispatch with x2w_startup::NativeRecording and NativeDispatchStates and compares the fenced output.
// replay  SDK CACHE REQ.cdx COMPILED.cso VIEWS ADAPTER EXPECT REFERENCE_SLOT [EXPECTED_WORDS]
//         Offline request replay: decodes VIEWS, refuses before the dispatch or runs the compiled.cso shader on a
//         heap rebuilt from the captured slot types and compares with the reference slot data.
// EXPECT is "ok" or "refused". Exit 0: outcome as expected. 1: output mismatch for an expected-ok case. 3: the
// outcome differs from the expectation. 4: setup error. 77: capability skip.
#include "desktop_gpu.h"
#include "native_compute_pipeline.h"
#include "startup_stubs/native_dispatch_states.h"
#include "startup_stubs/native_global_srv.h"
#include "startup_stubs/native_recording.h"
#include <dxgi1_4.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
using x2w_startup::GlobalSrvType;
using x2w_startup::GlobalSrvView;
namespace fs = std::filesystem;

namespace {
constexpr unsigned slot_count = 4, slot_words = 64, selector_page_slot = 4, output_page_slot = 5;
constexpr unsigned output_words = 72;
constexpr std::uint32_t sentinel = 0x69696969u;

// Backing data of a raw buffer slot, deterministic from the slot index.
std::uint32_t slot_word(unsigned slot, unsigned index) { return 0xA0000000u + slot * 0x10000000u + index; }

struct Rig {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    UINT64 fence_value = 0;
    x2w_startup::NativeRecording recording;
    x2w_startup::NativeDescriptorTable page;
    ComPtr<ID3D12Resource> selector, output;
    std::vector<ComPtr<ID3D12Resource>> slots{slot_count}; // keeps the backing resources of the heap alive
    std::vector<ComPtr<ID3D12Resource>> upload_keep;

    ~Rig() {
        if (event) CloseHandle(event);
    }

    // Returns 77 when the adapter cannot run the fixture; 0 when the rig is ready.
    int open(const std::wstring &adapter) {
        ComPtr<IDXGIFactory4> factory;
        ComPtr<IDXGIAdapter> chosen;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 77;
        if (adapter == L"warp") {
            if (FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&chosen)))) return 77;
        } else {
            for (UINT i = 0; i < 8; ++i) {
                ComPtr<IDXGIAdapter1> candidate;
                if (factory->EnumAdapters1(i, &candidate) == DXGI_ERROR_NOT_FOUND) break;
                DXGI_ADAPTER_DESC1 desc{};
                candidate->GetDesc1(&desc);
                if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                    chosen = candidate;
                    std::printf("adapter=%ls\n", desc.Description);
                    break;
                }
            }
            if (!chosen) { std::puts("no hardware adapter"); return 77; }
        }
        if (FAILED(D3D12CreateDevice(chosen.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
            std::puts("no device");
            return 77;
        }
        D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_6};
        D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
        if (FAILED(check_feature(device.Get(), D3D12_FEATURE_SHADER_MODEL, &model, sizeof model)) ||
            model.HighestShaderModel < D3D_SHADER_MODEL_6_6 ||
            FAILED(check_feature(device.Get(), D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof options)) ||
            options.ResourceBindingTier < D3D12_RESOURCE_BINDING_TIER_3) {
            std::printf("adapter lacks shader model 6.6 or resource binding tier 3 (tier %d)\n",
                        int(options.ResourceBindingTier));
            return 77;
        }
        // The native module declares WaveSize=32 (the guest wave), so an adapter without 32-lane waves cannot run it.
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 waves{};
        if (FAILED(check_feature(device.Get(), D3D12_FEATURE_D3D12_OPTIONS1, &waves, sizeof waves)) || !waves.WaveOps ||
            waves.WaveLaneCountMin > 32 || waves.WaveLaneCountMax < 32) {
            std::printf("adapter has no 32-lane waves (%u..%u)\n", waves.WaveLaneCountMin, waves.WaveLaneCountMax);
            return 77;
        }
        D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
        check(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "queue");
        check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "allocator");
        check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)),
              "list");
        check(list->Close(), "close");
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "fence");
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) throw std::runtime_error("event");
        // The descriptor page of one native recording: slots 0..3 are selectable, 4 is t0, 5 is u0.
        page = recording.allocate(device.Get(), 6);
        return 0;
    }

    D3D12_CPU_DESCRIPTOR_HANDLE cpu(unsigned index) const {
        auto handle = page.cpu;
        handle.ptr += SIZE_T(index) * page.increment;
        return handle;
    }
    D3D12_GPU_DESCRIPTOR_HANDLE first_gpu() const {
        auto handle = page.gpu;
        handle.ptr += UINT64(selector_page_slot) * page.increment;
        return handle;
    }

    void wait() {
        check(queue->Signal(fence.Get(), ++fence_value), "signal");
        check(fence->SetEventOnCompletion(fence_value, event), "fence event");
        if (WaitForSingleObject(event, 30000) != WAIT_OBJECT_0) throw std::runtime_error("fence wait timed out");
    }
    // Records `body` on the one list and waits for its fence.
    template<class Body> void run(Body body) {
        check(allocator->Reset(), "allocator reset");
        check(list->Reset(allocator.Get(), nullptr), "list reset");
        body(list.Get());
        check(list->Close(), "list close");
        ID3D12CommandList *lists[]{list.Get()};
        queue->ExecuteCommandLists(1, lists);
        wait();
    }
    // A command list with no commands: what a refused request submits instead of the dispatch.
    void run_empty() {
        run([](ID3D12GraphicsCommandList *) {});
    }
    void fill(ID3D12Resource *destination, const std::vector<std::uint32_t> &words) {
        const UINT64 bytes = words.size() * 4;
        auto upload = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void *mapped = nullptr;
        D3D12_RANGE none{};
        check(upload->Map(0, &none, &mapped), "upload map");
        std::memcpy(mapped, words.data(), bytes);
        upload->Unmap(0, nullptr);
        run([&](ID3D12GraphicsCommandList *l) { l->CopyBufferRegion(destination, 0, upload.Get(), 0, bytes); });
    }
    std::vector<std::uint32_t> read(ID3D12Resource *source, unsigned words) {
        auto readback = make_buffer(device.Get(), UINT64(words) * 4, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        run([&](ID3D12GraphicsCommandList *l) { l->CopyBufferRegion(readback.Get(), 0, source, 0, UINT64(words) * 4); });
        std::vector<std::uint32_t> out(words);
        void *mapped = nullptr;
        D3D12_RANGE range{0, UINT64(words) * 4};
        check(readback->Map(0, &range, &mapped), "readback map");
        std::memcpy(out.data(), mapped, out.size() * 4);
        D3D12_RANGE written{};
        readback->Unmap(0, &written);
        return out;
    }

    // t0 (the selector, a raw buffer) and u0 (72 words of output) in their page slots.
    void create_io() {
        selector = make_buffer(device.Get(), 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
        output = make_buffer(device.Get(), output_words * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_R32_TYPELESS;
        srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Buffer.NumElements = 1;
        srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        device->CreateShaderResourceView(selector.Get(), &srv, cpu(selector_page_slot));
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_R32_TYPELESS;
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.NumElements = output_words;
        uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device->CreateUnorderedAccessView(output.Get(), nullptr, &uav, cpu(output_page_slot));
    }
    void fill_sentinel() { fill(output.Get(), std::vector<std::uint32_t>(output_words, sentinel)); }
    bool sentinels_intact() {
        for (auto word : read(output.Get(), output_words))
            if (word != sentinel) return false;
        return true;
    }

    // Creates the backing resource and the SRV of one selectable slot. Returns the captured view: the class is
    // classified from the exact description passed to CreateShaderResourceView, the size from the resource.
    GlobalSrvView add_slot(unsigned slot, GlobalSrvType type, std::uint64_t bytes) {
        D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (type == GlobalSrvType::texture) {
            D3D12_RESOURCE_DESC texture{};
            texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            texture.Width = texture.Height = 8;
            texture.DepthOrArraySize = texture.MipLevels = 1;
            texture.Format = DXGI_FORMAT_R32_UINT;
            texture.SampleDesc.Count = 1;
            slots[slot] = resource(device.Get(), texture, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
            desc.Format = DXGI_FORMAT_R32_UINT;
            desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            desc.Texture2D.MipLevels = 1;
        } else if (type == GlobalSrvType::raw_buffer || type == GlobalSrvType::typed_buffer) {
            if (!bytes || bytes % 4 || bytes > (1u << 20))
                throw std::runtime_error("slot " + std::to_string(slot) + " has an unusable buffer size");
            slots[slot] = make_buffer(device.Get(), bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
            std::vector<std::uint32_t> data(bytes / 4);
            for (unsigned i = 0; i < data.size(); ++i) data[i] = slot_word(slot, i);
            fill(slots[slot].Get(), data);
            desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            desc.Buffer.NumElements = static_cast<UINT>(bytes / 4);
            if (type == GlobalSrvType::raw_buffer) {
                desc.Format = DXGI_FORMAT_R32_TYPELESS;
                desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            } else {
                desc.Format = DXGI_FORMAT_R32_UINT;
            }
        } else {
            throw std::runtime_error(std::string("view type ") + x2w_startup::global_srv_type_name(type) +
                                     " cannot be rebuilt by this fixture");
        }
        device->CreateShaderResourceView(slots[slot].Get(), &desc, cpu(slot));
        GlobalSrvView view;
        if (!x2w_startup::classify_global_srv_view(slot, &desc, sizeof desc, view))
            throw std::runtime_error("slot view could not be classified");
        if (view.type != type) throw std::runtime_error("slot view classified as another type");
        view.bytes = type == GlobalSrvType::texture ? 0 : slots[slot]->GetDesc().Width;
        return view;
    }

    unsigned read_selector(unsigned value) {
        fill(selector.Get(), {value});
        return read(selector.Get(), 1)[0];
    }

    // Barriers from the views of the dispatch (NativeDispatchStates), the dispatch, the barriers back, the fence.
    // `state` is the pipeline-state object to run (null: the one bind() set).
    void dispatch(const x2w::NativeComputePipeline &pipeline, ID3D12PipelineState *state, ID3D12Resource *selected) {
        struct View {
            ComPtr<ID3D12Resource> resource;
            bool write = false;
            x2w_startup::NativeTextureWindow texture{};
            struct { bool upload = false; } buffer;
        };
        std::vector<View> views;
        views.push_back({selector, false, {}, {}});
        views.push_back({output, true, {}, {}});
        views.push_back({selected, false, {}, {}});
        x2w_startup::NativeDispatchStates states(views);
        std::vector<D3D12_RESOURCE_BARRIER> before, after;
        states.whole_states([&](ID3D12Resource *resource, D3D12_RESOURCE_STATES target) {
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON, target};
            before.push_back(b);
            std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
            after.push_back(b);
        });
        run([&](ID3D12GraphicsCommandList *l) {
            l->ResourceBarrier(UINT(before.size()), before.data());
            pipeline.bind(l, page.heap, first_gpu(), page.increment);
            if (state) l->SetPipelineState(state);
            l->Dispatch(1, 1, 1);
            l->ResourceBarrier(UINT(after.size()), after.data());
        });
    }
};

std::vector<std::uint32_t> reference_output(unsigned slot) {
    std::vector<std::uint32_t> words(output_words, sentinel);
    for (unsigned i = 0; i < slot_words; ++i) words[i] = slot_word(slot, i);
    return words;
}

// 0: the 72 words equal the reference; 1 prints the first mismatch.
int compare_output(Rig &rig, unsigned reference_slot, const fs::path &reference_path = {}) {
    auto expected = reference_output(reference_slot);
    if (!reference_path.empty()) {
        std::ifstream input(reference_path, std::ios::binary);
        if (!input.read(reinterpret_cast<char *>(expected.data()), output_words * sizeof(expected[0])) ||
            input.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("expected output file must contain exactly 72 dwords");
    }
    const auto actual = rig.read(rig.output.Get(), output_words);
    for (unsigned i = 0; i < output_words; ++i)
        if (actual[i] != expected[i]) {
            std::printf("mismatch word %u: got 0x%08x expected 0x%08x (reference slot %u)\n", i, actual[i], expected[i],
                        reference_slot);
            return 1;
        }
    if (reference_path.empty())
        std::printf("output matches slot %u data (%u words) and the %u sentinel words are intact\n", reference_slot,
                    slot_words, output_words - slot_words);
    else
        std::printf("output matches CPU reference (%u words)\n", output_words);
    return 0;
}

// The refused path of both modes: no dispatch is recorded, and the output must be untouched.
int refused_path(Rig &rig, bool expect_ok) {
    rig.run_empty();
    const bool intact = rig.sentinels_intact();
    std::printf("no dispatch submitted; u0 sentinels %s\n", intact ? "intact" : "CHANGED");
    return (!expect_ok && intact) ? 0 : 3;
}

int write_root(const fs::path &destination) {
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameter.DescriptorTable = {2, ranges};
    D3D12_ROOT_SIGNATURE_DESC desc{1, &parameter, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> blob, error;
    if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error)))
        throw std::runtime_error("root signature serialization failed");
    std::ofstream file(destination, std::ios::binary);
    file.write(static_cast<const char *>(blob->GetBufferPointer()), static_cast<std::streamsize>(blob->GetBufferSize()));
    file.close();
    if (!file) throw std::runtime_error("could not write the root signature");
    return 0;
}

// The pipeline must read t0 and u0 through the tables and the selected slot through the heap.
void check_pipeline_shape(const x2w::NativeComputePipeline &pipeline) {
    if (!pipeline.heap_indexed || pipeline.srv_count != 1 || pipeline.uav_count != 1 || pipeline.constant_words)
        throw std::runtime_error("native pipeline has an unexpected binding layout");
}

// The pipeline contract a request's metadata allows; empty error when it is translated.
x2w::HeapViewContract heap_contract(const x2w_startup::GlobalSrvRequest &request, std::string &error) {
    const auto contract = x2w_startup::global_srv_contract(request, error);
    return contract == x2w_startup::GlobalSrvContract::typed_dword ? x2w::HeapViewContract::typed_dword
                                                                   : x2w::HeapViewContract::raw_buffer;
}

int record(int argc, wchar_t **argv) {
    if (argc < 8 || argc > 11) return 2;
    const std::wstring expect = argv[7];
    bool guard = true, need_given = false;
    GlobalSrvType needed = GlobalSrvType::raw_buffer;
    for (int a = 8; a < argc; ++a) {
        const std::wstring_view flag = argv[a];
        if (flag == L"--no-guard" && guard) {
            guard = false;
        } else if (flag == L"--need" && a + 1 < argc && !need_given) {
            need_given = true;
            std::string name;  // a view type name: ASCII, anything else names no type
            for (const wchar_t c : std::wstring_view(argv[++a])) name += c < 0x80 ? static_cast<char>(c) : '?';
            needed = x2w_startup::global_srv_type_from_name(name);
            if (needed == GlobalSrvType::none) return 2;
        } else {
            return 2;
        }
    }
    if (expect != L"ok" && expect != L"refused") return 2;
    const unsigned selector = static_cast<unsigned>(std::stoul(argv[6]));
    const fs::path cdx = argv[4];
    Rig rig;
    if (int skip = rig.open(argv[5])) return skip;
    try {
        x2w::ShaderDxilCompiler compiler(argv[2]);
        rig.create_io();
        x2w_startup::GlobalSrvRequest request;
        request.needed = needed;
        request.views.push_back(rig.add_slot(0, GlobalSrvType::raw_buffer, slot_words * 4));
        request.views.push_back(rig.add_slot(1, GlobalSrvType::raw_buffer, slot_words * 4));
        request.views.push_back(rig.add_slot(2, GlobalSrvType::texture, 0));
        request.views.push_back(rig.add_slot(3, GlobalSrvType::typed_buffer, slot_words * 4));
        rig.fill_sentinel();
        // The selector is GPU data: it is read back (one CPU/GPU sync) before the dispatch is recorded.
        request.selected = rig.read_selector(selector);
        request.has_selected = true;
        auto views_path = cdx;
        views_path.replace_extension(L".views");
        std::ofstream(views_path, std::ios::binary) << x2w_startup::encode_global_srv_request(request);
        std::printf("captured select=%u need=%s views=%s\n", request.selected,
                    x2w_startup::global_srv_type_name(request.needed), views_path.string().c_str());
        // The contract depends on the captured views, not on the selection: the shader is translated for every record,
        // a refused selection included.
        std::string error;
        const auto contract = heap_contract(request, error);
        if (!error.empty()) {
            std::printf("refused: %s\n", error.c_str());
            return refused_path(rig, expect == L"ok");
        }
        auto pipeline = x2w::create_native_compute_pipeline(rig.device.Get(), compiler, cdx, 0, argv[3], true, contract);
        check_pipeline_shape(pipeline);
        const auto diagnostic = x2w_startup::refuse_global_srv_selection(request.views, request.selected, request.needed);
        if (!diagnostic.empty() && guard) {
            std::printf("refused select=%u: %s\n", request.selected, diagnostic.c_str());
            return refused_path(rig, expect == L"ok");
        }
        if (request.selected >= slot_count || request.views[request.selected].type != request.needed)
            throw std::runtime_error("--no-guard only dispatches a slot of the needed view type");
        rig.dispatch(pipeline, nullptr, rig.slots[request.selected].Get());
        if (expect == L"refused") {
            // Discrimination control: without the guard the shader ran, so the output must have changed.
            const bool intact = rig.sentinels_intact();
            std::printf("dispatch ran (%s); %s\n", guard ? "no refusal" : "guard disabled",
                        intact ? "sentinels intact" : "sentinels changed");
            return 3;
        }
        return compare_output(rig, request.selected);
    } catch (const std::exception &error) {
        std::printf("setup error: %s\n", error.what());
        return 4;
    }
}

int replay(int argc, wchar_t **argv) {
    if (argc != 10 && argc != 11) return 2;
    const std::wstring expect = argv[8];
    if (expect != L"ok" && expect != L"refused") return 2;
    const fs::path cdx = argv[4], cso = argv[5], views_path = argv[6];
    const unsigned reference_slot = static_cast<unsigned>(std::stoul(argv[9]));
    Rig rig;
    if (int skip = rig.open(argv[7])) return skip;
    try {
        rig.create_io();
        rig.fill_sentinel();
        std::string text, error;
        x2w_startup::GlobalSrvRequest request;
        std::ifstream file(views_path, std::ios::binary);
        if (!file) {
            error = "global SRV view metadata file missing: " + views_path.string();
        } else {
            text.assign(std::istreambuf_iterator<char>(file), {});
            x2w_startup::decode_global_srv_request(text, request, error);
        }
        if (error.empty())
            error = x2w_startup::refuse_global_srv_selection(request.views, request.selected, request.needed);
        auto contract = x2w::HeapViewContract::raw_buffer;
        if (error.empty()) contract = heap_contract(request, error);
        if (!error.empty()) {
            std::printf("refused: %s\n", error.c_str());
            return refused_path(rig, expect == L"ok");
        }
        x2w::ShaderDxilCompiler compiler(argv[2]);
        auto pipeline = x2w::create_native_compute_pipeline(rig.device.Get(), compiler, cdx, 0, argv[3], true, contract);
        check_pipeline_shape(pipeline);
        std::ifstream shader(cso, std::ios::binary);
        if (!shader) throw std::runtime_error("missing compiled shader " + cso.string());
        const std::string dxil((std::istreambuf_iterator<char>(shader)), {});
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = pipeline.root.Get();
        pso.CS = {dxil.data(), dxil.size()};
        ComPtr<ID3D12PipelineState> compiled;
        check(rig.device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&compiled)), "compiled shader pipeline");
        for (const auto &view : request.views) {
            if (view.slot >= slot_count) throw std::runtime_error("captured slot outside the fixture heap page");
            rig.add_slot(view.slot, view.type, view.bytes);
        }
        rig.read_selector(request.selected);
        rig.dispatch(pipeline, compiled.Get(), rig.slots[request.selected].Get());
        const int result = compare_output(rig, reference_slot, argc == 11 ? fs::path(argv[10]) : fs::path{});
        if (expect == L"refused") return 3; // ran when a refusal was expected
        return result;
    } catch (const std::exception &error) {
        std::printf("setup error: %s\n", error.what());
        return 4;
    }
}
} // namespace

int wmain(int argc, wchar_t **argv) {
    try {
        if (argc >= 2 && std::wstring_view(argv[1]) == L"root" && argc == 3) return write_root(argv[2]);
        if (argc >= 2 && std::wstring_view(argv[1]) == L"record") return record(argc, argv);
        if (argc >= 2 && std::wstring_view(argv[1]) == L"replay") return replay(argc, argv);
    } catch (const std::exception &error) {
        std::printf("setup error: %s\n", error.what());
        return 4;
    }
    std::fputs("usage: native_global_srv_heap_replay root OUT.root | record SDK CACHE REQ.cdx ADAPTER SELECTOR ok|refused "
               "[--no-guard] [--need raw_buffer|typed_buffer] | replay SDK CACHE REQ.cdx COMPILED.cso VIEWS ADAPTER ok|refused REFERENCE_SLOT [EXPECTED_WORDS]\n", stderr);
    return 2;
}
