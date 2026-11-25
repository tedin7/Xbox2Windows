// Source-built writer and lifetime probes for the cached mesh fixture.
struct MeshLeaseWriter {
    static constexpr UINT64 page = 65536, word_offset = 0x40;
    static constexpr std::array<UINT, 3> old_groups{1, 1, 1}, new_groups{2, 1, 1};
    static constexpr UINT leading_guard = 0xa5a5a5a5, trailing_guard = 0x5a5a5a5a;
    using Reserve = void* (WINAPI*)(void*, SIZE_T, DWORD, ULONGLONG, DWORD);
    using Allocate = BOOL (WINAPI*)(ULONG_PTR, ULONG_PTR*, ULONG_PTR*);
    using Map = BOOL (WINAPI*)(void* const*, ULONG_PTR, const ULONG_PTR*);
    using Free = BOOL (WINAPI*)(ULONG_PTR, ULONG_PTR*);
    using Pin = BOOL (WINAPI*)(UINT, ULONG_PTR, SIZE_T, X2WXMemSectionLease*, SIZE_T);
    using Unpin = BOOL (WINAPI*)(UINT, ULONG64);

    ID3D12Device* device;
    ID3D12Device* guest;
    HMODULE xmem;
    const UINT64 record_bytes;
    const UINT64 record_offset;
    const UINT expected_group_x;
    Allocate alloc;
    Map map;
    Free free_pages;
    Pin pin;
    Unpin unpin;
    unsigned char* cpu{};
    UINT64 mirror{};
    ULONG_PTR original[2]{};
    std::pair<ULONG64, ULONG64> published_page;
    ComPtr<IUnknown> placed, recording, queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12Resource> writer_source;
    ComPtr<ID3D12Fence> gate, completion;
    bool submitted{};

    template<class T> T function(const char* name)
    {
        const auto fn = reinterpret_cast<T>(GetProcAddress(xmem, name));
        check(fn ? S_OK : E_FAIL, name);
        return fn;
    }

    MeshLeaseWriter(ID3D12Device* d, ID3D12Device* g, const wchar_t* dll,
                    ID3D12RootSignature* writer_root = nullptr, ID3D12PipelineState* writer = nullptr,
                    MeshLeaseWriter* mirror_owner = nullptr, ID3D12PipelineState* append = nullptr)
        : device(d), guest(g), xmem(LoadLibraryW(dll)), record_bytes(writer ? sizeof(old_groups) : sizeof(red)),
          record_offset(mirror_owner ? 0x80 : word_offset), expected_group_x(mirror_owner && !append ? 0 : new_groups[0]),
          alloc(function<Allocate>("XMemAllocatePhysicalPages")),
          map(function<Map>("XMemMapPhysicalPagesScatter")), free_pages(function<Free>("XMemFreePhysicalPages")),
          pin(function<Pin>("x2w_xmem_lease_section")), unpin(function<Unpin>("x2w_xmem_release_section"))
    {
        if (mirror_owner) {
            cpu = mirror_owner->cpu + page;
            mirror = mirror_owner->mirror + page;
        } else {
            check(SetEnvironmentVariableA("X2W_XMEM_PERMISSIVE", "1") &&
                  SetEnvironmentVariableA("X2W_XMEM_GPU_MIRROR", "1") ? S_OK : E_FAIL, "enable XMem mirror");
            const auto reserve = function<Reserve>("XMemVirtualAlloc");
            constexpr UINT64 window = UINT64{16} << 30;
            cpu = static_cast<unsigned char*>(reserve(nullptr, window, 0x20402000, 0x21, 4));
            mirror = reinterpret_cast<UINT64>(reserve(nullptr, window, 0x20402000, 3, 0x801));
            check(cpu && mirror ? S_OK : E_FAIL, "reserve mirrored XMem");
            ULONG_PTR count = 2;
            void* const addresses[]{cpu, cpu + page};
            check(alloc(0x280, &count, original) && count == 2 && map(addresses, 2, original) ? S_OK : E_FAIL,
                  "map two XMem pages");
        }
        if (writer) {
            std::memcpy(cpu + record_offset, old_groups.data(), sizeof(old_groups));
            if (!mirror_owner) {
                std::memcpy(cpu + page + record_offset, old_groups.data(), sizeof(old_groups));
                std::memcpy(cpu + record_offset + 16, old_groups.data(), sizeof(old_groups));
            }
            std::memcpy(cpu + record_offset - 4, &leading_guard, sizeof(leading_guard));
            std::memcpy(cpu + record_offset + record_bytes, &trailing_guard, sizeof(trailing_guard));
        } else {
            std::memcpy(cpu + record_offset, &green, sizeof(green));
            std::memcpy(cpu + page + record_offset, &green, sizeof(green));
        }
        published_page = inspect(address());

        const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, writer && !mirror_owner ? 2 * page : page,
            1, 1, 1, DXGI_FORMAT_UNKNOWN,
            {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
        using Place = HRESULT (WINAPI*)(IUnknown*, void*, const D3D12_RESOURCE_DESC*, UINT, const void*, REFIID, void**);
        check(slot<Place>(guest, 133)(guest, reinterpret_cast<void*>(mirror), &desc, D3D12_RESOURCE_STATE_COMMON,
                                     nullptr, __uuidof(ID3D12Resource), reinterpret_cast<void**>(placed.GetAddressOf())),
              "create copied XMem placement through slot 133");
        const auto type = writer ? D3D12_COMMAND_LIST_TYPE_COMPUTE : D3D12_COMMAND_LIST_TYPE_COPY;
        check(guest->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator)), "create writer allocator");
        recording = begin_guest_list(guest, allocator.Get(), type);
        if (writer) {
            writer_source = make_buffer(device, sizeof(UINT) + record_bytes, D3D12_HEAP_TYPE_DEFAULT,
                                        D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            const auto list = native_object<ID3D12GraphicsCommandList>(device, guest, recording.Get());
            transition(list.Get(), writer_source.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            list->SetPipelineState(writer);
            list->SetComputeRootSignature(writer_root);
            list->SetComputeRootUnorderedAccessView(0, writer_source->GetGPUVirtualAddress());
            list->Dispatch(1, 1, 1);
            if (append) {
                D3D12_RESOURCE_BARRIER between_dispatches{};
                between_dispatches.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                between_dispatches.UAV.pResource = writer_source.Get();
                list->ResourceBarrier(1, &between_dispatches);
                list->SetPipelineState(append);
                list->Dispatch(1, 1, 1);
            }
            transition(list.Get(), writer_source.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        } else {
            writer_source = make_buffer(device, sizeof(red), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
            const D3D12_RANGE empty{};
            void* mapped = nullptr;
            check(writer_source->Map(0, &empty, &mapped), "map writer source");
            std::memcpy(mapped, &red, sizeof(red));
            writer_source->Unmap(0, nullptr);
        }
        using Copy = void (WINAPI*)(IUnknown*, IUnknown*, UINT64, IUnknown*, UINT64, UINT64);
        const UINT64 source_offset = writer && !mirror_owner ? argument_offset : 0;
        slot<Copy>(recording.Get(), 15)(recording.Get(), placed.Get(), record_offset, writer_source.Get(),
                                        source_offset, record_bytes);
        check(slot<HRESULT (WINAPI*)(IUnknown*)>(recording.Get(), 9)(recording.Get()), "close slot 15 writer");
        queue = create_queue(type);
        gate = create_fence();
        completion = create_fence();
    }

    ~MeshLeaseWriter()
    {
        if (submitted) {
            gate->Signal(1);
            mesh_fence_reached(completion.Get(), 5000);
        }
    }

    UINT64 address() const { return mirror + record_offset; }

    ComPtr<IUnknown> create_queue(D3D12_COMMAND_LIST_TYPE type)
    {
        const D3D12_COMMAND_QUEUE_DESC desc{type};
        ComPtr<IUnknown> result;
        check(guest->CreateCommandQueue(&desc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(result.GetAddressOf())),
              "create wrapped lease queue");
        return result;
    }

    ComPtr<ID3D12Fence> create_fence()
    {
        ComPtr<ID3D12Fence> result;
        check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&result)), "create lease fixture fence");
        return result;
    }

    std::pair<ULONG64, ULONG64> inspect(UINT64 guest_address)
    {
        X2WXMemSectionLease lease{};
        check(pin(x2w_xmem_section_lease_version, guest_address, record_bytes, &lease, sizeof(lease)) ? S_OK : E_FAIL,
              "inspect XMem page");
        CloseHandle(lease.section);
        check(unpin(x2w_xmem_section_lease_version, lease.id) ? S_OK : E_FAIL, "release inspection lease");
        return {lease.section_offset, lease.first_byte};
    }

    void start()
    {
        auto native = native_object<ID3D12CommandQueue>(device, guest, queue.Get());
        check(native->Wait(gate.Get(), 1), "hold writer behind gate");
        submit_wrapped(queue.Get(), recording.Get());
        if (const HRESULT status = native->Signal(completion.Get(), 1); FAILED(status)) {
            gate->Signal(1);
            check(status, "signal writer completion");
        }
        submitted = true;
    }

    void finish()
    {
        check(gate->Signal(1), "open writer gate");
        check(mesh_fence_reached(completion.Get(), 5000) ? S_OK : E_FAIL, "wait for slot 15 writer");
        check(device->GetDeviceRemovedReason(), "slot 15 GPU completion");
        if (record_bytes == sizeof(new_groups)) {
            UINT observed[5]{};
            const UINT expected[]{leading_guard, expected_group_x, new_groups[1], new_groups[2], trailing_guard};
            std::memcpy(observed, cpu + record_offset - sizeof(UINT), sizeof(observed));
            for (std::size_t i = 0; i < std::size(expected); ++i) {
                if (observed[i] != expected[i])
                    std::fprintf(stderr, "XMem mesh argument/guard %zu: got %08x, expected %08x\n", i, observed[i], expected[i]);
                check(observed[i] == expected[i] ? S_OK : E_FAIL, "check the compute-written XMem record and guards");
            }
            std::printf("XMem DispatchMesh record and guards: {%u,1,1} with both guards intact\n", expected_group_x);
            return;
        }
        UINT actual{};
        std::memcpy(&actual, cpu + record_offset, sizeof(actual));
        if (actual != red)
            std::fprintf(stderr, "leased slot 15 writer: 0x%08x != 0x%08x\n", actual, red);
        check(actual == red ? S_OK : E_FAIL, "leased XMem writer changed the CPU view");
    }

    void open_after_wait_probe(ID3D12Fence* done)
    {
        const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        check(event ? S_OK : E_FAIL, "create bounded wait probe event");
        const auto registered = done->SetEventOnCompletion(1, event);
        const DWORD result = SUCCEEDED(registered) ? WaitForSingleObject(event, 50) : WAIT_FAILED;
        // Open the writer even if the ordering check fails.
        check(gate->Signal(1), "open producer after ordering probe");
        if (SUCCEEDED(registered)) WaitForSingleObject(event, 5000);
        CloseHandle(event);
        check(result == WAIT_TIMEOUT ? S_OK : E_FAIL, "consumer waited for the held writer");
    }

    void free_original()
    {
        recording.Reset();
        placed.Reset();
        void* const addresses[]{cpu, cpu + page};
        const ULONG_PTR empty[]{0, 0};
        check(map(addresses, 2, empty) && free_pages(2, original) ? S_OK : E_FAIL, "release the original XMem grant");
    }

    std::pair<ULONG64, ULONG64> allocate_probe(UINT64 index)
    {
        ULONG_PTR token{}, count = 1;
        void* const address = cpu + index * page;
        check(alloc(0x280, &count, &token) && count == 1 && map(&address, 1, &token) ? S_OK : E_FAIL, "map probe page");
        const auto identity = inspect(mirror + index * page + word_offset);
        const ULONG_PTR empty{};
        check(map(&address, 1, &empty) && free_pages(1, &token) ? S_OK : E_FAIL, "free probe page");
        return identity;
    }

    void check_retirement(const char* label)
    {
        // The GPU fence is already complete. Allow the runtime's retirement
        // callback to run without issuing more calls to the wrapped queue.
        const auto deadline = GetTickCount64() + 3000;
        do {
            if (allocate_probe(2) == published_page) return;
            Sleep(1);
        } while (GetTickCount64() < deadline);
        check(E_FAIL, label);
    }
};

bool supports_lease_heap(ID3D12Device* device)
{
    D3D12_FEATURE_DATA_EXISTING_HEAPS support{};
    return SUCCEEDED(check_feature(device, D3D12_FEATURE_EXISTING_HEAPS, &support, sizeof(support))) && support.Supported;
}

void check_lease_debug(ID3D12InfoQueue* info)
{
    if (!info) return;
    for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
        SIZE_T bytes{};
        check(info->GetMessage(i, nullptr, &bytes), "size D3D12 debug message");
        std::vector<unsigned char> storage(bytes);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        check(info->GetMessage(i, message, &bytes), "read D3D12 debug message");
        if (message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING) {
            std::fprintf(stderr, "%s\n", message->pDescription);
            check(E_FAIL, "D3D12 debug layer reported a warning or error");
        }
    }
    std::puts("D3D12 warnings/errors: 0");
}

bool check_mesh_indirect_xmem(ID3D12Device* device, ID3D12Device* guest, const wchar_t* xmem,
                              ID3D12InfoQueue* info, const std::wstring& log_path,
                              ID3D12RootSignature* writer_root, ID3D12PipelineState* writer,
                              ID3D12RootSignature* root, IUnknown* pipeline, ID3D12CommandSignature* signature)
{
    MeshLeaseWriter source(device, guest, xmem, writer_root, writer);
    const auto queue = source.create_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
    const auto draw = [&](UINT64 offset, UINT left, UINT right) {
        MeshLeaseDraw args{0, queue.Get(), left};
        args.argument = source.placed.Get();
        args.argument_offset = offset;
        args.expected_right = right;
        return args;
    };
    const auto verify = [&](const MeshLeaseDraw& args, const char* label) {
        return run(device, guest, nullptr, nullptr, root, pipeline, signature, DrawKind::mesh,
                   false, Refusal::none, label, nullptr, false, nullptr, false, false, 0, false, false, &args);
    };
    constexpr auto offset = MeshLeaseWriter::word_offset;
    if (!verify(draw(offset, black, black), "mesh arguments without a submitted writer refused")) return false;
    source.start();
    if (!verify(draw(offset + MeshLeaseWriter::page, black, black), "mesh arguments on a different page refused") ||
        !verify(draw(offset + 16, black, black), "mesh arguments at a different offset refused")) return false;

    const bool skip_wait = GetEnvironmentVariableA("X2W_TEST_MESH_XMEM_SKIP_WAIT", nullptr, 0) != 0;
    auto consume = draw(offset, red, skip_wait ? black : green);
    if (!skip_wait) consume.queued = [&](ID3D12Fence* done) { source.open_after_wait_probe(done); };
    if (!verify(consume, skip_wait ? "mesh arguments without a wait execute one stale group"
                                   : "mesh arguments with a wait execute two fresh groups")) return false;
    source.finish();
    if (skip_wait) {
        source.free_original();
    } else {
        const auto gate = source.create_fence();
        struct ReleaseGate {
            ID3D12Fence* gate;
            ~ReleaseGate() { gate->Signal(1); }
        } release{gate.Get()};
        const auto native = native_object<ID3D12CommandQueue>(device, guest, queue.Get());
        check(native->Wait(gate.Get(), 1), "hold the indirect consumer before execution");
        consume.drop_recording = true;
        consume.queued = [&](ID3D12Fence*) {
            source.free_original();
            check(source.allocate_probe(2) != source.published_page ? S_OK : E_FAIL,
                  "in-flight indirect arguments pin the original slot");
            check(gate->Signal(1), "release the indirect consumer");
        };
        if (!verify(consume, "submitted indirect mesh keeps arguments live after unmapping")) return false;
        source.check_retirement("indirect consumer completion returns the original slot to XMem");
    }

    std::ifstream log{std::filesystem::path(log_path)};
    const std::string lines{std::istreambuf_iterator<char>(log), {}};
    log.close();
    DeleteFileW(log_path.c_str());
    constexpr char no_writer[] = "mesh XMem read has no submitted writer for the required size";
    constexpr char mismatch[] = "mesh XMem read does not match the writer's leased page and byte range";
    const auto wrong_page = lines.find(mismatch);
    check(lines.find(no_writer) != std::string::npos && wrong_page != std::string::npos &&
          lines.find(mismatch, wrong_page + sizeof(mismatch) - 1) != std::string::npos ? S_OK : E_FAIL,
          "check missing writer, wrong-page and wrong-range refusals");
    check_lease_debug(info);
    return true;
}

bool check_mesh_dual_xmem(ID3D12Device* device, ID3D12Device* guest, const wchar_t* xmem,
                          ID3D12InfoQueue* info, const std::wstring& log_path,
                          ID3D12RootSignature* writer_root, ID3D12PipelineState* initialize, ID3D12PipelineState* append,
                          ID3D12RootSignature* root, IUnknown* pipeline, ID3D12CommandSignature* signature)
{
    const bool init_only = !append;
    MeshLeaseWriter colour(device, guest, xmem);
    MeshLeaseWriter arguments(device, guest, xmem, writer_root, initialize, &colour, append);
    check(colour.published_page.first != arguments.published_page.first ? S_OK : E_FAIL,
          "combined mesh writers use different physical pages");
    const auto queue = colour.create_queue(D3D12_COMMAND_LIST_TYPE_DIRECT);
    const auto draw = [&](UINT left, UINT right) {
        MeshLeaseDraw result{colour.address(), queue.Get(), left};
        result.argument = arguments.placed.Get();
        result.argument_offset = arguments.record_offset;
        result.expected_right = right;
        return result;
    };
    const auto verify = [&](const MeshLeaseDraw& args, const char* label) {
        return run(device, guest, nullptr, nullptr, root, pipeline, signature, DrawKind::mesh,
                   false, Refusal::none, label, nullptr, false, nullptr, false, false, 1, false, false, &args);
    };
    const auto release_pages = [&] {
        arguments.recording.Reset();
        arguments.placed.Reset();
        colour.free_original();
    };
    const auto refused = draw(black, black);
    if (!verify(refused, "combined cached mesh without submitted writers refused")) return false;
    if (GetEnvironmentVariableA("X2W_MESH86_FIXTURE_REJECT_MARKER", nullptr, 0)) {
        colour.start();
        arguments.start();
        colour.finish();
        arguments.finish();
        const bool result = verify(refused, "combined cached mesh rejects invalid marker with both writers");
        release_pages();
        check_lease_debug(info);
        return result;
    }
    if (!GetEnvironmentVariableA("X2W_MESH86_FIXTURE_READY", nullptr, 0)) {
        release_pages();
        check_lease_debug(info);
        return true;
    }

    const bool skip_srv = GetEnvironmentVariableA("X2W_TEST_MESH_XMEM_SKIP_SRV_WAIT", nullptr, 0) != 0;
    const bool skip_arguments = GetEnvironmentVariableA("X2W_TEST_MESH_XMEM_SKIP_ARGUMENT_WAIT", nullptr, 0) != 0;
    check(static_cast<int>(skip_srv) + static_cast<int>(skip_arguments) + static_cast<int>(init_only) <= 1 ? S_OK : E_FAIL,
          "select one combined mesh control");
    const bool control = skip_srv || skip_arguments;
    auto consume = init_only ? refused : draw(skip_srv ? green : red, skip_srv ? green : skip_arguments ? black : red);
    const char* label = init_only ? "combined mesh initialized arguments: zero groups" :
                        skip_srv ? "combined mesh stale SRV: two green halves" :
                        skip_arguments ? "combined mesh stale arguments: left red, right clear" :
                                         "combined mesh fresh inputs: two red halves";
    colour.start();
    arguments.start();
    bool ordered = false;
    consume.queued = [&](ID3D12Fence* done) {
        const bool blocked = !mesh_fence_reached(done, 50);
        // With one wait omitted, release the producer which is still required.
        // The other producer remains held until the entire draw completes.
        auto& first = skip_srv ? arguments : colour;
        auto& second = skip_srv ? colour : arguments;
        const HRESULT opened_first = first.gate->Signal(1);
        const bool first_finished = mesh_fence_reached(first.completion.Get(), 5000);
        const bool finished_after_first = mesh_fence_reached(done, control ? 5000 : 50);
        const HRESULT opened_second = second.gate->Signal(1);
        ordered = blocked && SUCCEEDED(opened_first) && first_finished &&
                  SUCCEEDED(opened_second) && finished_after_first == control;
    };
    if (!verify(consume, label)) return false;
    check(ordered ? S_OK : E_FAIL, "combined mesh consumer followed the expected producer gates");
    colour.finish();
    arguments.finish();

    if (!control && !init_only) {
        auto remapped = refused;
        ULONG_PTR replacement{}, count = 1;
        check(colour.alloc(0x280, &count, &replacement) && count == 1 ? S_OK : E_FAIL,
              "allocate replacement argument page");
        void* const replaced_address = arguments.cpu;
        remapped.before_submit = [&] {
            check(colour.map(&replaced_address, 1, &replacement) ? S_OK : E_FAIL,
                  "remap only the argument page before submission");
            std::memcpy(arguments.cpu + arguments.record_offset, MeshLeaseWriter::old_groups.data(),
                        sizeof(MeshLeaseWriter::old_groups));
            check(arguments.inspect(arguments.address()) != arguments.published_page &&
                  colour.inspect(colour.address()) == colour.published_page ? S_OK : E_FAIL,
                  "argument page identity changed while SRV page stayed mapped");
        };
        const bool unchanged = verify(remapped, "combined mesh submit with remapped argument page leaves target clear");
        check(colour.map(&replaced_address, 1, &colour.original[1]) && colour.free_pages(1, &replacement) ? S_OK : E_FAIL,
              "restore the original argument page");
        if (!unchanged || !verify(draw(red, red), "combined mesh reads both original ranges after restoring mapping")) return false;
    }

    release_pages();
    std::ifstream log{std::filesystem::path(log_path)};
    check(log ? S_OK : E_FAIL, "open combined mesh handoff log");
    const std::string lines{std::istreambuf_iterator<char>(log), {}};
    log.close();
    DeleteFileW(log_path.c_str());
    check(lines.find("mesh XMem read has no submitted writer for the required size") != std::string::npos &&
          (control || init_only || lines.find("mesh XMem submit refused: writer or live page identity changed") != std::string::npos)
          ? S_OK : E_FAIL, "check combined mesh missing-writer and remapped-argument refusals");
    check_lease_debug(info);
    return true;
}
