// Pass thermometer, submission side (included by pass_stats.inl inside namespace pass_stats): the reduction list
// recorded after each title submission, and the per-present report.

const char* Tracker::ensure_pipelines(ID3D12Device* device)
{
    if (pipelines_.tried) return pipelines_.failure;
    pipelines_.tried = true;
    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER parameters[3]{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[0].Constants = {0, 0, 8};
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    parameters[1].Descriptor = {0, 0};
    parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[2].DescriptorTable = {1, &range};
    const D3D12_ROOT_SIGNATURE_DESC desc{3, parameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> serialized, error;
    if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &serialized, &error)) ||
        FAILED(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                                           IID_PPV_ARGS(&pipelines_.root))))
        return pipelines_.failure = "reduction root signature";
    static const char* const kinds[] = {"0", "1", "2", "3"};
    for (unsigned k = 0; k < 4; ++k) {
        const D3D_SHADER_MACRO macros[] = {{"KIND", kinds[k]}, {nullptr, nullptr}};
        ComPtr<ID3DBlob> code, messages;
        if (FAILED(D3DCompile(m::reduction_hlsl, sizeof(m::reduction_hlsl) - 1, "pass stats reduction", macros, nullptr,
                              "main", "cs_5_0", 0, 0, &code, &messages))) {
            log("x2w-pass stats: reduction %u does not compile: %s", k,
                messages ? static_cast<const char*>(messages->GetBufferPointer()) : "no message");
            return pipelines_.failure = "reduction shader";
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline{};
        pipeline.pRootSignature = pipelines_.root.Get();
        pipeline.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        if (FAILED(device->CreateComputePipelineState(&pipeline, IID_PPV_ARGS(&pipelines_.kinds[k]))))
            return pipelines_.failure = "reduction pipeline";
    }
    return nullptr;
}

void Tracker::execute(ID3D12CommandQueue* queue, const std::vector<ID3D12CommandList*>& lists)
{
    std::lock_guard<std::mutex> guard(lock_);
    Submission s;
    for (ID3D12CommandList* list : lists) {
        ++counts_[executed];
        const auto found = recordings_.find(static_cast<ID3D12GraphicsCommandList*>(list));
        if (found == recordings_.end()) continue;
        ++counts_[tracked];
        for (const Record& record : found->second.passes) s.reports.push_back({record});
    }
    if (s.reports.empty()) return;
    s.present = InterlockedCompareExchange(&frame_number, 0, 0) + 1;
    build(queue, s);
    submissions_.push_back(std::move(s));
}

void Tracker::build(ID3D12CommandQueue* queue, Submission& s)
{
    for (std::size_t r = 0; r < s.reports.size(); ++r) {
        const auto& writes = s.reports[r].record.writes;
        for (std::size_t w = 0; w < writes.size(); ++w) {
            if (!writes[w].snapshot) continue;
            const unsigned groups = m::groups_for(writes[w].texels);
            s.jobs.push_back({r, w, groups, s.words});
            s.words += u64{groups} * m::group_words;
        }
    }
    if (s.jobs.empty()) return;
    const auto type = queue->GetDesc().Type;
    ID3D12Device* device = device_of(the_device);
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE) s.failure = "queue type";
    if (!s.failure) s.failure = ensure_pipelines(device);
    if (!s.failure && !record_reductions(s, device, type)) s.failure = "reduction list";
    if (s.failure) return;
    auto& fence = fences_[queue];
    if (!fence.first && FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.first)))) {
        s.failure = "fence";
        return;
    }
    ID3D12CommandList* lists[] = {s.list.Get()};
    queue->ExecuteCommandLists(1, lists);
    s.fence = fence.first;
    s.value = ++fence.second;
    if (FAILED(queue->Signal(s.fence.Get(), s.value))) s.failure = "fence signal";
}

bool Tracker::record_reductions(Submission& s, ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type)
{
    const u64 bytes = s.words * 4;
    const D3D12_HEAP_PROPERTIES gpu{D3D12_HEAP_TYPE_DEFAULT}, readback{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                               D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, static_cast<UINT>(s.jobs.size()),
                                               D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    if (FAILED(device->CreateCommandAllocator(type, IID_PPV_ARGS(&s.allocator))) ||
        FAILED(device->CreateCommandList(0, type, s.allocator.Get(), nullptr, IID_PPV_ARGS(&s.list))) ||
        FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&s.heap))) ||
        FAILED(device->CreateCommittedResource(&gpu, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&s.result))))
        return false;
    buffer.Flags = D3D12_RESOURCE_FLAG_NONE;
    const D3D12_HEAP_PROPERTIES upload{D3D12_HEAP_TYPE_UPLOAD};
    void* zeros = nullptr;
    if (FAILED(device->CreateCommittedResource(&upload, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_GENERIC_READ,
                                               nullptr, IID_PPV_ARGS(&s.zeros))) ||
        FAILED(s.zeros->Map(0, nullptr, &zeros)) || !zeros)
        return false;
    std::memset(zeros, 0, static_cast<std::size_t>(bytes));  // the groups fold into zeroed records with atomics
    s.zeros->Unmap(0, nullptr);
    if (FAILED(device->CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&s.readback))))
        return false;
    ID3D12GraphicsCommandList* list = s.list.Get();
    ID3D12DescriptorHeap* heaps[] = {s.heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(pipelines_.root.Get());
    list->SetComputeRootUnorderedAccessView(1, s.result->GetGPUVirtualAddress());
    barrier(list, s.result.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(s.result.Get(), 0, s.zeros.Get(), 0, bytes);
    barrier(list, s.result.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const UINT increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    for (std::size_t j = 0; j < s.jobs.size(); ++j) {
        const Job& job = s.jobs[j];
        const Write& w = s.reports[job.report].record.writes[job.write];
        D3D12_SHADER_RESOURCE_VIEW_DESC view{};
        view.Format = w.info.view;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (w.info.sample == m::Sample::words) {
            view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            view.Buffer = {0, static_cast<UINT>(w.texels), 0, D3D12_BUFFER_SRV_FLAG_RAW};
        } else {
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            view.Texture2D = {0, 1, 0, 0.0f};
        }
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = s.heap->GetCPUDescriptorHandleForHeapStart();
        D3D12_GPU_DESCRIPTOR_HANDLE handle = s.heap->GetGPUDescriptorHandleForHeapStart();
        cpu.ptr += j * increment;
        handle.ptr += j * increment;
        device->CreateShaderResourceView(w.snapshot.Get(), &view, cpu);
        barrier(list, w.snapshot.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        const std::uint32_t constants[8] = {static_cast<std::uint32_t>(w.buffer ? w.texels : w.width), w.height,
                                            static_cast<std::uint32_t>(w.texels), job.groups,
                                            static_cast<std::uint32_t>(job.word), 0, 0, 0};
        list->SetPipelineState(pipelines_.kinds[static_cast<unsigned>(w.info.sample)].Get());
        list->SetComputeRoot32BitConstants(0, 8, constants, 0);
        list->SetComputeRootDescriptorTable(2, handle);
        list->Dispatch(job.groups, 1, 1);
        barrier(list, w.snapshot.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON);
    }
    barrier(list, s.result.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(s.readback.Get(), 0, s.result.Get(), 0, bytes);
    return SUCCEEDED(list->Close());
}

// Waits (10 s at most) for the submission's reductions and returns their result words; empty when there are none.
std::vector<std::uint32_t> Tracker::read(Submission& s)
{
    std::vector<std::uint32_t> words;
    if (s.failure || s.jobs.empty()) return words;
    if (s.fence->GetCompletedValue() < s.value) {
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (event && SUCCEEDED(s.fence->SetEventOnCompletion(s.value, event))) WaitForSingleObject(event, 10000);
        if (event) CloseHandle(event);
    }
    if (s.fence->GetCompletedValue() < s.value) {
        s.failure = "reduction timeout";
        return words;
    }
    const D3D12_RANGE range{0, static_cast<SIZE_T>(s.words * 4)}, none{};
    void* memory = nullptr;
    if (FAILED(s.readback->Map(0, &range, &memory)) || !memory) {
        s.failure = "readback";
        return words;
    }
    words.assign(static_cast<const std::uint32_t*>(memory), static_cast<const std::uint32_t*>(memory) + s.words);
    s.readback->Unmap(0, &none);
    return words;
}

std::string key_text(const std::vector<u64>& keys)
{
    if (keys.empty()) return "-";
    char text[20];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(keys.front()));
    return text;
}

std::string Tracker::line(LONG frame, int index, const Report& report, const Submission& s,
                          const std::vector<std::uint32_t>& words, m::PassView& view)
{
    const Record& r = report.record;
    std::string out = "{\"present\":" + std::to_string(frame) + ",\"pass\":" + std::to_string(index) + ",\"kind\":\"" +
                      kind_name(r.kind) + "\",\"calls\":";
    out += r.first_call < 0 ? std::string("null") : "[" + std::to_string(r.first_call) + "," + std::to_string(r.last_call) + "]";
    out += ",\"recorded\":" + std::to_string(r.calls) + ",\"refused\":" + std::to_string(r.refused) + ",\"keys\":[";
    for (std::size_t k = 0; k < r.keys.size(); ++k) out += (k ? ",\"" : "\"") + key_text({r.keys[k]}) + '"';
    out += "],\"inputs\":";
    if (!r.inputs_known) {
        out += "null";
    } else {
        out += '[';
        for (std::size_t i = 0; i < r.inputs.size(); ++i) out += (i ? "," : "") + std::to_string(r.inputs[i].id);
        out += ']';
    }
    view.clear_only = r.clear_only;
    view.inputs_known = r.inputs_known;
    for (const Input& in : r.inputs) view.inputs.push_back(reinterpret_cast<std::uintptr_t>(in.resource));
    bool reduced = false, signal = false;
    out += ",\"writes\":[";
    for (std::size_t w = 0; w < r.writes.size(); ++w) {
        const Write& write = r.writes[w];
        view.writes.push_back(reinterpret_cast<std::uintptr_t>(write.resource));
        char head[192];
        std::snprintf(head, sizeof(head), "%s{\"resource\":%llu,\"format\":%d,\"view\":%d,\"width\":%llu,\"height\":%u,\"mip\":%u,\"slice\":%u",
                      w ? "," : "", static_cast<unsigned long long>(write.id), static_cast<int>(write.format),
                      static_cast<int>(write.view), static_cast<unsigned long long>(write.width), write.height, write.mip,
                      write.slice);
        out += head;
        if (write.buffer) out += ",\"offset\":" + std::to_string(write.offset) + ",\"bytes\":" + std::to_string(write.texels * 4);
        const auto job = std::find_if(s.jobs.begin(), s.jobs.end(), [&](const Job& j) {
            return &s.reports[j.report] == &report && j.write == w;
        });
        const char* reason = write.reason ? write.reason : s.failure ? s.failure : job == s.jobs.end() ? "not reduced" : nullptr;
        if (reason || job->word + u64{job->groups} * m::group_words > words.size()) {
            out += std::string(",\"stats\":null,\"reason\":\"") + (reason ? reason : "readback") + "\"}";
            continue;
        }
        const m::Stats stats = m::combine(words.data() + job->word, job->groups, write.texels, write.info.channels);
        reduced = true;
        signal |= m::has_signal(stats);
        out += ",\"stats\":" + m::stats_json(stats) + '}';
    }
    view.signal = !reduced ? m::Signal::unknown : signal ? m::Signal::signal : m::Signal::dead;
    out += "],\"signal\":";
    out += !reduced ? "null" : signal ? "true" : "false";
    out += "}\n";
    return out;
}

void Tracker::write_lines(const std::string& text)
{
    if (!file_tried_) {
        file_tried_ = true;
        char dir[MAX_PATH]{};
        DWORD n = GetEnvironmentVariableA("X2W_PASS_STATS_DIR", dir, sizeof(dir));
        if (!n || n >= sizeof(dir)) n = GetEnvironmentVariableA("X2W_PRESENT_PICTURES", dir, sizeof(dir));
        if (!n || n >= sizeof(dir)) std::snprintf(dir, sizeof(dir), ".");
        const std::string path = std::string(dir) + "\\passes.jsonl";
        file_ = std::fopen(path.c_str(), "wb");
        log("x2w-pass stats: %s %s (cap %u passes per present, %llu MiB of snapshots)", file_ ? "writing" : "cannot open",
            path.c_str(), config().cap, static_cast<unsigned long long>(config().budget >> 20));
    }
    if (!file_) return;
    std::fwrite(text.data(), 1, text.size(), file_);
    std::fflush(file_);
}

std::string pass_text(int index, const Record* record, bool refused)
{
    if (index < 0 || !record) return "none";
    std::string text = "#" + std::to_string(index) + " (key " + key_text(record->keys) + ", " + kind_name(record->kind);
    if (refused) text += ", refused calls " + std::to_string(record->refused);
    return text + ")";
}

void Tracker::report_collapse(LONG frame, ID3D12Resource* surface, const std::vector<m::PassView>& views,
                              const std::vector<const Record*>& records)
{
    const m::Collapse c = m::walk(views, reinterpret_cast<std::uintptr_t>(surface));
    const auto at = [&](int i) { return i >= 0 ? records[static_cast<std::size_t>(i)] : nullptr; };
    if (c.final_pass < 0) {
        log("x2w-replay pass collapse: present %ld: no recorded pass wrote the presented surface (resource %llu, %zu passes)",
            frame, static_cast<unsigned long long>(creation_id(surface)), views.size());
    } else if (c.last_signal == c.final_pass) {
        log("x2w-replay pass collapse: present %ld: no collapse: the last pass that wrote the presented surface, %s, has signal",
            frame, pass_text(c.final_pass, at(c.final_pass), false).c_str());
    } else if (c.first_dead < 0) {
        log("x2w-replay pass collapse: present %ld: last pass with signal = %s, no all-zero/NaN pass after it (final pass %s has no stats)",
            frame, pass_text(c.last_signal, at(c.last_signal), false).c_str(), pass_text(c.final_pass, at(c.final_pass), false).c_str());
    } else {
        log("x2w-replay pass collapse: present %ld: last pass with signal = %s, first all-zero/NaN pass after it = %s",
            frame, pass_text(c.last_signal, at(c.last_signal), false).c_str(),
            pass_text(c.first_dead, at(c.first_dead), true).c_str());
    }
}

void Tracker::present(LONG frame, ID3D12Resource* surface)
{
    std::lock_guard<std::mutex> guard(lock_);
    std::vector<m::PassView> views;
    std::vector<const Record*> records;
    std::string text;
    std::size_t reductions = 0;
    const char* failure = nullptr;
    for (Submission& s : submissions_) {
        const std::vector<std::uint32_t> words = read(s);
        reductions += s.jobs.size();
        if (s.failure) failure = s.failure;
        for (const Report& report : s.reports) {
            m::PassView view;
            text += line(frame, static_cast<int>(views.size()), report, s, words, view);
            views.push_back(std::move(view));
            records.push_back(&report.record);
        }
    }
    write_lines(text);
    const unsigned* c = counts_;
    log("x2w-pass stats: present %ld: %zu passes, %zu reductions in %zu submissions%s%s; hooks: draws %u, not drawn %u, "
        "dispatches %u, copies %u, clears %u, closes %u, lists executed %u (tracked %u)", frame, views.size(), reductions,
        submissions_.size(), failure ? ", failure: " : "", failure ? failure : "", c[draws], c[not_drawn_calls],
        c[dispatches], c[copies], c[clears], c[closes], c[executed], c[tracked]);
    std::fill(std::begin(counts_), std::end(counts_), 0u);
    report_collapse(frame, surface, views, records);
    submissions_.clear();
    passes_this_present_ = 0;
    bytes_this_present_ = 0;
}
