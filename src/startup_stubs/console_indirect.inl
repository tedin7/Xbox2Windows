// Included by d3d12_xs_device.cpp after the command-list state helpers.
enum class ConsoleIndirectPipeline {
    none, desktop_compute, console_compute, console_compute_ready, console_graphics, console_mesh, desktop_graphics, count
};

bool console_compute_ready(const ComputeManifest* manifest);

ConsoleIndirectPipeline console_indirect_pipeline(const Pass& pass)
{
    using Pipeline = ConsoleIndirectPipeline;
    if (!pass.pipeline) return Pipeline::none;
    const auto& state = pass.pipeline->own();
    if (pass.compute_pipeline) return Pipeline::desktop_compute;
    if (state.console_compute)
        return console_compute_ready(state.compute_manifest) ? Pipeline::console_compute_ready : Pipeline::console_compute;
    if (state.console_mesh || pass.indirect_graphics.mesh) return Pipeline::console_mesh;
    return state.real ? Pipeline::desktop_graphics : Pipeline::console_graphics;
}

enum ConsoleIndirectWrite : unsigned { indirect_dispatch_write = 1, indirect_copy_write = 2 };
std::atomic<LONG> console_indirect_frame{};

// Bounded history of recorded writes. Busy or overwritten entries are missed rather than
// stalling a recording thread. A match describes a recorded write, not a completed GPU operation.
struct ConsoleIndirectWriteRecord {
    std::atomic<u64> sequence{}, offset{}, bytes{};
    std::atomic<const IUnknown*> buffer{};
    std::atomic<LONG> frame{};
    std::atomic<unsigned> source{};
};
struct ConsoleIndirectWriteBucket {
    std::atomic<unsigned> next{};
    ConsoleIndirectWriteRecord records[8];
};
ConsoleIndirectWriteBucket console_indirect_writes[4096];

ConsoleIndirectWriteBucket& console_indirect_write_bucket(const IUnknown* buffer)
{
    return console_indirect_writes[(reinterpret_cast<std::uintptr_t>(buffer) >> 8) % std::size(console_indirect_writes)];
}

void note_console_indirect_write(const IUnknown* buffer, u64 offset, u64 bytes, ConsoleIndirectWrite source)
{
    if (!buffer || !bytes) return;
    auto& bucket = console_indirect_write_bucket(buffer);
    auto& record = bucket.records[bucket.next.fetch_add(1, std::memory_order_relaxed) % std::size(bucket.records)];
    auto sequence = record.sequence.load();
    if ((sequence & 1) || !record.sequence.compare_exchange_strong(sequence, sequence + 1)) return;
    record.buffer.store(buffer);
    record.offset.store(offset);
    record.bytes.store(bytes);
    record.frame.store(console_indirect_frame.load(std::memory_order_relaxed));
    record.source.store(source);
    record.sequence.store(sequence + 2);
}

unsigned console_indirect_write_sources(const IUnknown* buffer, u64 offset, u64 bytes)
{
    if (!buffer || !bytes) return 0;
    unsigned sources = 0;
    const LONG frame = console_indirect_frame.load(std::memory_order_relaxed);
    for (const auto& record : console_indirect_write_bucket(buffer).records) {
        const auto sequence = record.sequence.load();
        if (!sequence || (sequence & 1) || record.frame.load() != frame || record.buffer.load() != buffer) continue;
        const auto first = record.offset.load();
        const auto size = record.bytes.load();
        const auto source = record.source.load();
        if (sequence == record.sequence.load() && x2w_startup::indirect_ranges_overlap(offset, bytes, first, size))
            sources |= source;
    }
    return sources;
}

constexpr int console_list_types = 8; // D3D12 types 0 through 6, then unknown.
constexpr int console_pipeline_types = static_cast<int>(ConsoleIndirectPipeline::count);
volatile LONG64 console_indirect_refusals[console_list_types][console_pipeline_types][2][4];

void note_console_indirect_refusal(int type, ConsoleIndirectPipeline pipeline, const Wrapper* buffer, u64 offset, u64 bytes)
{
    const int list = type >= 0 && type < console_list_types - 1 ? type : console_list_types - 1;
    auto* native = buffer ? InterlockedCompareExchangePointer(
        reinterpret_cast<void* volatile*>(&buffer->own().real), nullptr, nullptr) : nullptr;
    const bool desktop = native != nullptr;
    const auto sources = console_indirect_write_sources(static_cast<IUnknown*>(native), offset, bytes);
    InterlockedIncrement64(&console_indirect_refusals[list][static_cast<int>(pipeline)][desktop][sources]);
}

void log_console_indirect_refusals()
{
    constexpr const char* lists[] = {"direct", "bundle", "compute", "copy", "video-decode", "video-process", "video-encode", "unknown"};
    constexpr const char* pipelines[] = {"none", "desktop-compute", "console-compute", "console-compute-ready",
                                         "console-graphics", "console-mesh", "desktop-graphics"};
    struct Row { LONG64 count; int list, pipeline, desktop, sources; };
    std::array<Row, console_list_types * console_pipeline_types * 8> rows{};
    std::size_t used = 0;
    for (int list = 0; list < console_list_types; ++list)
        for (int pipeline = 0; pipeline < console_pipeline_types; ++pipeline)
            for (int desktop = 0; desktop < 2; ++desktop)
                for (int sources = 0; sources < 4; ++sources)
                    if (const auto count = InterlockedCompareExchange64(&console_indirect_refusals[list][pipeline][desktop][sources], 0, 0))
                        rows[used++] = {count, list, pipeline, desktop, sources};
    std::sort(rows.begin(), rows.begin() + used, [](const Row& a, const Row& b) { return a.count > b.count; });
    const std::size_t detailed = used <= 8 ? used : 7;
    for (std::size_t i = 0; i < detailed; ++i) {
        const auto& row = rows[i];
        log("x2w-render indirect console-signature: list=%s pipeline=%s desktop=%d dispatch-write=%d copy-write=%d calls=%lld",
            lists[row.list], pipelines[row.pipeline], row.desktop, bool(row.sources & indirect_dispatch_write),
            bool(row.sources & indirect_copy_write), static_cast<long long>(row.count));
    }
    if (used > detailed) {
        LONG64 calls = 0;
        for (auto i = detailed; i < used; ++i) calls += rows[i].count;
        log("x2w-render indirect console-signature: other rows=%zu calls=%lld", used - detailed, static_cast<long long>(calls));
    }
}

extern "C" __declspec(dllexport) void x2w_d3d12_fixture_console_indirect_totals()
{
    log_console_indirect_refusals();
}

ID3D12CommandSignature* console_indirect_signature(WrapperOwn& state, IndirectCommand command)
{
    static_assert(sizeof(D3D12_DISPATCH_ARGUMENTS) == 12 && sizeof(D3D12_DISPATCH_MESH_ARGUMENTS) == 12);
    const bool mesh = command == IndirectCommand::dispatch_mesh;
    auto* slot = mesh ? &state.console_mesh_signature : &state.console_dispatch_signature;
    if (auto* existing = InterlockedCompareExchangePointer(slot, nullptr, nullptr))
        return static_cast<ID3D12CommandSignature*>(existing);
    D3D12_INDIRECT_ARGUMENT_DESC argument{};
    argument.Type = mesh ? D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH : D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    const D3D12_COMMAND_SIGNATURE_DESC desc{state.indirect_bytes, 1, &argument, 0};
    ID3D12CommandSignature* created = nullptr;
    if (FAILED(device_of(state.signature_device)->CreateCommandSignature(&desc, nullptr, IID_PPV_ARGS(&created)))) return nullptr;
    // Like other native signatures held by wrappers, keep the published object alive for recorded lists.
    if (auto* existing = InterlockedCompareExchangePointer(slot, created, nullptr)) {
        created->Release();
        return static_cast<ID3D12CommandSignature*>(existing);
    }
    return created;
}
