// Synchronous operations on the public GDK save provider identity. Every answer comes from the
// store's shared publication lock (save_store.h), so a commit from any process is seen whole or
// not at all within one call.
namespace save_read {
using BlobInfo = x2w_save_pack::BlobInfo;  // the caller-buffer layout shared with slots 21 and 22
using Blob = x2w_save_pack::Blob;
struct ContainerInfo {
    const char* name;
    const char* display_name;
    std::uint32_t blob_count;
    u64 bytes;
    std::time_t modified;
    bool needs_sync;
};
static_assert(sizeof(ContainerInfo) == 48 && offsetof(ContainerInfo, bytes) == 24);
constexpr u64 small_buffer = 0x80830007, missing_blob = 0x80830008;

// Name of an open container handle; a closed handle is empty, as for the asynchronous read.
std::string container_name(u64 handle)
{
    if (handle < save_container_handle || handle - save_container_handle >= max_save_handles) return {};
    AcquireSRWLockShared(&save_lock);
    struct Unlock { ~Unlock() { ReleaseSRWLockShared(&save_lock); } } unlock;
    return save_table.container_name(static_cast<std::size_t>(handle - save_container_handle));
}

std::time_t modified_time(const std::string& container)
{
    std::error_code ec;
    const auto file_time = std::filesystem::last_write_time(std::filesystem::path(storage_path()) / "x2w-gamesave" / container, ec);
    if (ec) return std::time(nullptr);
    const auto elapsed = file_time - std::filesystem::file_time_type::clock::now();
    return std::chrono::system_clock::to_time_t(std::chrono::system_clock::now() +
        std::chrono::duration_cast<std::chrono::system_clock::duration>(elapsed));
}

// 13: GetContainerInfo(provider, name, context, callback); 14: EnumerateContainerInfo(provider,
// context, callback). Each container's blob count and bytes come from one list_blobs call.
// Names match case-insensitively, as the store's Windows directories do.
u64 containers(bool exact, u64* args)
{
    const auto callback = reinterpret_cast<bool(WINAPI*)(const ContainerInfo*, void*)>(args[exact ? 3 : 2]);
    auto* context = reinterpret_cast<void*>(args[exact ? 2 : 1]);
    std::string wanted;
    if (args[0] != save_provider_handle || !callback ||
        (exact && (!read_save_name(args[1], wanted) || !x2w_save::valid_name(wanted)))) return e_invalidarg;
    struct Snapshot { std::string name; std::uint32_t count; u64 size; std::time_t modified; };
    std::vector<Snapshot> snapshots;
    const auto store = save_store();
    std::vector<x2w_save::ContainerEntry> entries;
    if (auto s = store.list_containers(entries); s != x2w_save::Status::ok) return save_status(s);
    for (const auto& entry : entries) {
        if (exact && _stricmp(entry.name.c_str(), wanted.c_str()) != 0) continue;
        std::vector<x2w_save::BlobEntry> blobs;
        const auto s = store.list_blobs(entry.name, blobs);
        if (s == x2w_save::Status::not_found) continue;  // deleted since the container list was taken
        if (s != x2w_save::Status::ok) return save_status(s);
        if (blobs.size() > UINT32_MAX) return e_fail;
        u64 bytes = 0;
        for (const auto& blob : blobs) bytes += blob.bytes;
        snapshots.push_back({entry.name, static_cast<std::uint32_t>(blobs.size()), bytes, modified_time(entry.name)});
    }
    // A missing name retains the stand-in's empty lookup result.
    for (const auto& item : snapshots) {
        const char* name = item.name.c_str();
        const ContainerInfo info{name, name, item.count, item.size, item.modified, false};
        if (!callback(&info, context)) break;
    }
    return 0;
}

// 18: EnumerateBlobInfo(container, context, callback); 19: EnumerateBlobInfoByName(container,
// prefix, context, callback). One list_blobs call; the prefix matches case-insensitively, as the
// store's Windows file names do.
u64 blobs(u64* args, bool by_name)
{
    const auto name = container_name(args[0]);
    const auto callback = reinterpret_cast<bool(WINAPI*)(const BlobInfo*, void*)>(args[by_name ? 3 : 2]);
    auto* context = reinterpret_cast<void*>(args[by_name ? 2 : 1]);
    std::string prefix;
    if (name.empty() || !callback || (by_name && args[1] && !read_save_name(args[1], prefix))) return e_invalidarg;
    std::vector<x2w_save::BlobEntry> entries;
    if (auto s = save_store().list_blobs(name, entries); s != x2w_save::Status::ok) return save_status(s);
    for (const auto& entry : entries) {
        if (_strnicmp(entry.name.c_str(), prefix.c_str(), prefix.size()) != 0) continue;
        const BlobInfo info{entry.name.c_str(), static_cast<std::uint32_t>(entry.bytes)};
        if (!callback(&info, context)) break;
    }
    return 0;
}

// 20: ReadBlobData(container, names, &count, capacity, records). The requested set is read with
// one read_blobs call; a count of 0 means every blob, listed and read under one list_blobs call
// whose needed size is checked against the capacity before any payload is read. Either way the
// capacity is checked against that one snapshot, and the destination is written only on success.
u64 read(u64* args)
{
    const auto name = container_name(args[0]);
    const auto* names = reinterpret_cast<const u64*>(args[1]);
    auto* count = reinterpret_cast<std::uint32_t*>(args[2]);
    const auto capacity = static_cast<std::size_t>(args[3]);
    if (name.empty() || !x2w_local::writable(count, sizeof(*count))) return e_invalidarg;
    const auto requested = *count;
    if (requested > capacity / sizeof(Blob)) return small_buffer;
    if (requested && !x2w_local::readable(names, sizeof(*names) * u64{requested})) return e_invalidarg;
    std::vector<x2w_save_pack::Piece> pieces(requested);
    std::vector<std::string> wanted(requested);
    for (std::uint32_t i = 0; i < requested; ++i)
        if (!read_save_name(names[i], wanted[i]) || !x2w_save::valid_name(wanted[i])) return e_invalidarg;
    const auto store = save_store();
    std::vector<std::vector<unsigned char>> data;
    if (requested) {
        if (auto s = store.read_blobs(name, wanted, data); s != x2w_save::Status::ok) return save_status(s);
        for (std::uint32_t i = 0; i < requested; ++i) pieces[i].name = std::move(wanted[i]);
    } else {
        std::vector<x2w_save::BlobEntry> all;
        bool fits = false;
        const auto s = store.list_blobs(name, all, &data, [&](const std::vector<x2w_save::BlobEntry>& listing) {
            std::vector<x2w_save_pack::PieceSize> sizes;
            for (const auto& entry : listing) sizes.push_back({entry.name.size(), static_cast<std::size_t>(entry.bytes)});
            std::size_t needed = 0;
            fits = x2w_save_pack::bytes_needed(sizes.data(), sizes.size(), needed) && needed <= capacity;
            return fits;
        });
        if (s != x2w_save::Status::ok) return save_status(s);
        if (!fits) return small_buffer;
        pieces.resize(all.size());
        for (std::size_t i = 0; i < all.size(); ++i) pieces[i].name = std::move(all[i].name);
    }
    if (data.size() != pieces.size() || pieces.size() > UINT32_MAX) return e_fail;
    for (std::size_t i = 0; i < pieces.size(); ++i) pieces[i].bytes = std::move(data[i]);
    std::size_t bytes = 0;
    if (!x2w_save_pack::bytes_needed(pieces, bytes) || bytes > capacity) return small_buffer;
    if (bytes && (!x2w_local::writable(reinterpret_cast<void*>(args[4]), bytes) || args[4] % alignof(Blob))) return e_invalidarg;
    x2w_save_pack::write(pieces, reinterpret_cast<void*>(args[4]));
    *count = static_cast<std::uint32_t>(pieces.size());
    return 0;
}
}

bool save_read_call(int slot, u64* args, u64* result)
{
    try {
        switch (slot) {
        case 13: case 14: *result = save_read::containers(slot == 13, args); break;
        case 18: case 19: *result = save_read::blobs(args, slot == 19); break;
        case 20: *result = save_read::read(args); break;
        default: return false;
        }
    } catch (const std::bad_alloc&) {
        *result = e_outofmemory;
    } catch (const std::length_error&) {
        *result = e_outofmemory;
    } catch (const std::filesystem::filesystem_error&) {
        *result = e_fail;
    }
    return true;
}
