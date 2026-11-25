// Run the actual provider lookup and methods with a source-built QueryApiImpl service.
#include "../src/startup_stubs/runtime_proxy.cpp"
#include <array>

namespace {
const GUID provider{0x704c3f58,0xe629,0x4cc2,{0xb1,0x97,0x30,0x51,0x1b,0x99,0x6f,0xe2}};
const GUID interface_id{0x1bfff3af,0xf14a,0x40a3,{0x8e,0x35,0x9a,0xda,0x90,0x65,0x93,0xf9}};
constexpr std::array<unsigned char,4> language{11,0,0xff,0};
constexpr std::array<unsigned char,2> brightness{0,15};

QueryApi system_async_query{};
HRESULT WINAPI fixture_query(const GUID* cls, const GUID* iid, void** out)
{
    if (*cls == submit_async_family)
        return system_async_query ? system_async_query(cls,iid,out) : E_NOINTERFACE;
    static void* token;
    *out = &token;
    return S_OK;
}
struct Client {
    void* object{};
    u64 call(int slot, u64 a=0, u64 b=0, u64 c=0, u64 d=0, u64 e=0) {
        auto method = reinterpret_cast<Method>((*reinterpret_cast<void***>(object))[slot]);
        return method(object, a,b,c,d,e,0,0,0,0,0,0);
    }
};
template<class T> u64 address(T* p) { return reinterpret_cast<u64>(p); }
struct Found { std::string name; unsigned calls{}; };
bool WINAPI find_container(const save_read::ContainerInfo* info, void* state) {
    auto& found = *static_cast<Found*>(state);
    ++found.calls;
    if (info->blob_count == 2 && info->bytes == language.size() + brightness.size() &&
        info->modified > 0 && !info->needs_sync && std::strcmp(info->display_name, info->name) == 0)
        found.name = info->name;
    return true;
}
bool WINAPI stop_container(const save_read::ContainerInfo*, void* count) {
    ++*static_cast<unsigned*>(count);
    return false;
}
struct Metadata { std::vector<std::string> names; std::size_t required{}; std::string container; bool lock_free = true; };
bool WINAPI collect_blob(const save_read::BlobInfo* info, void* state) {
    auto& data = *static_cast<Metadata*>(state);
    data.names.emplace_back(info->name);
    if (!data.container.empty()) {
        // No store lock may be held while the caller's callback runs: a writer must be able to lock.
        x2w_save::FileLock probe;
        const auto file = std::filesystem::path(storage_path()) / "x2w-gamesave" / (data.container + std::string(x2w_save::lock_suffix));
        if (probe.acquire(file, true, std::chrono::milliseconds(0)) != x2w_save::LockResult::locked) data.lock_free = false;
    }
    data.required += sizeof(save_read::Blob) + std::strlen(info->name) + 1 + info->size;
    return true;
}
int write(Client& client, u64 p, bool asynchronous) {
    const auto generated = "preferences-" + std::to_string(GetCurrentProcessId());
    u64 saved{}, empty{}, update{};
    if (client.call(16, p, address("empty"), address(&empty)) ||
        client.call(16, p, address(generated.c_str()), address(&saved)) ||
        client.call(23, saved, 0, address(&update)) ||
        client.call(25, update, address("language"), address(language.data()), language.size()) ||
        client.call(25, update, address("brightness"), address(brightness.data()), brightness.size())) return 2;
    if (asynchronous) {
        const auto dll = LoadLibraryExW(L"xgameruntime.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!dll) return 77;
        system_async_query = reinterpret_cast<QueryApi>(GetProcAddress(dll,"QueryApiImpl"));
        void* runtime{};
        if (!system_async_query || FAILED(system_async_query(&submit_async_family,&submit_async_family,&runtime)) || !runtime)
            return 77;
        auto** methods = *static_cast<void***>(runtime);
        struct { void* queue{}; void* ctx{}; void* callback{}; unsigned char internal[32]{}; } block;
        using CreateQueue = HRESULT(WINAPI*)(void*,unsigned,unsigned,void**);
        if (FAILED(reinterpret_cast<CreateQueue>(methods[12])(runtime,3,3,&block.queue))) return 77;
        const char* key="language";
        std::uint32_t count=1;
        alignas(8) unsigned char data[128]{};
        if (client.call(28,update,0) != e_fail ||
            client.call(20,saved,address(&key),address(&count),sizeof(data),address(data)) != save_read::missing_blob ||
            client.call(28,update,address(&block)) || client.call(29,address(&block))) return 13;
        reinterpret_cast<void(WINAPI*)(void*,void*)>(methods[17])(runtime,block.queue);
        reinterpret_cast<ULONG(WINAPI*)(void*)>(methods[2])(runtime);
    } else if (client.call(27, update)) return 2;
    client.call(24, update);
    client.call(17, saved);
    client.call(17, empty);
    return 0;
}
int read(Client& client, u64 p) {
    Found found, exact;
    unsigned stopped{};
    if (client.call(14, p, address(&found), address(&find_container)) || found.calls != 2 || found.name.empty() ||
        client.call(14, p, address(&stopped), address(&stop_container)) || stopped != 1 ||
        client.call(13, p, address(found.name.c_str()), address(&exact), address(&find_container)) ||
        exact.calls != 1 || exact.name != found.name ||
        client.call(13, p, address("absent"), address(&exact), address(&find_container)) != S_OK ||
        exact.calls != 1) return 3;
    u64 saved{};
    Metadata meta;
    meta.container = found.name;
    if (client.call(16, p, address(found.name.c_str()), address(&saved)) ||
        client.call(18, saved, address(&meta), address(&collect_blob)) || meta.names.size() != 2) return 4;
    if (!meta.lock_free) return 15;
    std::sort(meta.names.begin(), meta.names.end());
    if (meta.names != std::vector<std::string>{"brightness","language"}) return 5;
    // Slot 19 (by name prefix): only the matching blob, in either case; a success must call back.
    Metadata by_prefix;
    if (client.call(19, saved, address("LANG"), address(&by_prefix), address(&collect_blob)) ||
        by_prefix.names != std::vector<std::string>{"language"} ||
        by_prefix.required != sizeof(save_read::Blob) + 9 + language.size()) return 14;
    const char* names[]{meta.names[0].c_str(), meta.names[1].c_str()};
    std::vector<u64> words((meta.required + 7)/8 + 2, ~u64{});
    auto* records = reinterpret_cast<save_read::Blob*>(words.data() + 1);
    std::uint32_t count = 2;
    const auto unchanged = words;
    if (client.call(20, saved, address(names), address(&count), meta.required - 1, address(records)) != 0x80830007 ||
        words != unchanged) return 6;
    names[1] = "missing";
    if (client.call(20, saved, address(names), address(&count), meta.required, address(records)) != 0x80830008 ||
        words != unchanged) return 7;
    names[1] = meta.names[1].c_str();
    if (client.call(20, saved, address(names), address(&count), meta.required, address(records)) || count != 2) return 8;
    const auto inside = [&](const void* ptr, std::size_t bytes) {
        const auto location = address(ptr), base = address(records);
        return location >= base && location - base <= meta.required && bytes <= meta.required - (location - base);
    };
    for (unsigned i = 0; i < count; ++i) {
        const auto& record = records[i];
        const auto size = i ? language.size() : brightness.size();
        if (record.info.size != size || !inside(record.info.name,meta.names[i].size()+1) ||
            std::strcmp(names[i],record.info.name) || !inside(record.data,size) ||
            std::memcmp(record.data,i ? language.data() : brightness.data(),size)) return 9;
    }
    if (words.front() != ~u64{} || reinterpret_cast<unsigned char*>(records)[meta.required] != 0xff ||
        words.back() != ~u64{}) return 10;
    count = 0;
    const auto filled = words;
    if (client.call(20, saved, 0, address(&count), meta.required - 1, address(records)) != 0x80830007 ||
        words != filled || count != 0) return 16;
    if (client.call(20, saved, 0, address(&count), meta.required, address(records)) || count != 2) return 11;
    client.call(17, saved);
    // A closed container handle is refused by every synchronous query.
    Metadata closed;
    count = 2;
    if (client.call(18, saved, address(&closed), address(&collect_blob)) != e_invalidarg ||
        client.call(19, saved, address("lang"), address(&closed), address(&collect_blob)) != e_invalidarg ||
        client.call(20, saved, address(names), address(&count), meta.required, address(records)) != e_invalidarg ||
        !closed.names.empty()) return 17;
    auto other_id = interface_id;
    ++other_id.Data1;
    Client other;
    Found uncalled;
    if (FAILED(proxy_query(&provider,&other_id,&other.object)) ||
        other.call(14, p, address(&uncalled), address(&find_container)) || uncalled.calls) return 12;
    std::printf("discovered %s, 2 blobs, %zu packed bytes, 6 payload bytes; short and missing kept guards\n",
        found.name.c_str(), meta.required);
    return 0;
}
}
int main(int argc, char** argv) {
    if (argc != 2) return 1;
    SetEnvironmentVariableA("X2W_FAKE_USER","1");
    fill_slots(std::make_index_sequence<max_slots>{});
    real_query = fixture_query;
    Client client;
    u64 handle{};
    if (FAILED(proxy_query(&provider,&interface_id,&client.object)) ||
        client.call(3, local_user_handle, address("fixture"), 0, address(&handle)) || !handle) return 1;
    if (std::strcmp(argv[1],"write-async") == 0) return write(client,handle,true);
    return std::strcmp(argv[1],"write") == 0 ? write(client,handle,false) : read(client,handle);
}
