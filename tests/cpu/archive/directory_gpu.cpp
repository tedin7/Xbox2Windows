// Fixed-size source-built directory batch host. The executable computes both request ranges.
#include "quad_frames.hpp"

#include <array>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <io.h>
#include <cstdint>
#include <vector>

using Word = std::uint64_t;
extern "C" __declspec(dllimport) HRESULT WINAPI DStorageGetFactory(const GUID*, void**);

namespace {
struct Object { void** slots; };
using Ptr = Object*;
template<unsigned slot, class Result, class... Args>
Result invoke(Ptr self, Args... args)
{
    return reinterpret_cast<Result(WINAPI*)(Ptr, Args...)>(self->slots[slot])(self, args...);
}
struct Inputs { const Word* words; Word bytes, file_size; unsigned first, second; Word bias; };
alignas(8) std::array<Word, 8192> prefix;
const wchar_t* archive_name;
Inputs inputs{};
Ptr file = nullptr, queue = nullptr, status = nullptr;
constexpr unsigned page = 4096;
constexpr unsigned guard = 32;
struct Destination {
    unsigned page_offset = 0;
    std::vector<unsigned char> guarded;
};
std::array<Destination, 2> destinations;
unsigned count = 0;
FILE* trace = nullptr;

class CloseBatch {
public:
    ~CloseBatch() {
        if (file) invoke<3, void>(file);
        const unsigned file_refs = file ? invoke<2, unsigned>(file) : 0;
        const unsigned status_refs = status ? invoke<2, unsigned>(status) : 0;
        const unsigned queue_refs = queue ? invoke<2, unsigned>(queue) : 0;
        if (trace) {
            std::fprintf(trace, "storage release %u %u %u\n", file_refs, status_refs, queue_refs);
            std::fclose(trace);
        }
    }
};
}

extern "C" __declspec(dllexport) const Inputs* directory_input()
{
    // A fixed-size byte window limits the input to the EXE. Only the EXE interprets its directory.
    if (_setmode(_fileno(stdin), _O_BINARY) == -1) return nullptr;
    unsigned selector = 0;
    if (std::fread(&inputs.first, sizeof(inputs.first), 1, stdin) != 1 ||
        std::fread(&inputs.second, sizeof(inputs.second), 1, stdin) != 1 ||
        std::fread(&inputs.bias, sizeof(inputs.bias), 1, stdin) != 1 ||
        std::fread(&selector, sizeof(selector), 1, stdin) != 1 || selector > 1) return nullptr;
    archive_name = selector ? L"../alternate.ttpf" : L"../fixture.ttpf";
    std::ifstream input(archive_name, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() < 0) return nullptr;
    inputs.file_size = static_cast<Word>(input.tellg());
    input.seekg(0);
    inputs.words = prefix.data();
    inputs.bytes = (std::min)(inputs.file_size, static_cast<Word>(sizeof(prefix)));
    if (!input.read(reinterpret_cast<char*>(prefix.data()), inputs.bytes)) return nullptr;
    return &inputs;
}

bool begin_batch()
{
    void* result = nullptr;
    if (FAILED(DStorageGetFactory(nullptr, &result)) || !result) return false;
    auto factory = static_cast<Ptr>(result);
    const GUID* iid = nullptr;
    const HRESULT file_result = invoke<4, HRESULT>(factory, archive_name, iid, &file);
    const unsigned char queue_description[0x28] = {};
    const HRESULT queue_result = invoke<10, HRESULT>(factory, static_cast<const void*>(queue_description), iid, &queue);
    const HRESULT status_result = invoke<5, HRESULT>(factory, 2u, "BC pages", iid, &status);
    invoke<2, unsigned>(factory);
    trace = std::fopen("ranges.txt", "w");
    BY_HANDLE_FILE_INFORMATION identity{};
    if (FAILED(file_result) || FAILED(queue_result) || FAILED(status_result) || !trace ||
        FAILED((invoke<4, HRESULT>(file, &identity)))) {
        CloseBatch cleanup;
        return false;
    }
    const Word id = (Word{identity.nFileIndexHigh} << 32) | identity.nFileIndexLow;
    std::fprintf(trace, "file %016llx\n", id);
    return true;
}

extern "C" __declspec(dllexport) unsigned enqueue_page(unsigned index, Word offset, unsigned size, Word destination)
{
    const unsigned capacity = static_cast<unsigned>(destination >> 32);
    const unsigned page_offset = static_cast<unsigned>(destination);
    if (count >= 2 || !size || (capacity != page && capacity != 16 * page) ||
        page_offset % page || page_offset > capacity - page || (!count && !begin_batch())) return 0;
    auto& output = destinations[count];
    output.page_offset = page_offset;
    output.guarded.assign(guard + capacity + guard, 0xa5);
    alignas(8) unsigned char record[0x48]{};
    const Word fields[]{1, reinterpret_cast<Word>(output.guarded.data() + guard), 0, capacity,
                        reinterpret_cast<Word>(file), offset, size};
    std::memcpy(record, fields, sizeof(fields));
    invoke<3, void>(queue, record);
    invoke<4, void>(queue, status, count++);
    std::fprintf(trace, "request %u %llu %u %u %u\n", index, static_cast<unsigned long long>(offset), size, capacity, page_offset);
    return size;
}

extern "C" __declspec(dllexport) unsigned draw_pages()
{
    CloseBatch cleanup;
    if (count != 2) return 3;
    invoke<6, void>(queue);
    const auto deadline = GetTickCount64() + 5000;
    while (!invoke<3, bool>(status, 1u)) {
        if (GetTickCount64() >= deadline) return 4;
        Sleep(1);
    }
    bool failed = false;
    for (unsigned i = 0; i < count; ++i) {
        const auto hr = invoke<4, HRESULT>(status, i);
        std::fprintf(trace, "status %08lx\n", static_cast<unsigned long>(hr));
        failed |= FAILED(hr);
    }
    std::array<unsigned char, 2 * page> blocks;
    std::ofstream complete("destinations.bin", std::ios::binary);
    for (unsigned i = 0; i < count; ++i) {
        const auto& out = destinations[i];
        if (!complete.write(reinterpret_cast<const char*>(out.guarded.data()), out.guarded.size())) return 5;
        std::memcpy(blocks.data() + page * i, out.guarded.data() + guard + out.page_offset, page);
    }
    std::ofstream pages("pages.bin", std::ios::binary);
    if (!pages.write(reinterpret_cast<const char*>(blocks.data()), blocks.size())) return 5;
    if (failed) return 6;
    QuadFrames frames;
    const HRESULT created = frames.create(128);
    if (FAILED(created)) return 7;
    char adapter[320]{};
    if (FAILED(frames.adapter(adapter, sizeof(adapter)))) return 7;
    std::array<unsigned char, 128 * 128 * 4> pixels;
    TextureImage image{blocks.data(), blocks.size(), 64, 128, DXGI_FORMAT_BC7_UNORM};
    FrameResult result{};
    const HRESULT drawn = frames.draw(&image, pixels.data(), pixels.size(), result);
    const ULONG refs = frames.release();
    std::fprintf(trace, "adapter %s\nfence %llu\nrelease %lu\n", adapter, result.fence, refs);
    if (FAILED(drawn) || refs) return 8;
    return std::ofstream("pixels.bin", std::ios::binary)
        .write(reinterpret_cast<const char*>(pixels.data()), pixels.size()) ? 0 : 9;
}
