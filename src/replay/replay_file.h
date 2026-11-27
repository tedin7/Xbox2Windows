// Reads a replay recording (docs/startup-runtime.md#d3d12-replay-record, written by
// src/startup_stubs/d3d12_record.inl): the chunk framing <4s tag><u32 header bytes><u64 data bytes>, then the header,
// then the data. The file is mapped, not copied: blobs are read in place.
#pragma once

#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "../startup_stubs/replay_pages.h"

namespace x2w_replay {

using u64 = std::uint64_t;
constexpr std::uint32_t file_version = 3;  // 2 is read too: 7 argument words, rcx memory at ARGM 7
constexpr int call_args = 11;

struct CallRecord {
    u64 seq;
    std::uint32_t thread;
    std::uint16_t kind, slot;
    u64 object;
    u64 args[call_args];
    u64 ret;
    double seconds;
    std::int32_t present, pad;
};
static_assert(sizeof(CallRecord) == 136, "CALL header layout");
constexpr std::size_t call_v2_bytes = 104;  // format 2: args[7]

struct ArgMemory {
    std::uint32_t arg, depth;  // arg: the call's argument (low byte) | 0x100 * (word in the parent + 1) when nested
    u64 address, bytes, hash;
};

struct Output {
    std::uint32_t arg;
    u64 value;  // the wrapper the argument's memory held after the call
};

// A read-only run of records inside one of the Recording's pools (no allocation per call).
template <class T>
struct Span {
    const T* first = nullptr;
    std::size_t count = 0;
    const T* begin() const { return first; }
    const T* end() const { return first + count; }
    std::size_t size() const { return count; }
    const T& operator[](std::size_t i) const { return first[i]; }
};

struct Step {
    CallRecord call;
    Span<ArgMemory> memory;  // in the order written: a parent before what it points at
    Span<Output> outputs;
    const unsigned char* descriptor_arrays = nullptr;  // DARR payload, validated against the CALL counts
};

struct Blob {
    const unsigned char* data;
    u64 bytes;
};

struct ShaderFile {
    std::string name;  // a file name directly in the recorded run's shader cache
    Blob content;
};

// One MAP chunk (src/startup_stubs/replay_pages.h): page runs over [address, address + bytes) as they were at seq.
struct PageMap {
    u64 address, bytes, seq;
    std::uint32_t runs;
    const unsigned char* data;
    u64 data_bytes;
};

struct Recording {
    HANDLE file = INVALID_HANDLE_VALUE, mapping = nullptr;
    const unsigned char* base = nullptr;
    u64 size = 0;
    std::uint32_t version = 0;
    u64 frequency = 0;
    std::vector<Step> steps;  // sorted by seq
    std::vector<ArgMemory> memory_pool;  // every step's ARGM records, one run per step (Step::memory)
    std::vector<Output> output_pool;     // every step's OUTP records (Step::outputs)
    std::unordered_map<u64, Blob> blobs;
    std::vector<std::string> settings;
    std::vector<ShaderFile> shader_files;  // SHDR
    std::unordered_map<PageKey, const unsigned char*, PageKeyHash> pages;  // PAGE: key -> its 4 KiB
    std::vector<PageMap> maps;                                             // MAP, sorted by seq (stable)  // XENV: the recorded process's X2W_ settings, "name=value"
    u64 presents_marked = 0, end_calls = 0;
    bool ended = false;

    ~Recording()
    {
        if (base) UnmapViewOfFile(base);
        if (mapping) CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
    }
};

// Fills `out`; returns an empty string, or what is wrong with the file.
inline std::string read_recording(const wchar_t* path, Recording& out)
{
    out.file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (out.file == INVALID_HANDLE_VALUE) return "cannot open the recording";
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(out.file, &size) || size.QuadPart < 16) return "the recording is empty";
    out.size = static_cast<u64>(size.QuadPart);
    out.mapping = CreateFileMappingW(out.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!out.mapping) return "cannot map the recording";
    out.base = static_cast<const unsigned char*>(MapViewOfFile(out.mapping, FILE_MAP_READ, 0, 0, 0));
    if (!out.base) return "cannot map the recording";
    // The step a seq belongs to. ARGM, OUTP and DARR follow their CALL, so the last step answers almost always; the map is
    // built only when it does not (15M calls would hold about 1 GB of it).
    std::unordered_map<u64, std::size_t> by_seq;
    std::size_t indexed = 0;
    auto step_of = [&](u64 seq, std::size_t& index) {
        if (!out.steps.empty() && out.steps.back().call.seq == seq) {
            index = out.steps.size() - 1;
            return true;
        }
        for (; indexed < out.steps.size(); ++indexed) by_seq[out.steps[indexed].call.seq] = indexed;
        const auto found = by_seq.find(seq);
        if (found == by_seq.end()) return false;
        index = found->second;
        return true;
    };
    struct PendingMemory { std::size_t step; ArgMemory memory; };
    struct PendingOutput { std::size_t step; Output output; };
    std::vector<PendingMemory> pending_memory;
    std::vector<PendingOutput> pending_output;
    for (u64 at = 0; at < out.size;) {
        if (out.size - at < 16) return "cut chunk head at " + std::to_string(at);
        char tag[5]{};
        std::uint32_t header_bytes;
        u64 data_bytes;
        std::memcpy(tag, out.base + at, 4);
        std::memcpy(&header_bytes, out.base + at + 4, 4);
        std::memcpy(&data_bytes, out.base + at + 8, 8);
        const u64 header_at = at + 16, data_at = header_at + header_bytes;
        if (data_bytes > out.size || data_at > out.size - data_bytes) return std::string("cut chunk ") + tag + " at " + std::to_string(at);
        const unsigned char* h = out.base + header_at;
        const std::string name(tag);
        if (at == 0 && name != "XREC") return "not a replay recording (no XREC first)";
        if (name == "XREC" && header_bytes >= 16) {
            std::uint32_t call_bytes;
            std::memcpy(&out.version, h, 4);
            std::memcpy(&call_bytes, h + 4, 4);
            std::memcpy(&out.frequency, h + 8, 8);
            if (out.version != file_version && out.version != 2)
                return "recording version " + std::to_string(out.version) + ", this replayer reads 2 and " + std::to_string(file_version);
            if (call_bytes != (out.version == 2 ? call_v2_bytes : sizeof(CallRecord))) return "unexpected CALL size " + std::to_string(call_bytes);
        } else if (name == "CALL" && (header_bytes == sizeof(CallRecord) || (out.version == 2 && header_bytes == call_v2_bytes))) {
            Step step{};
            if (header_bytes == sizeof(CallRecord)) {
                std::memcpy(&step.call, h, sizeof(CallRecord));
            } else {
                // Format 2: seven argument words, then ret, seconds, present; rcx's memory is ARGM 7 (moved to 15 below).
                std::memcpy(&step.call, h, 24 + 7 * 8);
                std::memcpy(&step.call.ret, h + 24 + 7 * 8, call_v2_bytes - 24 - 7 * 8);
            }
            out.steps.push_back(step);
        } else if (name == "ARGM" && header_bytes >= 40) {
            u64 seq;
            ArgMemory m{};
            std::memcpy(&seq, h, 8);
            std::memcpy(&m.arg, h + 8, 4);
            std::memcpy(&m.depth, h + 12, 4);
            std::memcpy(&m.address, h + 16, 8);
            std::memcpy(&m.bytes, h + 24, 8);
            std::memcpy(&m.hash, h + 32, 8);
            if (out.version == 2 && (m.arg & 0xffu) == 7) m.arg = (m.arg & ~0xffu) | 15u;  // format 2's rcx index
            std::size_t index;
            if (!step_of(seq, index)) return "ARGM before its CALL at " + std::to_string(at);
            pending_memory.push_back({index, m});
        } else if (name == "DARR" && header_bytes == 8) {
            u64 seq;
            std::memcpy(&seq, h, 8);
            std::size_t index;
            if (!step_of(seq, index)) return "DARR before its CALL at " + std::to_string(at);
            Step& step = out.steps[index];
            if (step.call.kind != 0 || step.call.slot != 23 || step.descriptor_arrays)
                return "invalid DARR call at " + std::to_string(at);
            const auto& args = step.call.args;
            u64 expected = 0;
            for (const unsigned arg : {1u, 2u, 4u, 5u})
                if (args[arg]) expected += u64{static_cast<std::uint32_t>(args[arg < 3 ? 0 : 3])} *
                                          ((arg == 1 || arg == 4) ? 8 : 4);
            if (data_bytes != expected) return "DARR size mismatch at " + std::to_string(at);
            step.descriptor_arrays = out.base + data_at;
        } else if (name == "OUTP" && header_bytes >= 24) {
            u64 seq;
            Output o{};
            std::memcpy(&seq, h, 8);
            std::memcpy(&o.arg, h + 8, 4);
            std::memcpy(&o.value, h + 16, 8);
            std::size_t index;
            if (!step_of(seq, index)) return "OUTP before its CALL at " + std::to_string(at);
            pending_output.push_back({index, o});
        } else if (name == "BLOB" && header_bytes >= 16) {
            u64 hash, bytes;
            std::memcpy(&hash, h, 8);
            std::memcpy(&bytes, h + 8, 8);
            if (bytes != data_bytes) return "BLOB size mismatch at " + std::to_string(at);
            out.blobs[hash] = Blob{out.base + data_at, bytes};
        } else if (name == "XENV") {
            const char* text = reinterpret_cast<const char*>(out.base + data_at);
            for (u64 i = 0; i < data_bytes;) {
                const u64 length = strnlen(text + i, static_cast<std::size_t>(data_bytes - i));
                if (length) out.settings.emplace_back(text + i, static_cast<std::size_t>(length));
                i += length + 1;
            }
        } else if (name == "SHDR" && header_bytes) {
            out.shader_files.push_back({std::string(reinterpret_cast<const char*>(h), header_bytes), Blob{out.base + data_at, data_bytes}});
        } else if (name == "PAGE" && header_bytes == sizeof(PageKey)) {
            if (data_bytes != page_bytes) return "PAGE of " + std::to_string(data_bytes) + " bytes at " + std::to_string(at);
            PageKey key;
            std::memcpy(&key, h, sizeof(key));
            out.pages[key] = out.base + data_at;
        } else if (name == "MAP " && header_bytes == sizeof(MapHeader)) {
            MapHeader map;
            std::memcpy(&map, h, sizeof(map));
            if (map.address % page_bytes || map.bytes % page_bytes) return "MAP not page aligned at " + std::to_string(at);
            out.maps.push_back(PageMap{map.address, map.bytes, map.seq, map.runs, out.base + data_at, data_bytes});
        } else if (name == "PRES") {
            ++out.presents_marked;
        } else if (name == "XEND" && header_bytes >= 8) {
            std::memcpy(&out.end_calls, h, 8);
            out.ended = true;
        }
        at = data_at + data_bytes;
    }
    if (!out.version) return "no XREC header";
    // One run per step, in the order written: count, prefix sums, then place (stable, linear).
    std::vector<std::size_t> start(out.steps.size() + 1, 0);
    out.memory_pool.resize(pending_memory.size());
    for (const auto& item : pending_memory) ++start[item.step + 1];
    for (std::size_t i = 0; i < out.steps.size(); ++i) start[i + 1] += start[i];
    {
        std::vector<std::size_t> fill(start.begin(), start.end() - 1);
        for (const auto& item : pending_memory) out.memory_pool[fill[item.step]++] = item.memory;
        for (std::size_t i = 0; i < out.steps.size(); ++i) out.steps[i].memory = {out.memory_pool.data() + start[i], start[i + 1] - start[i]};
    }
    std::fill(start.begin(), start.end(), 0);
    out.output_pool.resize(pending_output.size());
    for (const auto& item : pending_output) ++start[item.step + 1];
    for (std::size_t i = 0; i < out.steps.size(); ++i) start[i + 1] += start[i];
    {
        std::vector<std::size_t> fill(start.begin(), start.end() - 1);
        for (const auto& item : pending_output) out.output_pool[fill[item.step]++] = item.output;
        for (std::size_t i = 0; i < out.steps.size(); ++i) out.steps[i].outputs = {out.output_pool.data() + start[i], start[i + 1] - start[i]};
    }
    std::vector<PendingMemory>().swap(pending_memory);
    std::vector<PendingOutput>().swap(pending_output);
    std::sort(out.steps.begin(), out.steps.end(), [](const Step& a, const Step& b) { return a.call.seq < b.call.seq; });
    std::stable_sort(out.maps.begin(), out.maps.end(), [](const PageMap& a, const PageMap& b) { return a.seq < b.seq; });
    return {};
}

}  // namespace x2w_replay
