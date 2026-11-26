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

namespace x2w_replay {

using u64 = std::uint64_t;
constexpr std::uint32_t file_version = 2;

struct CallRecord {
    u64 seq;
    std::uint32_t thread;
    std::uint16_t kind, slot;
    u64 object;
    u64 args[7];
    u64 ret;
    double seconds;
    std::int32_t present, pad;
};
static_assert(sizeof(CallRecord) == 104, "CALL header layout");

struct ArgMemory {
    std::uint32_t arg, depth;  // arg: the call's argument (low byte) | 0x100 * (word in the parent + 1) when nested
    u64 address, bytes, hash;
};

struct Output {
    std::uint32_t arg;
    u64 value;  // the wrapper the argument's memory held after the call
};

struct Step {
    CallRecord call;
    std::vector<ArgMemory> memory;  // in the order written: a parent before what it points at
    std::vector<Output> outputs;
};

struct Blob {
    const unsigned char* data;
    u64 bytes;
};

struct Recording {
    HANDLE file = INVALID_HANDLE_VALUE, mapping = nullptr;
    const unsigned char* base = nullptr;
    u64 size = 0;
    std::uint32_t version = 0;
    u64 frequency = 0;
    std::vector<Step> steps;  // sorted by seq
    std::unordered_map<u64, Blob> blobs;
    std::vector<std::string> settings;  // XENV: the recorded process's X2W_ settings, "name=value"
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
    std::unordered_map<u64, std::size_t> by_seq;
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
            if (out.version != file_version) return "recording version " + std::to_string(out.version) + ", this replayer reads " + std::to_string(file_version);
            if (call_bytes != sizeof(CallRecord)) return "unexpected CALL size " + std::to_string(call_bytes);
        } else if (name == "CALL" && header_bytes == sizeof(CallRecord)) {
            Step step{};
            std::memcpy(&step.call, h, sizeof(CallRecord));
            by_seq[step.call.seq] = out.steps.size();
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
            const auto found = by_seq.find(seq);
            if (found == by_seq.end()) return "ARGM before its CALL at " + std::to_string(at);
            out.steps[found->second].memory.push_back(m);
        } else if (name == "OUTP" && header_bytes >= 24) {
            u64 seq;
            Output o{};
            std::memcpy(&seq, h, 8);
            std::memcpy(&o.arg, h + 8, 4);
            std::memcpy(&o.value, h + 16, 8);
            const auto found = by_seq.find(seq);
            if (found == by_seq.end()) return "OUTP before its CALL at " + std::to_string(at);
            out.steps[found->second].outputs.push_back(o);
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
        } else if (name == "PRES") {
            ++out.presents_marked;
        } else if (name == "XEND" && header_bytes >= 8) {
            std::memcpy(&out.end_calls, h, 8);
            out.ended = true;
        }
        at = data_at + data_bytes;
    }
    if (!out.version) return "no XREC header";
    std::sort(out.steps.begin(), out.steps.end(), [](const Step& a, const Step& b) { return a.call.seq < b.call.seq; });
    return {};
}

}  // namespace x2w_replay
