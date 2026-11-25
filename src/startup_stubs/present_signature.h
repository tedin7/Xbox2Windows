// Experimental (docs/startup-runtime.md#input-script): a coarse signature of the presented picture,
// shared between the D3D12 stand-in (writer) and the GameInput stand-in (reader) of one process.
//
// The block is a named page-file mapping "Local\x2w_present_signature_<process id>", so each
// process has its own and an external tool can open it by process id. Layout (little endian):
//   +0x00 u32 magic "XSIG", u32 cells_x (32), u32 cells_y (18), u32 sampling interval in presents
//   +0x10 i64 sequence: odd while a sample is being written, twice the number of published samples otherwise
//   +0x18 i64 present number of the published sample
//   +0x20 i64 present number of the latest present (updated on every present)
//   +0x28 u8 luma[18][32]: mean luma (2R + 5G + B) / 8 of each 1/32 of 18 evenly spaced rows
//   +0x268 i32 fine wanted: set to 1 by a reader (request_fine) to ask for the fine grid as well
//   +0x26C u32 fine_x (64), u32 fine_y (72), i32 fine valid: the published sample has the fine grid
//   +0x278 u8 fine[72][64]: the same reduction over 72 rows and 64 columns
//   +0x1478 i32 sampling wanted: set to 1 by a reader (request_sampling) to ask for samples at all
// A reader copies the cells between two reads of an even, unchanged sequence. Sampling records GPU
// copies, so the presenter samples only after a reader asked for it (or X2W_PRESENT_SIGNATURE_EVERY
// is set); the fine grid costs 72 more one-texel rows per sample and is copied only while asked for.
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cwchar>

namespace x2w_present_signature {

constexpr int cells_x = 32, cells_y = 18, cells = cells_x * cells_y;
constexpr int fine_x = 64, fine_y = 72, fine_cells = fine_x * fine_y;
constexpr std::uint32_t block_magic = 0x47495358;  // "XSIG"

struct Block {
    std::uint32_t magic, cells_x, cells_y, every;
    volatile LONG64 sequence;
    volatile LONG64 sample_present;
    volatile LONG64 presents;
    std::uint8_t luma[cells];
    volatile LONG fine_wanted;
    std::uint32_t fine_x, fine_y;
    volatile LONG fine_valid;
    std::uint8_t fine[fine_cells];
    volatile LONG sampling_wanted;
};
static_assert(offsetof(Block, luma) == 0x28 && offsetof(Block, fine_wanted) == 0x268 && offsetof(Block, fine) == 0x278 &&
                  offsetof(Block, sampling_wanted) == 0x1478,
              "signature block layout");

struct Snapshot {
    std::uint64_t sequence;  // number of published samples (0 = none yet)
    std::uint64_t sample_present, presents;
    std::uint8_t luma[cells];
    bool fine_valid;
    std::uint8_t fine[fine_cells];
};

inline void block_name(wchar_t (&out)[64], DWORD process)
{
    std::swprintf(out, 64, L"Local\\x2w_present_signature_%lu", static_cast<unsigned long>(process));
}

// Writer side: creates (or opens) this process's block. The mapping handle stays open for the process.
inline Block* create_block(DWORD process, std::uint32_t every)
{
    wchar_t name[64];
    block_name(name, process);
    HANDLE mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(Block), name);
    if (!mapping) return nullptr;
    auto* block = static_cast<Block*>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Block)));
    if (!block) return nullptr;
    block->cells_x = cells_x;
    block->cells_y = cells_y;
    block->fine_x = fine_x;
    block->fine_y = fine_y;
    block->every = every;
    InterlockedExchange(reinterpret_cast<volatile LONG*>(&block->magic), static_cast<LONG>(block_magic));
    return block;
}

// Reader side: nullptr when no presenter has created the block (yet).
inline const Block* open_block(DWORD process)
{
    wchar_t name[64];
    block_name(name, process);
    HANDLE mapping = OpenFileMappingW(FILE_MAP_READ, FALSE, name);
    if (!mapping) return nullptr;
    const auto* block = static_cast<const Block*>(MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, sizeof(Block)));
    if (!block || block->magic != block_magic || block->cells_x != cells_x || block->cells_y != cells_y) {
        if (block) UnmapViewOfFile(block);
        CloseHandle(mapping);
        return nullptr;
    }
    return block;
}

// Reader side: sets one request flag in the block of `process`. Creates the block if no presenter has
// yet (the presenter's create_block then keeps the request). False if the block cannot be made.
inline bool request(DWORD process, volatile LONG Block::*flag)
{
    wchar_t name[64];
    block_name(name, process);
    HANDLE mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(Block), name);
    if (!mapping) return false;
    auto* block = static_cast<Block*>(MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, sizeof(Block)));
    if (!block) return false;
    InterlockedExchange(&(block->*flag), 1);
    return true;
}
// Asks the presenter to sample at all (a script with a picture wait).
inline bool request_sampling(DWORD process) { return request(process, &Block::sampling_wanted); }
// Asks the presenter for the fine grid as well.
inline bool request_fine(DWORD process) { return request(process, &Block::fine_wanted); }

// `fine` may be null: the sample then has no fine grid.
inline void publish(Block* block, const std::uint8_t (&luma)[cells], LONG64 present, const std::uint8_t* fine = nullptr)
{
    InterlockedIncrement64(&block->sequence);  // odd: writing
    std::memcpy(block->luma, luma, sizeof(luma));
    if (fine) std::memcpy(block->fine, fine, sizeof(block->fine));
    block->fine_valid = fine ? 1 : 0;
    InterlockedExchange64(&block->sample_present, present);
    InterlockedIncrement64(&block->sequence);  // even: stable
}
inline void note_present(Block* block, LONG64 present) { InterlockedExchange64(&block->presents, present); }

// False while no sample is published or the writer keeps overtaking the copy.
inline bool read(const Block* block, Snapshot& out)
{
    for (int attempt = 0; attempt < 8; ++attempt) {
        // Plain volatile reads: a reader maps the block read-only, so no interlocked operation here.
        const LONG64 before = block->sequence;
        MemoryBarrier();
        if (before & 1) continue;
        std::memcpy(out.luma, block->luma, sizeof(out.luma));
        out.fine_valid = block->fine_valid != 0;
        if (out.fine_valid) std::memcpy(out.fine, block->fine, sizeof(out.fine));
        out.sample_present = static_cast<std::uint64_t>(block->sample_present);
        MemoryBarrier();
        const LONG64 after = block->sequence;
        if (after != before) continue;
        out.sequence = static_cast<std::uint64_t>(before / 2);
        out.presents = static_cast<std::uint64_t>(block->presents);
        return out.sequence != 0;
    }
    return false;
}

enum class Layout { rgba8, bgra8, rgb10a2 };

inline std::uint8_t luma(unsigned r, unsigned g, unsigned b) { return static_cast<std::uint8_t>((2 * r + 5 * g + b) / 8); }

// One sampled row of `width` 4-byte texels into `columns` mean-luma cells.
inline void reduce_row(const unsigned char* row, std::uint32_t width, Layout layout, std::uint8_t* out, int columns = cells_x)
{
    for (int cell = 0; cell < columns; ++cell) {
        std::uint32_t first = static_cast<std::uint32_t>(std::uint64_t{width} * cell / columns);
        std::uint32_t end = static_cast<std::uint32_t>(std::uint64_t{width} * (cell + 1) / columns);
        if (end <= first) end = first + 1 <= width ? first + 1 : width;
        if (first >= end) first = end ? end - 1 : 0;
        std::uint64_t sum = 0;
        for (std::uint32_t x = first; x < end; ++x) {
            const unsigned char* t = row + std::size_t{x} * 4;
            unsigned r, g, b;
            if (layout == Layout::rgb10a2) {
                std::uint32_t v;
                std::memcpy(&v, t, 4);
                r = (v & 0x3FF) >> 2;
                g = ((v >> 10) & 0x3FF) >> 2;
                b = ((v >> 20) & 0x3FF) >> 2;
            } else {
                r = layout == Layout::rgba8 ? t[0] : t[2];
                g = t[1];
                b = layout == Layout::rgba8 ? t[2] : t[0];
            }
            sum += luma(r, g, b);
        }
        out[cell] = static_cast<std::uint8_t>(end > first ? sum / (end - first) : 0);
    }
}

// The sampled rows: the middle row of each of `bands` equal bands.
inline std::uint32_t sampled_row(int band, std::uint32_t height, int bands = cells_y)
{
    return static_cast<std::uint32_t>((std::uint64_t{2} * band + 1) * height / (2 * bands));
}

}  // namespace x2w_present_signature
