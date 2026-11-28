#include <cstdint>
#include <cstdio>

using Word = std::uint64_t;
extern "C" Word x2w_replay_invoke(void* target, Word rcx, const Word* args);

Word receive(Word result, Word a0, Word a1, Word a2, Word a3, Word a4, Word a5,
             Word a6, Word a7, Word a8, Word a9, Word a10)
{
    const Word received[] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10};
    for (Word i = 0; i < 11; ++i)
        if (received[i] != (0xfedcba9800000000ull | i)) return i + 1;
    *reinterpret_cast<Word*>(result) = a10;
    return result;
}

int main()
{
    Word args[11]{};
    for (Word i = 0; i < 11; ++i) args[i] = 0xfedcba9800000000ull | i;
    Word output = 0;
    const Word rcx = reinterpret_cast<Word>(&output);
    const Word result = x2w_replay_invoke(reinterpret_cast<void*>(&receive), rcx, args);
    if (result == rcx && output == args[10]) return 0;
    std::fprintf(stderr, "replay bridge: return 0x%llx, expected 0x%llx; output 0x%llx, expected 0x%llx\n",
                 result, rcx, output, args[10]);
    return 1;
}
