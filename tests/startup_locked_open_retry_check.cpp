#include "startup_stubs/locked_open_retry.h"

#include <cstdio>
#include <string_view>

int main(int argc, char** argv)
{
    const bool mutate = argc == 3 && std::string_view(argv[1]) == "--mutate" &&
                        std::string_view(argv[2]) == "no-retry";
    if (argc != 1 && !mutate) return 2;

    constexpr int max_tries = 20;
    constexpr int delay_ms = 10;
    const struct {
        const char* name;
        unsigned initial_code;
        int locked_calls;
        unsigned final_code;
        int expected_tries;
    } cases[] = {
        {"sharing violation clears", 32, 3, 0, 4},
        {"lock violation clears", 33, 3, 0, 4},
        {"opens immediately", 32, 0, 0, 1},
        {"permanent sharing violation", 32, max_tries, 32, max_tries},
        {"file not found", 2, 0, 2, 1},
        {"access denied", 5, 0, 5, 1},
        {"access denied after contention", 33, 1, 5, 2},
    };

    bool passed = true;
    for (const auto& test : cases) {
        int opens = 0;
        int sleeps = 0;
        int logs = 0;
        bool delays_match = true;
        bool log_matches = true;
        const bool succeeded = x2w_startup::open_with_lock_retry(
            [&] { return ++opens <= test.locked_calls ? test.initial_code : test.final_code; },
            [&](int delay) {
                ++sleeps;
                delays_match = delays_match && delay == delay_ms;
            },
            [&](int tries, unsigned code, bool opened) {
                ++logs;
                log_matches = tries == test.expected_tries && code == test.final_code &&
                              opened == (test.final_code == 0);
            },
            mutate ? 1 : max_tries, delay_ms);
        if (succeeded != (test.final_code == 0) || opens != test.expected_tries ||
            sleeps != test.expected_tries - 1 || logs != (test.expected_tries > 1 ? 1 : 0) ||
            !delays_match || !log_matches) {
            std::fprintf(stderr, "%s: opened=%d, opens=%d, sleeps=%d, logs=%d\n",
                         test.name, succeeded, opens, sleeps, logs);
            passed = false;
        }
    }
    return passed ? 0 : 1;
}
