#pragma once

namespace x2w_startup {

inline constexpr unsigned sharing_violation = 32;
inline constexpr unsigned lock_violation = 33;

template <typename Open, typename Sleep, typename Log>
bool open_with_lock_retry(Open open, Sleep sleep, Log log, int max_tries = 20, int delay_ms = 10)
{
    for (int attempt = 0; attempt < max_tries; ++attempt) {
        const int tries = attempt + 1;
        const unsigned code = open();
        const bool locked = code == sharing_violation || code == lock_violation;
        if (!locked || tries == max_tries) {
            if (tries > 1) log(tries, code, code == 0);
            return code == 0;
        }
        sleep(delay_ms);
    }
    return false;
}

}  // namespace x2w_startup
