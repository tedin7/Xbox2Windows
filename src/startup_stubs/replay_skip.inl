// Replay bisect switches (opt-in, only meant for the replay player; both are off when unset).
//   X2W_REPLAY_SKIP_KEYS=<key,key,...>  draws and dispatches of these shader keys are refused (not executed).
//       A token is lowercase hex; a token shorter than 16 digits matches the start of the 16-digit key.
//   X2W_REPLAY_SKIP_CALLS=<from-to>     draws and dispatches whose recorded call sequence number is in the range
//       (inclusive; a single number means from-to equal). The player publishes the current call with
//       x2w_d3d12_replay_skip_call before each call.
// Each skipped key is logged once. The totals go to replay_summary.json ("skipped").
// Included by d3d12_xs_device.cpp after pass_stats.inl, before the draw and dispatch code.

namespace replay_skip {
struct Config {
    std::vector<std::string> key_prefixes;
    bool have_calls = false;
    unsigned long long first_call = 0, last_call = 0;
    bool any() const { return !key_prefixes.empty() || have_calls; }
};

inline Config parse(const char* keys, const char* calls)
{
    Config c;
    for (const char* p = keys ? keys : ""; *p;) {
        const char* end = p;
        while (*end && *end != ',') ++end;
        std::string token;
        for (const char* q = p; q < end; ++q) {
            const char ch = static_cast<char>(std::tolower(static_cast<unsigned char>(*q)));
            if (ch == ' ' || ch == 'x') continue;
            token += ch;
        }
        if (token.size() > 16) token.clear();
        for (char ch : token)
            if (!std::isxdigit(static_cast<unsigned char>(ch))) { token.clear(); break; }
        if (!token.empty()) c.key_prefixes.push_back(token);
        p = *end ? end + 1 : end;
    }
    if (calls && *calls) {
        char* rest = nullptr;
        const unsigned long long from = std::strtoull(calls, &rest, 10);
        unsigned long long to = from;
        if (rest != calls) {
            if (*rest == '-') {
                char* rest2 = nullptr;
                const unsigned long long t = std::strtoull(rest + 1, &rest2, 10);
                if (rest2 != rest + 1) to = t;
            }
            if (to >= from) c.have_calls = true, c.first_call = from, c.last_call = to;
        }
    }
    return c;
}

inline bool key_listed(const Config& c, unsigned long long key)
{
    char text[24];
    std::snprintf(text, sizeof(text), "%016llx", key);
    for (const auto& prefix : c.key_prefixes)
        if (std::strncmp(text, prefix.c_str(), prefix.size()) == 0) return true;
    return false;
}
}  // namespace replay_skip

volatile LONG64 replay_skip_call = -1;  // x2w_d3d12_replay_skip_call; -1 outside a replay
LONG skipped_draws, skipped_dispatches;

const replay_skip::Config& replay_skip_config()
{
    static const replay_skip::Config config = [] {
        char keys[2048] = {}, calls[64] = {};
        GetEnvironmentVariableA("X2W_REPLAY_SKIP_KEYS", keys, sizeof(keys) - 1);
        GetEnvironmentVariableA("X2W_REPLAY_SKIP_CALLS", calls, sizeof(calls) - 1);
        return replay_skip::parse(keys, calls);
    }();
    return config;
}

// True when this draw (dispatch = false) or dispatch must not run. `key` may be 0 when the pipeline has none yet.
bool replay_skip_this(u64 key, bool dispatch)
{
    const auto& config = replay_skip_config();
    if (!config.any()) return false;
    const LONG64 call = InterlockedCompareExchange64(&replay_skip_call, 0, 0);
    const bool by_call = config.have_calls && call >= 0 && static_cast<unsigned long long>(call) >= config.first_call &&
                         static_cast<unsigned long long>(call) <= config.last_call;
    const bool by_key = key && replay_skip::key_listed(config, key);
    if (!by_call && !by_key) return false;
    const LONG n = InterlockedIncrement(dispatch ? &skipped_dispatches : &skipped_draws);
    static SRWLOCK lock = SRWLOCK_INIT;
    static std::vector<u64> logged;
    bool first = false;
    AcquireSRWLockExclusive(&lock);
    if (by_key && std::find(logged.begin(), logged.end(), key) == logged.end() && logged.size() < 256) {
        logged.push_back(key);
        first = true;
    }
    ReleaseSRWLockExclusive(&lock);
    if (first)
        log("x2w-replay skip: %s of shader %016llx refused by X2W_REPLAY_SKIP_KEYS (first at call %lld)",
            dispatch ? "dispatch" : "draw", key, static_cast<long long>(call));
    else if (by_call && n <= 3)
        log("x2w-replay skip: %s at call %lld refused by X2W_REPLAY_SKIP_CALLS", dispatch ? "dispatch" : "draw",
            static_cast<long long>(call));
    return true;
}

extern "C" __declspec(dllexport) void x2w_d3d12_replay_skip_call(unsigned long long call)
{
    InterlockedExchange64(&replay_skip_call, static_cast<LONG64>(call));
}

extern "C" __declspec(dllexport) void x2w_d3d12_replay_skip_totals(long long* draws, long long* dispatches)
{
    if (draws) *draws = InterlockedCompareExchange(&skipped_draws, 0, 0);
    if (dispatches) *dispatches = InterlockedCompareExchange(&skipped_dispatches, 0, 0);
}
