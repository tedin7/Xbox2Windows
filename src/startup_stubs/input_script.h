// Experimental (docs/startup-runtime.md#input-script): parser and playback of X2W_INPUT_SCRIPT for the
// GameInput stand-in (src/startup_stubs/gameinput.cpp). Pure C++, no Windows calls, so the
// checks in tests/startup_gameinput_script.cpp can drive it directly.
//
// A script is a comma-separated list. Each non-empty item is one of:
//   <mask hex>@<s>[+<d>]          press these GameInput button bits at <s> for <d> s (default 0.3)
//   <axis>=<value>@<s>[+<d>]      hold one axis at <value>: lx ly rx ry in [-1, 1], lt rt in [0, 1]
//   base=poll | base=start        time base of every entry (first item only; default poll)
//   wait[=<settle>[/<timeout>[/stop|/go]]][@<s>]
//                                 block the timeline until the presented picture changes and then
//                                 stays unchanged for <settle> s (default 1); after <timeout> s
//                                 (default 120) log a timeout and go on (go, default) or stop
//                                 options after ';': cells=<n> level=<n> region=<x0>:<y0>:<x1>:<y1> fine
//   mark[=<label>][@<s>]          log the script time and the present number at <s>
//   group[=<label>] ... end       a wait that times out inside skips the rest of the group
// <s> is absolute seconds on the time base, or +<seconds> after the previous timed item (its
// start; for a wait, its completion). wait and mark without @ mean @+0. Items after a wait are
// shifted by the time the wait blocked and never start before it completes.
// Active button entries OR their masks; among active entries on one axis the later item wins.
// Items that cannot be accepted are kept as rejections (1-based item index and reason).
#pragma once

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

namespace x2w_input_script {

constexpr std::size_t max_entries = 256;  // as the stand-in kept before holds and axes (#1859)
constexpr double default_duration = 0.3;
// Same order as the floats of the stand-in's GamepadState after its buttons.
enum Axis : int { axis_left_trigger, axis_right_trigger, axis_left_x, axis_left_y, axis_right_x, axis_right_y, axis_count };
inline const char* const axis_names[axis_count] = {"lt", "rt", "lx", "ly", "rx", "ry"};

constexpr double default_settle = 1.0, default_timeout = 120.0;
// The picture signal grids (src/startup_stubs/present_signature.h): the default 32 x 18 and the
// fine 64 x 72 one a wait can ask for. Two grids differ when at least `cells` cells (default 16)
// of the watched region differ by more than `level` levels (default 12), so by default a blinking
// cursor or a small counter does not count as a new screen.
constexpr int grid_x = 32, grid_y = 18, signal_cells = grid_x * grid_y;
constexpr int fine_x = 64, fine_y = 72, fine_cells = fine_x * fine_y;
constexpr int cell_tolerance = 12, changed_cells = 16;

struct Region {
    double x0 = 0, y0 = 0, x1 = 1, y1 = 1;  // normalised; a cell counts when its centre is inside
};
struct Entry {
    int axis;  // -1 for a button entry
    std::uint32_t mask;
    float value;
    double at, duration;
    std::size_t segment;  // number of waits before this item
    std::size_t item = 0;  // item index
    int group = -1;
};
struct Wait {
    std::size_t index;  // item index
    double at, settle, timeout;
    bool stop;
    int cells = changed_cells, level = cell_tolerance;
    Region region;
    bool fine = false;
    int group = -1;
};
struct Mark {
    std::size_t index;
    double at;
    std::size_t segment;
    std::string label;
    int group = -1;
};
struct Group {
    std::size_t index, end_index;  // items "group" and "end"
    std::string label;
    double end_at;  // the script time at "end" (the start of its last timed item)
    bool closed;
};

inline bool in_region(const Region& r, int x, int y, int width, int height)
{
    const double cx = (x + 0.5) / width, cy = (y + 0.5) / height;
    return cx >= r.x0 && cx < r.x1 && cy >= r.y0 && cy < r.y1;
}
inline int region_cells(const Wait& wait)
{
    const int width = wait.fine ? fine_x : grid_x, height = wait.fine ? fine_y : grid_y;
    int n = 0;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) n += in_region(wait.region, x, y, width, height);
    return n;
}
struct Rejection {
    std::size_t index;  // 1-based position among the non-empty items
    std::string reason, text;
};
struct Script {
    bool from_process_start = false;
    std::size_t items = 0;  // non-empty items, accepted or not
    std::vector<Entry> entries;
    std::vector<Wait> waits;
    std::vector<Mark> marks;
    std::vector<Group> groups;
    std::vector<Rejection> rejections;
    bool wants_fine() const
    {
        for (const Wait& wait : waits)
            if (wait.fine) return true;
        return false;
    }
};
struct Output {
    std::uint32_t buttons = 0;
    float axes[axis_count] = {};
    bool active[axis_count] = {};
};

namespace detail {
inline bool blank(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

// "@<seconds>[+<duration>]" or "@+<seconds>[+<duration>]" (after `cursor`) up to the end of the item.
inline const char* parse_when(const char* p, double cursor, Entry& entry, bool* had_duration = nullptr)
{
    if (*p != '@') return "missing @<seconds>";
    char* stop = nullptr;
    const bool relative = p[1] == '+';
    p += relative ? 2 : 1;
    entry.at = std::strtod(p, &stop);
    if (stop == p || !std::isfinite(entry.at) || (relative && !(entry.at >= 0))) return "bad time";
    if (relative) entry.at += cursor;
    p = stop;
    entry.duration = default_duration;
    if (had_duration) *had_duration = *p == '+';
    if (*p == '+') {
        entry.duration = std::strtod(++p, &stop);
        if (stop == p || !std::isfinite(entry.duration) || !(entry.duration > 0)) return "bad duration";
        p = stop;
    }
    return *p ? "trailing text" : nullptr;
}

inline bool label_char(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
}

inline bool parse_int(const std::string& text, int low, int high, int& out)
{
    char* stop = nullptr;
    const long value = std::strtol(text.c_str(), &stop, 10);
    if (text.empty() || *stop || value < low || value > high) return false;
    out = static_cast<int>(value);
    return true;
}

// Keyed wait options after the first ';': cells=<n>, level=<n>, region=<x0>:<y0>:<x1>:<y1>, fine.
inline const char* parse_wait_keys(const std::string& text, Wait& wait)
{
    std::size_t from = 0;
    while (from <= text.size()) {
        const std::size_t semicolon = text.find(';', from);
        const std::string key = text.substr(from, semicolon == std::string::npos ? std::string::npos : semicolon - from);
        if (key == "fine") {
            wait.fine = true;
        } else if (key.compare(0, 6, "cells=") == 0) {
            if (!parse_int(key.substr(6), 1, fine_cells, wait.cells)) return "bad wait option";
        } else if (key.compare(0, 6, "level=") == 0) {
            if (!parse_int(key.substr(6), 0, 254, wait.level)) return "bad wait option";
        } else if (key.compare(0, 7, "region=") == 0) {
            double v[4];
            const char* p = key.c_str() + 7;
            for (int i = 0; i < 4; ++i) {
                char* stop = nullptr;
                v[i] = std::strtod(p, &stop);
                if (stop == p || !std::isfinite(v[i]) || v[i] < 0 || v[i] > 1 || *stop != (i < 3 ? ':' : '\0')) return "bad region";
                p = stop + (i < 3 ? 1 : 0);
            }
            if (!(v[0] < v[2] && v[1] < v[3])) return "bad region";
            wait.region = {v[0], v[1], v[2], v[3]};
        } else if (!key.empty()) {
            return "bad wait option";
        }
        if (semicolon == std::string::npos) break;
        from = semicolon + 1;
    }
    const int watched = region_cells(wait);
    if (watched == 0) return "empty region";
    if (wait.cells > watched) return "cells larger than the region";
    return nullptr;
}

// The option text after "wait=": <settle>[/<timeout>[/stop|/go]][;<key>...], empty fields keep their default.
inline const char* parse_wait_options(const std::string& all, Wait& wait)
{
    const std::size_t semicolon = all.find(';');
    const std::string text = all.substr(0, semicolon);
    std::string parts[3];
    std::size_t count = 0, from = 0;
    for (;;) {
        if (count == 3) return "bad wait option";
        const std::size_t slash = text.find('/', from);
        parts[count++] = text.substr(from, slash == std::string::npos ? std::string::npos : slash - from);
        if (slash == std::string::npos) break;
        from = slash + 1;
    }
    for (std::size_t i = 0; i < count && i < 2; ++i) {
        if (parts[i].empty()) continue;
        char* stop = nullptr;
        const double value = std::strtod(parts[i].c_str(), &stop);
        if (*stop || !std::isfinite(value) || value < 0 || (i == 1 && value == 0)) return "bad wait option";
        (i == 0 ? wait.settle : wait.timeout) = value;
    }
    if (count == 3) {
        if (parts[2] != "stop" && parts[2] != "go") return "bad wait option";
        wait.stop = parts[2] == "stop";
    }
    return semicolon == std::string::npos ? nullptr : parse_wait_keys(all.substr(semicolon + 1), wait);
}

// wait and mark items: `value` is the text after '=' (may be empty), `when` the text from '@'
// (empty means @+0). Adds the item to `script` and sets `at`.
inline const char* parse_event(const std::string& name, const std::string& value, const std::string& when, std::size_t index,
                               double cursor, Script& script, double& at)
{
    Entry timing{};
    bool had_duration = false;
    if (when.empty()) {
        timing.at = cursor;
    } else if (const char* reason = parse_when(when.c_str(), cursor, timing, &had_duration)) {
        return reason;
    }
    if (had_duration) return "duration not allowed on wait or mark";
    if (!script.waits.empty() && timing.at < script.waits.back().at) return "time before the preceding wait";
    if (name == "wait") {
        Wait wait{index, timing.at, default_settle, default_timeout, false};
        if (const char* reason = parse_wait_options(value, wait)) return reason;
        script.waits.push_back(wait);
    } else {
        const std::string label = value.empty() ? "mark" + std::to_string(index) : value;
        if (label.size() > 32) return "bad label";
        for (char c : label)
            if (!label_char(c)) return "bad label";
        script.marks.push_back({index, timing.at, script.waits.size(), label});
    }
    at = timing.at;
    return nullptr;
}

// One trimmed, non-empty item. Returns the rejection reason, or nullptr with `entry` filled, or
// `directive` set for base=, wait and mark (which add themselves to `script`; `event_at` is then
// the time of the wait or mark).
inline const char* parse_item(const std::string& item, std::size_t index, double cursor, Script& script, Entry& entry,
                              bool& directive, double& event_at)
{
    directive = false;
    event_at = -1;
    const bool first = index == 1;
    const std::size_t equals = item.find('='), at = item.find('@');
    for (const char* event : {"wait", "mark"}) {
        const std::size_t length = std::strlen(event);
        const bool keys = event[0] == 'w' && item.size() > length && item[length] == ';';  // wait;cells=2 (no positional part)
        if (item.compare(0, length, event) != 0 || (item.size() > length && item[length] != '=' && item[length] != '@' && !keys))
            continue;
        const std::size_t value_end = at == std::string::npos ? item.size() : at;
        const std::string value = keys ? item.substr(length, value_end - length)
                                  : item.size() > length && item[length] == '=' ? item.substr(length + 1, value_end - length - 1) : "";
        directive = true;
        return parse_event(event, value, at == std::string::npos ? "" : item.substr(at), index, cursor, script, event_at);
    }
    if (equals != std::string::npos && (at == std::string::npos || equals < at)) {
        std::string name = item.substr(0, equals);
        while (!name.empty() && blank(name.back())) name.pop_back();
        const char* rest = item.c_str() + equals + 1;
        if (name == "base") {
            std::string value = rest;
            while (!value.empty() && blank(value.front())) value.erase(0, 1);
            if (!first) return "base= must be the first item";
            if (value != "poll" && value != "start") return "unknown base (use poll or start)";
            script.from_process_start = value == "start";
            directive = true;
            return nullptr;
        }
        int axis = -1;
        for (int i = 0; i < axis_count; ++i)
            if (name == axis_names[i]) axis = i;
        if (axis < 0) return "unknown token";
        char* stop = nullptr;
        const double value = std::strtod(rest, &stop);
        if (stop == rest || !std::isfinite(value)) return "bad value";
        const bool trigger = axis == axis_left_trigger || axis == axis_right_trigger;
        if (value > 1 || value < (trigger ? 0 : -1)) return trigger ? "value out of range [0, 1]" : "value out of range [-1, 1]";
        entry.axis = axis;
        entry.mask = 0;
        entry.value = static_cast<float>(value);
        return parse_when(stop, cursor, entry);
    }
    // Button entry: strtoul and strtod accept what the earlier sscanf("%x@%lf") form accepted.
    const char* p = item.c_str();
    char* stop = nullptr;
    errno = 0;
    const unsigned long long mask = std::strtoull(p, &stop, 16);
    if (stop == p || errno == ERANGE || (mask > 0xFFFFFFFFull && mask < 0xFFFFFFFF00000000ull)) return "bad mask";
    entry.axis = -1;
    entry.mask = static_cast<std::uint32_t>(mask);
    entry.value = 0;
    return parse_when(stop, cursor, entry);
}
}  // namespace detail

inline Script parse(const char* text)
{
    Script script;
    if (!text) return script;
    std::size_t index = 0;
    double cursor = 0, group_floor = -HUGE_VAL;
    int group = -1, rejected_nested = 0;
    for (const char* p = text;;) {
        const char* end = p;
        while (*end && *end != ',') ++end;
        const char* begin = p;
        const char* last = end;
        while (begin < last && detail::blank(*begin)) ++begin;
        while (last > begin && detail::blank(last[-1])) --last;
        if (begin < last) {
            const std::string item(begin, last);
            ++index;
            script.items = index;
            // group[=<label>] ... end: not timed, no nesting.
            const bool is_group = item.compare(0, 5, "group") == 0 && (item.size() == 5 || item[5] == '=' || item[5] == '@');
            const bool is_end = item == "end" || item.compare(0, 4, "end@") == 0;
            if (is_group || is_end) {
                const char* reason = nullptr;
                if (item.find('@') != std::string::npos) {
                    reason = "group and end take no time";
                    if (is_group && group >= 0) ++rejected_nested;
                } else if (is_group && group >= 0) {
                    reason = "nested group";
                    ++rejected_nested;
                } else if (is_group) {
                    const std::string label = item.size() > 5 ? item.substr(6) : "group" + std::to_string(index);
                    bool good = !label.empty() && label.size() <= 32;
                    for (char c : label) good = good && detail::label_char(c);
                    if (good) {
                        script.groups.push_back({index, 0, label, cursor, false});
                        group = static_cast<int>(script.groups.size()) - 1;
                    } else {
                        reason = "bad label";
                    }
                } else if (rejected_nested) {
                    reason = "end of a rejected nested group";
                    --rejected_nested;
                } else if (group < 0) {
                    reason = "end without group";
                } else {
                    Group& g = script.groups[static_cast<std::size_t>(group)];
                    g.end_index = index;
                    g.end_at = cursor;
                    g.closed = true;
                    group_floor = cursor;
                    group = -1;
                }
                if (reason) script.rejections.push_back({index, reason, item});
                if (!*end) break;
                p = end + 1;
                continue;
            }
            Entry entry{};
            bool directive = false;
            double event_at = -1;
            const std::size_t waits = script.waits.size(), marks = script.marks.size();
            const std::size_t timed = script.entries.size() + waits + marks;
            const char* reason = detail::parse_item(item, index, cursor, script, entry, directive, event_at);
            if (!reason && !directive && waits && entry.at < script.waits.back().at) reason = "time before the preceding wait";
            const bool event = script.waits.size() != waits || script.marks.size() != marks;
            if (!reason && (!directive || event) && (event ? event_at : entry.at) < group_floor)
                reason = "time before the end of the preceding group";
            if (!reason && (!directive || event) && timed >= max_entries) reason = "over 256 entries";
            if (reason) {
                script.waits.resize(waits);
                script.marks.resize(marks);
            } else if (event) {
                if (script.waits.size() != waits) script.waits.back().group = group;
                if (script.marks.size() != marks) script.marks.back().group = group;
            }
            if (reason) {
                script.rejections.push_back({index, reason, item});
            } else if (!directive) {
                entry.segment = script.waits.size();
                entry.item = index;
                entry.group = group;
                script.entries.push_back(entry);
                cursor = entry.at;
            } else if (event) {
                cursor = event_at;
            }
        }
        if (!*end) break;
        p = end + 1;
    }
    // An unclosed group is rejected; its items stay, ungrouped.
    if (group >= 0) {
        const Group& g = script.groups[static_cast<std::size_t>(group)];
        Rejection r{g.index, "group not closed", g.label == "group" + std::to_string(g.index) ? "group" : "group=" + g.label};
        auto at = script.rejections.begin();
        while (at != script.rejections.end() && at->index < r.index) ++at;
        script.rejections.insert(at, r);
        for (Entry& e : script.entries) e.group = e.group == group ? -1 : e.group;
        for (Wait& w : script.waits) w.group = w.group == group ? -1 : w.group;
        for (Mark& m : script.marks) m.group = m.group == group ? -1 : m.group;
    }
    return script;
}

// State at `t` seconds on the script's time base.
inline Output play(const Script& script, double t)
{
    Output out;
    for (const Entry& entry : script.entries) {
        if (!(t >= entry.at && t < entry.at + entry.duration)) continue;
        if (entry.axis < 0) {
            out.buttons |= entry.mask;
        } else {
            out.axes[entry.axis] = entry.value;
            out.active[entry.axis] = true;
        }
    }
    return out;
}

inline double first_start(const Script& script)
{
    double first = script.entries.empty() ? 0 : script.entries.front().at;
    for (const Entry& entry : script.entries) first = entry.at < first ? entry.at : first;
    return first;
}
inline double last_end(const Script& script)
{
    double last = 0;
    for (const Entry& entry : script.entries) last = entry.at + entry.duration > last ? entry.at + entry.duration : last;
    return last;
}

// ---- waits and marks ------------------------------------------------------------------------------
// The picture signal: grids of mean luma (src/startup_stubs/present_signature.h) with a sequence
// number that grows with every published sample.
struct Signal {
    bool available = false;  // a presenter has published at least one sample
    std::uint64_t sequence = 0, present = 0;
    std::uint8_t cells[signal_cells] = {};
    bool fine_available = false;  // this sample carries the fine grid
    std::uint8_t fine[fine_cells] = {};
};
// Cells of the wait's grid and region that differ by more than the wait's level.
inline int wait_distance(const Wait& wait, const std::uint8_t* a, const std::uint8_t* b)
{
    const int width = wait.fine ? fine_x : grid_x, height = wait.fine ? fine_y : grid_y;
    int differing = 0;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            const int i = y * width + x;
            if (in_region(wait.region, x, y, width, height)) differing += (a[i] > b[i] ? a[i] - b[i] : b[i] - a[i]) > wait.level;
        }
    return differing;
}
inline int signal_distance(const std::uint8_t* a, const std::uint8_t* b)
{
    int differing = 0;
    for (int i = 0; i < signal_cells; ++i) differing += (a[i] > b[i] ? a[i] - b[i] : b[i] - a[i]) > cell_tolerance;
    return differing;
}

struct Event {
    enum Type { wait_begun, wait_changed, wait_done, wait_timeout, mark, group_skipped } type;
    std::size_t item;      // item index
    double at;             // seconds on the time base
    double elapsed;        // waits: seconds since the wait began
    bool signal;           // a presenter signal was available
    bool stopped;          // wait_timeout: the script stops
    std::uint64_t present; // mark: present number, 0 without a presenter
    std::string label;     // mark; wait_timeout and group_skipped: the group skipped (empty if none)
    std::size_t group_item = 0, end_item = 0;  // group_skipped: the group's "group" and "end" items
};

// Scripts with waits or marks. Real time `now` is seconds on the script's time base; a wait that
// began at its scheduled time b and completed at c shifts every later item by c - b.
class Timeline {
public:
    explicit Timeline(const Script& script)
        : script_(script), shift_(script.waits.size() + 1, 0.0), marked_(script.marks.size(), false), failed_(script.groups.size(), 0)
    {
    }

    void advance(double now, const Signal& signal, std::vector<Event>& events)
    {
        while (!stopped_ && done_ < script_.waits.size()) {
            const Wait& wait = script_.waits[done_];
            const double begin = wait.at + shift_[done_];
            // A fine wait reads the fine grid, which a sample carries once the presenter has seen the request.
            const bool available = wait.fine ? signal.fine_available : signal.available;
            const std::uint8_t* cells = wait.fine ? signal.fine : signal.cells;
            const std::size_t count = wait.fine ? fine_cells : signal_cells;
            if (!waiting_) {
                if (now < begin) break;
                waiting_ = true;
                changed_ = false;
                have_reference_ = available;
                if (have_reference_) take(cells, count, signal.sequence);
                events.push_back({Event::wait_begun, wait.index, begin, 0, available, false, 0, {}});
            }
            if (!have_reference_ && available) {
                have_reference_ = true;
                take(cells, count, signal.sequence);
            } else if (have_reference_ && available && signal.sequence != sequence_) {
                sequence_ = signal.sequence;
                if (!changed_ && wait_distance(wait, reference_, cells) >= wait.cells) {
                    changed_ = true;
                    stable_since_ = now;
                    std::copy(cells, cells + count, last_);
                    events.push_back({Event::wait_changed, wait.index, now, now - begin, true, false, 0, {}});
                } else if (changed_ && wait_distance(wait, last_, cells) >= wait.cells) {
                    stable_since_ = now;
                    std::copy(cells, cells + count, last_);
                }
            }
            double completed = 0;
            if (changed_ && now - stable_since_ >= wait.settle) {
                completed = now;
                events.push_back({Event::wait_done, wait.index, now, now - begin, true, false, 0, {}});
            } else if (now - begin >= wait.timeout) {
                completed = begin + wait.timeout;
                const bool skip = wait.group >= 0 && !wait.stop;
                const Group* g = skip ? &script_.groups[static_cast<std::size_t>(wait.group)] : nullptr;
                events.push_back({Event::wait_timeout, wait.index, now, now - begin, have_reference_, wait.stop, 0, g ? g->label : std::string()});
                if (wait.stop) {
                    stopped_ = true;
                    waiting_ = false;
                    break;
                }
                if (g) {
                    // The rest of the group is skipped: its later waits resolve at once, and the script
                    // time jumps from this wait to the group's end, so the first item after "end" is timed
                    // from this timeout.
                    failed_[static_cast<std::size_t>(wait.group)] = wait.index;
                    std::size_t last = done_;
                    while (last + 1 < script_.waits.size() && script_.waits[last + 1].group == wait.group) ++last;
                    const double shift = shift_[done_] + (completed - begin) - (g->end_at - wait.at);
                    for (std::size_t k = done_ + 1; k <= last + 1; ++k) shift_[k] = shift;
                    events.push_back({Event::group_skipped, wait.index, completed, now - begin, have_reference_, false, 0, g->label,
                                      g->index, g->end_index});
                    done_ = last + 1;
                    waiting_ = false;
                    continue;
                }
            } else {
                break;
            }
            shift_[done_ + 1] = shift_[done_] + (completed - begin);
            ++done_;
            waiting_ = false;
        }
        for (std::size_t i = 0; i < script_.marks.size(); ++i) {
            const Mark& mark = script_.marks[i];
            if (marked_[i] || mark.segment > done_ || skipped(mark.group, mark.index)) continue;
            const double at = mark.at + shift_[mark.segment];
            if (now < at) continue;
            marked_[i] = true;
            events.push_back({Event::mark, mark.index, at, 0, signal.available, false, signal.available ? signal.present : 0, mark.label});
        }
    }

    // Same rule as play(), with each item moved by the waits before it; items behind an
    // unfinished or stopping wait are not active.
    Output play(double now) const
    {
        Output out;
        for (const Entry& entry : script_.entries) {
            if (entry.segment > done_ || skipped(entry.group, entry.item)) continue;
            const double at = entry.at + shift_[entry.segment];
            if (!(now >= at && now < at + entry.duration)) continue;
            if (entry.axis < 0) {
                out.buttons |= entry.mask;
            } else {
                out.axes[entry.axis] = entry.value;
                out.active[entry.axis] = true;
            }
        }
        return out;
    }

    bool stopped() const { return stopped_; }
    bool waiting() const { return waiting_; }
    std::size_t completed_waits() const { return done_; }
    // End of the last entry once every wait has resolved; negative before that.
    double end() const
    {
        if (!stopped_ && done_ < script_.waits.size()) return -1;
        double last = 0;
        for (const Entry& entry : script_.entries)
            if (entry.segment <= done_ && !skipped(entry.group, entry.item))
                last = std::fmax(last, entry.at + shift_[entry.segment] + entry.duration);
        return last;
    }

private:
    void take(const std::uint8_t* cells, std::size_t count, std::uint64_t sequence)
    {
        std::copy(cells, cells + count, reference_);
        sequence_ = sequence;
    }
    // Items of a group after the step whose wait timed out are skipped.
    bool skipped(int group, std::size_t item) const
    {
        return group >= 0 && failed_[static_cast<std::size_t>(group)] && item > failed_[static_cast<std::size_t>(group)];
    }

    const Script& script_;
    std::vector<double> shift_;
    std::vector<bool> marked_;
    std::vector<std::size_t> failed_;  // per group: item index of the wait that timed out (0 = none)
    std::size_t done_ = 0;
    bool stopped_ = false, waiting_ = false, changed_ = false, have_reference_ = false;
    double stable_since_ = 0;
    std::uint64_t sequence_ = 0;
    std::uint8_t reference_[fine_cells] = {}, last_[fine_cells] = {};
};

}  // namespace x2w_input_script
