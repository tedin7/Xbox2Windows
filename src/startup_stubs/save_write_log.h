// Experimental (docs/startup-runtime.md): the log lines for the game-save stand-in's writes, so a run shows whether
// the title's saves (and its privacy acceptance) reach the disk. The first 20 writes are logged in full; later ones
// only when they fail (up to 20 failures) and as a totals line every 25 writes. No Windows types: tested on its own.
#pragma once

#include <cstdio>
#include <string>

namespace x2w_save {
class WriteLog {
public:
    static constexpr long full_lines = 20;
    static constexpr long totals_every = 25;

    // Counts one finished commit and returns the text to log ("" when nothing is due; lines are separated by '\n').
    std::string record(const std::string& container, unsigned long actions, unsigned long long bytes, bool ok, int status,
                       unsigned long windows_error)
    {
        ++writes_;
        if (ok) written_ += bytes;
        else ++failures_;
        std::string text;
        char line[320];
        if (writes_ <= full_lines || (!ok && failures_ <= full_lines)) {
            std::snprintf(line, sizeof line,
                          "x2w-runtime game save write #%ld: container \"%.100s\", %lu actions, %llu bytes -> %s (status %d, Windows error %lu)",
                          writes_, container.c_str(), actions, bytes, ok ? "ok" : "FAILED", status, windows_error);
            text += line;
        }
        if (writes_ % totals_every == 0) {
            std::snprintf(line, sizeof line, "x2w-runtime game save totals: %ld writes, %ld failed, %llu bytes written", writes_,
                          failures_, written_);
            if (!text.empty()) text += '\n';
            text += line;
        }
        return text;
    }

private:
    long writes_ = 0, failures_ = 0;
    unsigned long long written_ = 0;
};
}  // namespace x2w_save
