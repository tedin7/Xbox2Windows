#pragma once

#include <charconv>
#include <string_view>

namespace x2w {
// Bump whenever a native translation of an unchanged request can change meaning.
constexpr unsigned native_compute_output_revision = 1;

enum class NativeComputeOutput { unmarked, current, incompatible };

inline NativeComputeOutput native_compute_output(std::string_view manifest)
{
    auto output = NativeComputeOutput::unmarked;
    while (!manifest.empty()) {
        const auto end = manifest.find('\n');
        auto line = manifest.substr(0, end);
        manifest.remove_prefix(end == std::string_view::npos ? manifest.size() : end + 1);
        if (!line.empty() && line.back() == '\r')
            line.remove_suffix(1);
        if (line != "native" && line.substr(0, 7) != "native ")
            continue;
        if (output != NativeComputeOutput::unmarked || line.size() <= 7)
            return NativeComputeOutput::incompatible;
        unsigned revision = 0;
        const auto last = line.data() + line.size();
        const auto parsed = std::from_chars(line.data() + 7, last, revision);
        if (parsed.ec != std::errc{} || parsed.ptr != last || revision != native_compute_output_revision)
            return NativeComputeOutput::incompatible;
        output = NativeComputeOutput::current;
    }
    return output;
}
}
