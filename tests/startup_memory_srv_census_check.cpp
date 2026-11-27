// The limits of src/startup_stubs/memory_srv_census.h without a device: detail flags for the first 8
// distinct refused shapes, the first 4 pipelines and the first 8 written entries, shapes only for
// description-dependent reasons, the overflow counters, and a periodic report that is skipped when
// nothing changed.
#include "startup_stubs/memory_srv_as.h"
#include "startup_stubs/memory_srv_census.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(const char* what, bool ok)
{
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}
D3D12_RESOURCE_DESC texture(UINT64 width, DXGI_FORMAT format, UINT16 mips)
{
    return {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, 64, 1, mips, format, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN,
            D3D12_RESOURCE_FLAG_NONE};
}
template <class Census> std::vector<std::string> lines(Census& census, bool only_if_changed)
{
    std::vector<std::string> out;
    census.report([&](const char* line) { out.emplace_back(line); }, only_if_changed);
    return out;
}
}  // namespace

int main()
{
    using namespace x2w_startup;
    {
        // The cause texture_unavailable names (d3d12_memory_srv.inl, texture_unavailable_reason). A lookup that never tried to
        // publish is a missing resource whatever the size; a layout over the 64 MiB single-copy limit is named by size and
        // never touches its pages; within the limit unreadable texels and an unexplained failure stay apart.
        bool touched = false;
        const auto readable = [&](bool value) { return [&touched, value] { touched = true; return value; }; };
        const std::uint64_t limit = max_published_texture_bytes;
        check("limit is 64 MiB", limit == 64ull << 20);
        check("no publication: no resource", !std::strcmp(texture_unavailable_cause(false, 0, readable(true)), "texture_no_resource"));
        check("no publication, huge layout: still no resource", !std::strcmp(texture_unavailable_cause(false, limit + 1, readable(false)), "texture_no_resource"));
        touched = false;
        check("over the limit: named by size", !std::strcmp(texture_unavailable_cause(true, limit + 1, readable(true)), "texture_over_64mib"));
        check("over the limit: the pages are not touched", !touched);
        check("over the limit and unreadable: still size", !std::strcmp(texture_unavailable_cause(true, limit + 1, readable(false)), "texture_over_64mib"));
        check("exactly the limit is within it", !std::strcmp(texture_unavailable_cause(true, limit, readable(false)), "texture_unreadable"));
        check("within the limit, readable: other failure", !std::strcmp(texture_unavailable_cause(true, 4096, readable(true)), "texture_unavailable"));
        check("within the limit, unreadable texels", !std::strcmp(texture_unavailable_cause(true, 4096, readable(false)), "texture_unreadable"));
        for (const char* name : {"texture_no_resource", "texture_unreadable", "texture_over_64mib"})
            check("each cause has its own census reason", memory_srv_reason_index(name) < memory_srv_reason_count);
    }
    MemorySrvCensus srv;
    const auto a = texture(64, DXGI_FORMAT_BC1_UNORM, 7);
    check("first shape is detailed", srv.note("unsupported_description", &a));
    check("repeat is not", !srv.note("unsupported_description", &a));
    check("same shape, other reason is a new shape", srv.note("texture_unavailable", &a));
    check("reasons without a shape keep none", !srv.note("null_address", &a) && !srv.note("null_destination", &a));
    check("unreadable description keeps no shape", !srv.note("unsupported_description", nullptr));
    // Every name the acceleration-structure resolver can return is a census reason of its own (not "other")
    // and, because that path does not read the resource description, keeps no shape. The names come from
    // as_srv_refusal_name, so a fourth refusal that is not added to memory_srv_reasons fails here.
    {
        int as_names = 0;
        MemorySrvCensus probe;
        for (int r = static_cast<int>(AsSrvRefusal::none) + 1;; ++r) {
            const char* name = as_srv_refusal_name(static_cast<AsSrvRefusal>(r));
            if (!std::strcmp(name, "none")) break;
            ++as_names;
            const int index = memory_srv_reason_index(name);
            check("acceleration-structure refusal has its own census reason", index < memory_srv_reason_count);
            check("acceleration-structure refusal keeps no shape", !memory_srv_reason_has_shape(index) && !probe.note(name, &a));
        }
        check("the acceleration-structure refusals were enumerated", as_names >= 3);
        MemorySrvCensus fresh;
        fresh.note("as_unaligned", &a);
        const auto fresh_report = lines(fresh, false);
        check("as_unaligned is reported by name", fresh_report.size() == 1 && fresh_report[0].find(" as_unaligned 1,") != std::string::npos &&
              fresh_report[0].find(" other 0;") != std::string::npos);
    }
    const auto b = texture(128, DXGI_FORMAT_BC7_UNORM, 8), c = texture(256, DXGI_FORMAT_R8G8B8A8_UNORM, 9),
               d = texture(512, DXGI_FORMAT_BC3_UNORM, 10);
    check("third shape is detailed", srv.note("buffer_unavailable", &b));
    check("fourth shape is detailed", srv.note("unsupported_description", &c));
    check("fifth shape is detailed", srv.note("unsupported_description", &d));
    check("unknown reason counts as other", !srv.note("acme_reason", &d));
    for (UINT64 w = 1; w <= 192; ++w) {
        const auto e = texture(1000 + w, DXGI_FORMAT_R8_UNORM, 1);
        const bool detailed = srv.note("unsupported_description", &e);
        if (w <= 5) check("fourth to eighth shapes of one reason are detailed", detailed);
        if (w == 6) check("ninth shape of one reason is not", !detailed);
    }
    // A reason that filled its share cannot hide a reason that shows up later.
    const auto late = texture(777, DXGI_FORMAT_R16_UNORM, 3);
    check("a late reason still keeps its shape", srv.note("gpu_written_resource", &late));
    auto report = lines(srv, true);
    check("report has a summary and a line per kept shape", report.size() == 1 + 3 + MemorySrvCensus::max_shapes_per_reason);
    check("summary", !report.empty() && report[0] ==
          "x2w-d3d12 memory SRV refusals (slot 70): total 203; null_destination 1, unreadable_resource 0, unreadable_view 0, "
          "null_address 1, unsupported_description 197, destination_outside_view_heaps 0, gpu_written_resource 1, "
          "as_unaligned 0, as_not_placed 0, as_wrong_state 0, buffer_unavailable 1, texture_unavailable 1, texture_no_resource 0, texture_unreadable 0, texture_over_64mib 0, partial_tile_xg_rows 0, partial_tile_beyond_xg_size 0, unsupported_desktop_format 0, "
          "console_layout_xg_rows 0, console_layout_without_footprints 0, console_layout_beyond_xg_size 0, overlapping_resource 0, "
          "other 1; distinct shapes 19 (calls past the shape table 179)");
    check("first shape line", report.size() > 1 && report[1] ==
          "x2w-d3d12 memory SRV refused shape 1: reason=unsupported_description dim=3 fmt=71 size=64x64x1 mips=7 layout=0x0 "
          "flags=0x0 calls=2");
    check("unchanged periodic report is skipped", lines(srv, true).empty());
    srv.note("unsupported_description", &a);
    const auto changed = lines(srv, true);
    check("periodic report lists only the shapes counted since", changed.size() == 2 && changed[1].find("shape 1:") != std::string::npos);
    check("a forced report lists every shape", lines(srv, false).size() == report.size());
    check("forced report still prints", lines(srv, false).size() == report.size());
    srv.note("null_address", nullptr);
    check("forced report does not hide the next change", lines(srv, false).size() == report.size() && lines(srv, true).size() == 1);

    TokenNullCensus tokens;
    check("first pipeline is detailed", tokens.note(0x1000000000000001ull));
    check("its repeat is not", !tokens.note(0x1000000000000001ull));
    check("second pipeline is detailed", tokens.note(2));
    check("third pipeline is detailed", tokens.note(3));
    for (std::uint64_t p = 10; p < 10 + TokenNullCensus::max_pipelines; ++p) {
        const bool detailed = tokens.note(p);
        if (p == 10) check("fourth pipeline is detailed", detailed);
        if (p == 11) check("fifth pipeline is not", !detailed);
    }
    const auto token_report = lines(tokens, true);
    check("token report", token_report.size() == 1 && token_report[0].rfind(
          "x2w-draw token texture nulls: total 20 in 16 pipelines; 1000000000000001=2 0000000000000002=1 0000000000000003=1 ", 0) == 0 &&
          token_report[0].size() > 10 && token_report[0].compare(token_report[0].size() - 8, 8, " other=3") == 0);
    check("unchanged token report is skipped", lines(tokens, true).empty());

    MemorySrvWrites writes;
    check("first entry is the first", writes.note(0x1000) == 1);
    check("a rewrite is not new", writes.note(0x1000) == 0);
    for (std::uint64_t e = 2; e <= 8; ++e) check("entries 2 to 8 are numbered", writes.note(0x1000 + 0x40 * e) == e);
    check("ninth entry is only counted", writes.note(0x9000) == 0);
    const auto write_report = lines(writes, true);
    check("write report", write_report.size() == 1 &&
          write_report[0] == "x2w-d3d12 memory SRV writes (slot 70): calls 10, distinct entries 9 (untracked 0)");
    check("unchanged write report is skipped", lines(writes, true).empty());

    char line[256];
    const std::uint32_t words[8] = {0, 0, 0, 0, 0, 0, 0, 0x12345678};
    report_token_entry(line, sizeof(line), 0x1000000000000001ull, 1, 0x7ff000, words);
    check("entry line", !std::strcmp(line, "x2w-draw token texture null: pipeline 1000000000000001 srv 1 entry 0x7ff000 words "
                                           "00000000 00000000 00000000 00000000 00000000 00000000 00000000 12345678"));
    std::printf("memory SRV census: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
