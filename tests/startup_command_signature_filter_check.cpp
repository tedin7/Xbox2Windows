// Checks src/startup_stubs/command_signature_filter.h without a device (builds on any platform): which
// indirect argument descriptions device slot 41 recognises before calling desktop D3D12.
// usage: startup_command_signature_filter_check [--mutate off-by-one|first-only]
#include "startup_stubs/command_signature_filter.h"

#include <cstdio>
#include <cstring>

namespace {
// The shape of D3D12_INDIRECT_ARGUMENT_DESC the helper relies on: a Type member first.
struct Argument {
    unsigned Type;
    unsigned Payload[3];
};
constexpr unsigned dispatch_mesh = 10;
constexpr unsigned incrementing_constant = 11;

enum class Mutation { none, off_by_one, first_only };
Mutation mutation = Mutation::none;
int failures;

// The two wrong versions: a range check that excludes new desktop types, and a first-argument-only scan.
bool under_test(const Argument* arguments, std::size_t count)
{
    switch (mutation) {
    case Mutation::off_by_one:
        for (std::size_t i = 0; arguments && i < count; ++i)
            if (arguments[i].Type > dispatch_mesh) return true;
        return false;
    case Mutation::first_only: return x2w_startup::has_console_indirect_arg_type(arguments, count ? 1 : 0);
    default: return x2w_startup::has_console_indirect_arg_type(arguments, count);
    }
}

void expect(const char* name, const Argument* arguments, std::size_t count, bool expected)
{
    if (under_test(arguments, count) != expected) {
        std::printf("FAIL %s: expected %s\n", name, expected ? "the known console type" : "desktop validation");
        ++failures;
    }
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--mutate") == 0) {
        if (std::strcmp(argv[2], "off-by-one") == 0) mutation = Mutation::off_by_one;
        else if (std::strcmp(argv[2], "first-only") == 0) mutation = Mutation::first_only;
        else return 2;
    } else if (argc != 1) {
        return 2;
    }

    const Argument draw_indexed[] = {{2, {}}};
    expect("one desktop argument", draw_indexed, 1, false);
    const Argument mesh[] = {{dispatch_mesh, {}}};
    expect("the desktop mesh type", mesh, 1, false);
    const Argument increment[] = {{incrementing_constant, {}}};
    expect("the desktop incrementing-constant type reaches desktop validation", increment, 1, false);
    const Argument seen[] = {{0x69, {12, 0, 0}}};
    expect("the type seen in a title (0x69)", seen, 1, true);
    const Argument mixed[] = {{1, {}}, {0x69, {}}};
    expect("a console type after a desktop one", mixed, 2, true);
    const Argument all_desktop[] = {{0, {}}, {4, {}}, {9, {}}};
    expect("several desktop types", all_desktop, 3, false);
    const Argument wide[] = {{0xffffffffu, {}}};
    expect("an unrecognised type reaches desktop validation", wide, 1, false);
    expect("no arguments", draw_indexed, 0, false);
    expect("a null list", nullptr, 3, false);

    using x2w_startup::console_indirect_dispatch_record;
    using x2w_startup::indirect_ranges_overlap;
    static_assert(console_indirect_dispatch_record(12, 1));
    static_assert(!console_indirect_dispatch_record(16, 1));
    static_assert(!console_indirect_dispatch_record(12, 2));
    static_assert(!console_indirect_dispatch_record(12, 0));
    static_assert(indirect_ranges_overlap(8, 12, 16, 4));
    static_assert(indirect_ranges_overlap(16, 4, 8, 12));
    static_assert(!indirect_ranges_overlap(8, 12, 20, 4));
    static_assert(!indirect_ranges_overlap(8, 0, 8, 4));
    static_assert(indirect_ranges_overlap(~0ull - 3, 4, ~0ull - 1, 2));

    if (failures) return 1;
    std::printf("startup_command_signature_filter: PASS\n");
    return 0;
}
