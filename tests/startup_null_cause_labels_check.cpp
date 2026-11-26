// Checks src/startup_stubs/null_binding_cause.h without a device: the slot of the x2w-render totals
// line each cause is counted in, and the label printed for that slot. The device indexes its counts
// by the cause's value and prints slots 0 to 4 as entry, buffer, token, wrapper and dimension.
#include "startup_stubs/draw_cause.h"
#include "startup_stubs/null_binding_cause.h"

#include <cstdio>
#include <cstring>

namespace {
using x2w_local::BindingFacts;
int failures, cases;

const char* const labels[] = {"entry", "buffer", "token", "wrapper", "dimension", "ring_full"};

// Facts as the texture binding builds them: from a table, with a readable entry carrying a known
// view token for a resource wrapper of the shader's dimension.
BindingFacts texture()
{
    BindingFacts f;
    f.texture = f.from_table = f.entry_readable = f.token_magic = f.token_known = f.resource_wrapper = true;
    return f;
}

// Facts as the buffer binding builds them for a table entry the title wrote itself (no view token).
BindingFacts table_buffer(bool bound)
{
    BindingFacts f;
    f.from_table = f.entry_readable = true;
    f.bound = bound;
    return f;
}

int slot_of(const BindingFacts& f)
{
    return static_cast<int>(x2w_local::classify_null_binding(f));
}

void counted(const char* name, const BindingFacts& f, int slot)
{
    ++cases;
    const int got = slot_of(f);
    if (got != slot) {
        std::printf("FAIL %s: slot %d, expected %d (%s)\n", name, got, slot, labels[slot]);
        ++failures;
    } else if (std::strcmp(x2w_local::null_cause_names[got], labels[slot]) != 0) {
        std::printf("FAIL %s: slot %d named \"%s\", expected \"%s\"\n", name, got, x2w_local::null_cause_names[got],
                    labels[slot]);
        ++failures;
    }
}

void same_text(const char* name, const char* got, const char* expected)
{
    ++cases;
    if (std::strcmp(got, expected) != 0) {
        std::printf("FAIL %s:\n  got      %s\n  expected %s\n", name, got, expected);
        ++failures;
    }
}
}  // namespace

int main(int argc, char** argv)
{
    // --mutate buffer-token: the expectations of a totals line that prints the unknown-token count
    // under "buffer" and the buffer count under "token" (a negative control: the check must fail).
    const bool swapped = argc > 2 && std::strcmp(argv[1], "--mutate") == 0 && std::strcmp(argv[2], "buffer-token") == 0;
    if (argc > 1 && !swapped) {
        std::printf("unknown mutation\n");
        return 2;
    }
    const int entry = 0, buffer = swapped ? 2 : 1, token = swapped ? 1 : 2, wrapper = 3, dimension = 4, ring_full = 5;

    if (x2w_local::null_cause_count != 6) {
        std::printf("FAIL: %d causes, expected 6\n", x2w_local::null_cause_count);
        return 1;
    }
    for (int i = 0; i < 6; ++i) {
        ++cases;
        if (std::strcmp(x2w_local::null_cause_names[i], labels[i]) != 0) {
            std::printf("FAIL: slot %d named \"%s\", expected \"%s\"\n", i, x2w_local::null_cause_names[i], labels[i]);
            ++failures;
        }
    }

    // One binding per cause, each counted in its own slot.
    BindingFacts f = texture();
    f.entry_readable = f.token_magic = f.token_known = f.resource_wrapper = false;  // an unreadable entry reads as zeros
    counted("texture with an unreadable table entry", f, entry);
    counted("table buffer without a readable range", table_buffer(false), buffer);
    f = BindingFacts{};
    counted("vertex buffer without a readable range", f, buffer);
    f = texture();
    f.token_known = f.resource_wrapper = false;
    counted("texture whose view id the wrapper never issued", f, token);
    f = table_buffer(false);
    f.token_magic = true;
    counted("table buffer whose view id the wrapper never issued", f, token);
    f = texture();
    f.resource_wrapper = false;
    counted("texture view without a resource wrapper", f, wrapper);
    f = texture();
    f.dimension_matches = false;
    counted("texture view of another dimension", f, dimension);
    f = table_buffer(false);
    f.ring_failed = true;
    counted("table buffer whose copy found no upload ring space", f, ring_full);
    f = table_buffer(true);
    f.ring_failed = true;
    counted("a ring failure outranks a bound flag only for buffers", f, ring_full);

    // A frame of bindings in which each cause occurs a different power of two times, so a count
    // printed under another label changes the line. Valid bindings add nothing.
    struct Binding {
        BindingFacts facts;
        int times;
    };
    BindingFacts unreadable = texture(), unknown = texture(), no_wrapper = texture(), mismatch = texture();
    unreadable.entry_readable = unreadable.token_magic = unreadable.token_known = unreadable.resource_wrapper = false;
    unknown.token_known = unknown.resource_wrapper = false;
    no_wrapper.resource_wrapper = false;
    mismatch.dimension_matches = false;
    BindingFacts ring_buffer = table_buffer(false);
    ring_buffer.ring_failed = true;
    const Binding frame[] = {
        {texture(), 3}, {table_buffer(true), 2}, {unreadable, 1}, {table_buffer(false), 2},
        {unknown, 4},   {no_wrapper, 8},         {mismatch, 16}, {ring_buffer, 32},
    };
    long total = 0, by_slot[6] = {};
    for (const Binding& b : frame) {
        const int slot = slot_of(b.facts);
        if (slot == static_cast<int>(x2w_local::NullCause::none)) continue;
        if (slot < 0 || slot >= 6) {
            std::printf("FAIL: a null binding in slot %d, outside the six printed\n", slot);
            return 1;
        }
        total += b.times;
        by_slot[slot] += b.times;
    }
    // The device's format string, with the slots in the order it passes them.
    char line[200];
    std::snprintf(line, sizeof(line), "null bindings %ld (entry %ld, buffer %ld, token %ld, wrapper %ld, dimension %ld, ring_full %ld)",
                  total, by_slot[0], by_slot[1], by_slot[2], by_slot[3], by_slot[4], by_slot[5]);
    const char* const expected = swapped ? "null bindings 63 (entry 1, buffer 4, token 2, wrapper 8, dimension 16, ring_full 32)"
                                         : "null bindings 63 (entry 1, buffer 2, token 4, wrapper 8, dimension 16, ring_full 32)";
    same_text("totals line", line, expected);
    // The same counts labelled from the header's names.
    const char* const* n = x2w_local::null_cause_names;
    std::snprintf(line, sizeof(line), "null bindings %ld (%s %ld, %s %ld, %s %ld, %s %ld, %s %ld, %s %ld)", total, n[0], by_slot[0],
                  n[1], by_slot[1], n[2], by_slot[2], n[3], by_slot[3], n[4], by_slot[4], n[5], by_slot[5]);
    same_text("totals line from the header's names", line, expected);

    using namespace x2w_startup;
    same_text("pending draw", not_drawn_names[translation_pending_site], "translation_pending");
    same_text("refused draw", not_drawn_names[translation_refused_site], "translation_refused");
    same_text("failed load", not_drawn_names[translation_failed_to_load_site], "translation_failed_to_load");
    same_text("unproven depth view", not_drawn_names[d16_view_not_ready_site], "d16_view_not_ready");
    same_text("unready formatted buffer", not_drawn_names[formatted_buffer_not_ready_site], "formatted_buffer_not_ready");

    if (failures) {
        std::printf("startup_null_cause_labels: %d of %d FAILED\n", failures, cases);
        return 1;
    }
    std::printf("startup_null_cause_labels: PASS (%d cases)\n", cases);
    return 0;
}
