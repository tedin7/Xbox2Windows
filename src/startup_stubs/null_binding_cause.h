#pragma once

namespace x2w_local {
// Why a translated draw bound a null descriptor. Each null binding has exactly one cause, so the
// per-cause counts add up to the null_bindings total.
enum class NullCause : int {
    none = -1,
    table_entry,         // table entry is null, empty or unreadable
    buffer_unreadable,   // buffer address or size missing or unreadable, or the frame copy failed
    unknown_token,       // descriptor memory names a view the wrapper does not know
    texture_wrapper,     // the view's resource is missing or is not a resource wrapper
    dimension_mismatch,  // the view dimension does not match the shader
    ring_full,           // the guest memory was readable but the frame copy found no upload ring space (the draw is refused)
    integer_view_filtered,  // a typed manifest has no integer load binding for this texture
};
constexpr int null_cause_count = 7;
constexpr const char* null_cause_names[null_cause_count] = {"entry", "buffer", "token", "wrapper", "dimension", "ring_full", "integer_view_filtered"};

struct BindingFacts {
    bool texture = false;
    bool from_table = false;          // the descriptor comes from a descriptor table entry
    bool entry_readable = false;      // that entry is present and readable
    bool token_magic = false;         // its first word marks a wrapper-made view
    bool token_known = false;         // and the view id is one the wrapper issued
    bool resource_wrapper = false;    // texture: the view resolves to a resource wrapper
    bool dimension_matches = true;    // texture: the view dimension suits the shader
    bool bound = false;               // buffer: a readable range was bound
    bool ring_failed = false;         // buffer: the range is readable but the upload ring had no space for its copy
};

inline NullCause classify_null_binding(const BindingFacts& f)
{
    if (!f.texture && f.ring_failed) return NullCause::ring_full;
    // A buffer that ended up bound (also through the identity fallback) is not a null binding.
    if (!f.texture && f.bound) return NullCause::none;
    if (f.from_table && !f.entry_readable) return NullCause::table_entry;
    if (f.from_table && f.token_magic && !f.token_known) return NullCause::unknown_token;
    if (f.texture) {
        if (!f.token_magic) return NullCause::unknown_token;
        if (!f.resource_wrapper) return NullCause::texture_wrapper;
        return f.dimension_matches ? NullCause::none : NullCause::dimension_mismatch;
    }
    return f.bound ? NullCause::none : NullCause::buffer_unreadable;
}
}  // namespace x2w_local
