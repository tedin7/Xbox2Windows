#include "dxil_module.h"
#include "gfx1013_compute_native.h"

namespace x2w {
// Apply the startup rectangle rule (indexed_rect_geometry.h) to post-VS
// values. IA, base vertex, instance IDs and the original VS stay intact.
std::string native_rectangle_geometry(const std::vector<GraphicsLinkage> &outputs,
                                      const std::vector<GraphicsLinkage> &pixel_inputs) {
    using std::string;
    using std::to_string;
    if (std::any_of(pixel_inputs.begin(), pixel_inputs.end(), [](const auto &input) { return input.kind == 28; }))
        throw UnsupportedComputeShader("numeric barycentrics on an expanded rectangle");
    struct Channel {
        unsigned id, column, scalar;
        bool floating, flat;
        std::array<string, 3> vertex;
        string a, b, fourth;
    };
    DxilModule ir;
    std::vector<Channel> channels;
    std::array<std::array<string, 4>, 3> position;
    DxilModule::InputMask guard_inputs;
    bool has_position = false;
    for (const auto &r : outputs) {
        const bool pos = r.kind == 3;
        if (r.row >= 32 || !r.width || r.first + r.width > 4 || (!pos && r.kind != 0 && r.kind != 6 && r.kind != 7))
            throw UnsupportedComputeShader("unsupported rectangle stage interface");
        if (pos && (has_position || r.component != 9 || r.first || r.used != 15 || r.width != 4))
            throw UnsupportedComputeShader("rectangle requires a complete float4 position");
        has_position |= pos;
        auto consumer = std::find_if(pixel_inputs.begin(), pixel_inputs.end(),
                                     [&](const auto &p) { return p.name == r.name && p.index == r.index; });
        const bool flat =
            !pos && (r.component != 9 || (consumer != pixel_inputs.end() && consumer->interpolation == 1));
        const unsigned id = static_cast<unsigned>(ir.inputs.size());
        ir.inputs.push_back({r.name, r.index, r.kind, r.component, r.row, r.first, r.width, r.interpolation, r.used});
        ir.outputs.push_back(ir.inputs.back());
        if (flat)
            ir.outputs.back().interpolation = 1;
        for (unsigned col = 0; col < r.width; ++col) {
            if (!(r.used & (1u << col)))
                continue;
            Channel c{id, col, r.row * 4 + r.first + col, r.component == 9, flat};
            for (unsigned v = 0; v < (flat ? 1u : 3u); ++v) {
                c.vertex[v] =
                    ir.invoke(c.floating ? "float" : "i32", c.floating ? "loadInput.f32" : "loadInput.i32", 4,
                              {{"i32", to_string(id)}, {"i32", "0"}, {"i8", to_string(col)}, {"i32", to_string(v)}},
                              "nounwind readnone");
                if (pos)
                    position[v][col] = c.vertex[v];
            }
            if (pos)
                guard_inputs.set(c.scalar);
            channels.push_back(std::move(c));
        }
    }
    if (!has_position)
        throw UnsupportedComputeShader("rectangle requires a position");
    auto both = [&](const string &a, const string &b) { return ir.value("and i1 " + a + ", " + b); };
    auto either = [&](const string &a, const string &b) { return ir.value("or i1 " + a + ", " + b); };
    auto compare = [&](const char *relation, const string &a, const string &b) {
        return ir.value("fcmp " + string(relation) + " float " + a + ", " + b);
    };
    auto finite = [&](const string &v) {
        return ir.invoke("i1", "isSpecialFloat.f32", 10, {{"float", v}}, "nounwind readnone");
    };
    string valid = compare("ogt", position[0][3], "0.000000e+00");
    for (unsigned i = 0; i < 3; ++i) {
        for (const auto &v : position[i])
            valid = both(valid, finite(v));
        if (i)
            valid = both(valid, compare("oeq", position[0][3], position[i][3]));
    }
    std::array<string, 3> corner;
    for (unsigned i = 0; i < 3; ++i) {
        auto &p = position[i], &a = position[(i + 1) % 3], &b = position[(i + 2) % 3];
        auto x = compare("oeq", p[0], a[0]), y = compare("oeq", p[1], b[1]);
        auto other_x = compare("oeq", p[0], b[0]), other_y = compare("oeq", p[1], a[1]);
        corner[i] = both(either(both(x, y), both(other_x, other_y)),
                         both(compare("one", a[0], b[0]), compare("one", a[1], b[1])));
    }
    // A rotated rectangle has no corner that shares an axis with a neighbour, so the axis test above misses it
    // and the whole rectangle would be dropped (#3131). The right-angle corner is the vertex opposite the
    // rectangle's diagonal, the longest of its three edges: for corner c with neighbours a and b, |a - b| is the
    // diagonal and is longer than both sides. The sides must also meet at a right angle, so a shape that is not a
    // rectangle stays undrawn. The axis corners keep precedence, as they did before.
    auto squared = [&](const string &p, const string &q) {
        auto d = ir.value("fsub float " + p + ", " + q);
        return ir.value("fmul float " + d + ", " + d);
    };
    const auto distance2 = [&](const std::array<string, 4> &p, const std::array<string, 4> &q) {
        auto dx = squared(p[0], q[0]), dy = squared(p[1], q[1]);
        return ir.value("fadd float " + dx + ", " + dy);
    };
    std::array<string, 3> geometric;
    for (unsigned i = 0; i < 3; ++i) {
        auto &p = position[i], &a = position[(i + 1) % 3], &b = position[(i + 2) % 3];
        auto across = distance2(a, b), to_a = distance2(p, a), to_b = distance2(p, b);
        // (p - a) . (p - b), squared, against the binary32 value of 1e-6 of |p - a|^2 |p - b|^2: the two
        // sides meet at a right angle. The hex double is the only spelling LLVM IR accepts for this
        // non-representable decimal constant.
        auto diff = [&](const string &u, const string &v) { return ir.value("fsub float " + u + ", " + v); };
        auto times = [&](const string &u, const string &v) { return ir.value("fmul float " + u + ", " + v); };
        auto dot = ir.value("fadd float " + times(diff(p[0], a[0]), diff(p[0], b[0])) + ", " +
                            times(diff(p[1], a[1]), diff(p[1], b[1])));
        auto limit = ir.value("fmul float " + times(to_a, to_b) + ", 0x3EB0C6F7A0000000");
        geometric[i] = both(both(compare("ogt", across, to_a), compare("ogt", across, to_b)),
                            compare("ole", times(dot, dot), limit));
    }
    auto any_axis = either(corner[0], either(corner[1], corner[2]));
    for (unsigned i = 0; i < 3; ++i)
        corner[i] = either(corner[i], both(ir.value("xor i1 " + any_axis + ", true"), geometric[i]));
    valid = both(valid, either(corner[0], either(corner[1], corner[2])));
    // Only a non-degenerate rectangle has a unique matching corner.
    for (auto &c : channels) {
        if (c.flat)
            continue;
        auto rotate = [&](unsigned n) {
            auto rest = ir.value("select i1 " + corner[1] + ", float " + c.vertex[(n + 1) % 3] + ", float " +
                                 c.vertex[(n + 2) % 3]);
            return ir.value("select i1 " + corner[0] + ", float " + c.vertex[n] + ", float " + rest);
        };
        c.a = rotate(1);
        c.b = rotate(2);
        auto sum = ir.value("fadd float " + c.a + ", " + c.b);
        c.fourth = ir.value("fsub float " + sum + ", " + rotate(0));
        if (ir.outputs[c.id].kind == 3)
            valid = both(valid, finite(c.fourth));
    }
    ir.body << "  br i1 " << valid << ", label %draw, label %done\ndraw:\n";
    for (unsigned v = 0; v < 6; ++v) {
        for (const auto &c : channels) {
            const auto &value = c.flat ? c.vertex[0] : v < 3 ? c.vertex[v] : v == 3 ? c.a : v == 4 ? c.fourth : c.b;
            ir.invoke("void", c.floating ? "storeOutput.f32" : "storeOutput.i32", 5,
                      {{"i32", to_string(c.id)},
                       {"i32", "0"},
                       {"i8", to_string(c.column)},
                       {c.floating ? "float" : "i32", value}},
                      "nounwind");
            ir.output_dependencies[c.scalar] = guard_inputs;
            ir.output_dependencies[c.scalar].set(c.scalar);
        }
        ir.invoke("void", "emitStream", 97, {{"i8", "0"}}, "nounwind");
        if (v == 2 || v == 5)
            ir.invoke("void", "cutStream", 98, {{"i8", "0"}}, "nounwind");
    }
    ir.body << "  br label %done\ndone:\n  ret void\n";
    return ir.finish_compute(0, 0, 0, "gs");
}
} // namespace x2w
