#pragma once
#include <algorithm>
#include <bitset>
#include <cctype>
#include <climits>
#include <deque>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace x2w {
// LLVM's DXIL dialect. Resource declarations, handle annotations and entry metadata
// are produced together so the bytecode and the runtime binding contract agree.
class DxilModule {
  public:
    using Text = std::string;
    struct Resource {
        unsigned cls, reg, kind, component, width, bytes;
        unsigned count = 1;
        bool comparison = false;
    };
    std::ostringstream body, setup, globals;
    bool uses_wave = false;
    bool heap_used = false;
    unsigned features = 0;
    struct Signature {
        Text name;
        unsigned semantic_index, kind, component, row, first, width, interpolation, used = 0;
    };
    // The final bit marks rasterizer inputs without a packed signature row.
    // They vary between lanes but occupy no slot in the ViewID matrix.
    using InputMask = std::bitset<129>;
    bool track_inputs = false;
    bool loop_dependencies = false;
    std::map<Text, InputMask> dependencies;
    std::map<Text, std::vector<Text>> consumers;
    std::map<unsigned, std::vector<Text>> output_values;
    std::vector<Signature> inputs, outputs;
    std::map<unsigned, InputMask> output_dependencies;
    Text value(const Text &instruction) {
        auto name = "%t" + std::to_string(sequence++);
        if (track_inputs) {
            InputMask sources;
            for (size_t at = 0; (at = instruction.find('%', at)) != Text::npos;) {
                size_t end = at + 1;
                while (end < instruction.size() &&
                       (std::isalnum(static_cast<unsigned char>(instruction[end])) || instruction[end] == '_'))
                    ++end;
                auto operand = instruction.substr(at, end - at);
                if (loop_dependencies &&
                    (operand.rfind("%t", 0) == 0 || operand.rfind("%p", 0) == 0 || operand.rfind("%indexed", 0) == 0))
                    consumers[operand].push_back(name);
                auto found = dependencies.find(operand);
                if (found != dependencies.end())
                    sources |= found->second;
                at = end;
            }
            if (sources.any())
                dependencies.emplace(name, sources);
        }
        body << "  " << name << " = " << instruction << "\n";
        return name;
    }
    void finish_loop_dependencies() {
        std::deque<Text> pending;
        for (const auto &[name, mask] : dependencies)
            if (mask.any())
                pending.push_back(name);
        while (!pending.empty()) {
            auto source = std::move(pending.front());
            pending.pop_front();
            auto it = consumers.find(source);
            if (it == consumers.end())
                continue;
            const auto mask = dependencies.at(source);
            for (const auto &user : it->second) {
                auto &dest = dependencies[user];
                if ((mask & ~dest).none())
                    continue;
                dest |= mask;
                pending.push_back(user);
            }
        }
        for (const auto &[channel, values] : output_values)
            for (const auto &value : values)
                output_dependencies[channel] |= dependencies[value];
    }
    Text binary(const Text &op, const Text &a, const Text &b) { return value(op + " i32 " + a + ", " + b); }
    Text invoke(const Text &type, const Text &name, unsigned opcode, const std::vector<std::pair<Text, Text>> &args,
                const Text &attr) {
        Text decl = "declare " + type + " @dx.op." + name + "(i32",
             call = "call " + type + " @dx.op." + name + "(i32 " + std::to_string(opcode);
        for (auto &[t, v] : args) {
            decl += ", " + t;
            call += ", " + t + " " + v;
        }
        decl += ") " + attr;
        call += ")";
        auto [it, ok] = declarations.emplace(name, decl);
        if (!ok && it->second != decl)
            throw std::logic_error("inconsistent DXIL declaration");
        if (type != "void")
            return value(call);
        body << "  " << call << "\n";
        return {};
    }
    Text handle(Resource r, const Text &offset = {}) {
        const Text reg = std::to_string(r.reg), cls = std::to_string(r.cls);
        const auto found = std::find_if(resources.begin(), resources.end(), [&](const Resource &other) {
            return other.cls == r.cls && other.reg == r.reg;
        });
        Text h = "%h" + cls + "_" + reg;
        if (found != resources.end()) {
            if (found->kind != r.kind || found->width != r.width || found->component != r.component ||
                found->bytes != r.bytes || found->count != r.count || found->comparison != r.comparison)
                throw std::logic_error("incompatible DXIL binding declarations");
        } else
            resources.push_back(r);
        if (offset.empty() && !static_handles.insert(h).second)
            return h;
        Text index = reg;
        auto &out = offset.empty() ? setup : body;
        if (!offset.empty()) {
            index = r.reg ? binary("add", reg, offset) : offset;
            h = "%indexed" + std::to_string(sequence++);
            if (track_inputs) {
                if (auto it = dependencies.find(index); it != dependencies.end())
                    dependencies[h] = it->second;
                if (loop_dependencies)
                    consumers[index].push_back(h);
            }
        }
        const unsigned props = r.cls == 2 ? r.bytes : r.component + r.width * 256;
        out << "  " << h << "b = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, %dx.types.ResBind { i32 "
            << reg << ", i32 " << r.reg + r.count - 1 << ", i32 0, i8 " << cls << " }, i32 " << index << ", i1 "
            << (offset.empty() ? "false" : "true") << ")\n"
            << "  " << h << " = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle " << h
            << "b, %dx.types.ResourceProperties { i32 " << (r.kind + (r.cls == 1 ? 4096 : 0) + (r.comparison ? 32768 : 0)) << ", i32 " << props
            << " })\n";
        return h;
    }
    // A handle taken from the descriptor heap at a run-time index (SM 6.6 ResourceDescriptorHeap[index]), annotated with
    // the view class the shader reads: `kind` is the DXIL resource kind (11 raw buffer, 2 Texture2D) and `props` the
    // second ResourceProperties word. No resource record is declared: the binding is the whole heap.
    Text heap_handle(const Text &index, unsigned kind, unsigned props, bool uav = false) {
        heap_used = true;
        const Text h = "%heap" + std::to_string(sequence++);
        body << "  " << h << "b = call %dx.types.Handle @dx.op.createHandleFromHeap(i32 218, i32 " << index
             << ", i1 false, i1 true)\n"
             << "  " << h << " = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle " << h
             << "b, %dx.types.ResourceProperties { i32 " << (kind + (uav ? 4096 : 0)) << ", i32 " << props << " })\n";
        return h;
    }
    // A sampler taken from the directly indexed sampler heap at a constant index (SM 6.6 SamplerDescriptorHeap[index]).
    // The handle is annotated as a sampler (resource kind 14). The caller sets the module flag 0x80000000.
    Text sampler_heap_handle(const Text &index) {
        heap_used = true;
        const Text h = "%heap" + std::to_string(sequence++);
        body << "  " << h << "b = call %dx.types.Handle @dx.op.createHandleFromHeap(i32 218, i32 " << index
             << ", i1 true, i1 false)\n"
             << "  " << h << " = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle " << h
             << "b, %dx.types.ResourceProperties { i32 14, i32 0 })\n";
        return h;
    }
    Text finish_compute(unsigned x, unsigned y, unsigned z, const Text &stage = "cs") {
        std::ostringstream out;
        out << "target datalayout = "
               "\"e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64\"\n"
               "target triple = \"dxil-ms-dx\"\n"
               "%dx.types.Handle = type { i8* }\n%dx.types.ResBind = type { i32, i32, i32, i8 }\n"
               "%dx.types.ResourceProperties = type { i32, i32 }\n%dx.types.ResRet.i32 = type "
               "{i32,i32,i32,i32,i32}\n"
               "%dx.types.ResRet.f32 = type {float,float,float,float,i32}\n%dx.types.CBufRet.i32 = type "
               "{i32,i32,i32,i32}\n"
               "%dx.types.fouri32 = type {i32,i32,i32,i32}\n"
               "%dx.types.Dimensions = type {i32,i32,i32,i32}\n";
        std::vector<Text> metadata;
        auto node = [&](const Text &content) {
            auto id = "!" + std::to_string(metadata.size());
            metadata.push_back(content);
            return id;
        };
        auto list = [&](const std::vector<Text> &ids) {
            Text content;
            for (auto &id : ids) {
                if (!content.empty())
                    content += ", ";
                content += id;
            }
            return ids.empty() ? Text("null") : node(content);
        };
        std::vector<Text> groups[4];
        std::sort(resources.begin(), resources.end(),
                  [](const auto &a, const auto &b) { return std::pair{a.cls, a.reg} < std::pair{b.cls, b.reg}; });
        for (auto &r : resources) {
            const Text type = "%res" + std::to_string(r.cls) + "_" + std::to_string(r.reg);
            Text scalar = r.component == 9 ? "float" : "i32";
            Text member = r.cls == 2    ? "[" + std::to_string(r.bytes / 16) + " x <4 x i32>]"
                          : r.width > 1 ? "<" + std::to_string(r.width) + " x " + scalar + ">"
                                        : scalar;
            out << type << " = type { " << member << " }\n";
            const Text symbol = r.count == 1 ? type : "[" + std::to_string(r.count) + " x " + type + "]";
            Text md = "i32 " + std::to_string(groups[r.cls].size()) + ", " + symbol + "* undef, !\"\", i32 0, i32 " +
                      std::to_string(r.reg) + ", i32 " + std::to_string(r.count) + ", i32 " +
                      std::to_string(r.cls == 2   ? r.bytes
                                     : r.cls == 3 ? unsigned(r.comparison)
                                                  : r.kind);
            if (r.cls == 0)
                md += ", i32 0";
            if (r.cls == 1)
                md += ", i1 false, i1 false, i1 false";
            md += ", " + (r.component ? node("i32 0, i32 " + std::to_string(r.component)) : Text("null"));
            groups[r.cls].push_back(node(md));
        }
        Text res = "null";
        if (!resources.empty())
            res = node(list(groups[0]) + ", " + list(groups[1]) + ", " + list(groups[2]) + ", " + list(groups[3]));
        // AMD's DXIL compiler (amdxc64.dll) dereferences Module::getNamedMetadata("llvm.ident") without a null check.
        const auto ident = node("!\"x2w\""), version = node("i32 1, i32 6"),
                   model = node("!\"" + stage + "\", i32 6, i32 6");
        const unsigned flags = features | (uses_wave ? 524288 : 0);
        Text properties = "null", signature = "null", view_state;
        if (stage == "cs") {
            const auto size =
                node("i32 " + std::to_string(x) + ", i32 " + std::to_string(y) + ", i32 " + std::to_string(z));
            const auto wave = node("i32 32");
            properties = node((flags ? "i32 0, i64 " + std::to_string(flags) + ", " : "") + "i32 4, " + size +
                              ", i32 11, " + wave);
        } else {
            if (flags)
                properties = node("i32 0, i64 " + std::to_string(flags));
            if (stage == "gs")
                properties = node("i32 1, " + node("i32 3, i32 6, i32 1, i32 5, i32 1"));
            auto signature_list = [&](const std::vector<Signature> &rows, unsigned &scalars) {
                std::vector<Text> ids;
                for (const auto &r : rows) {
                    if (r.name.empty() ||
                        r.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") !=
                            Text::npos ||
                        (r.row >= 32 && r.row != UINT_MAX) || !r.width || r.first + r.width > 4)
                        throw std::runtime_error("invalid graphics signature");
                    if (r.row != UINT_MAX)
                        scalars = (std::max)(scalars, r.row * 4 + r.first + r.width);
                    auto index = node("i32 " + std::to_string(r.semantic_index));
                    auto used = r.used ? node("i32 3, i32 " + std::to_string(r.used)) : Text("null");
                    ids.push_back(node("i32 " + std::to_string(ids.size()) + ", !\"" + r.name + "\", i8 " +
                                       std::to_string(r.component) + ", i8 " + std::to_string(r.kind) + ", " + index +
                                       ", i8 " + std::to_string(r.interpolation) + ", i32 1, i8 " +
                                       std::to_string(r.width) + ", i32 " +
                                       (r.row == UINT_MAX ? "-1" : std::to_string(r.row)) + ", i8 " +
                                       (r.row == UINT_MAX ? "-1" : std::to_string(r.first)) + ", " + used));
                }
                return list(ids);
            };
            unsigned in_count = 0, out_count = 0;
            const auto in_sig = signature_list(inputs, in_count), out_sig = signature_list(outputs, out_count);
            signature = node(in_sig + ", " + out_sig + ", null");
            std::vector<unsigned> matrix{in_count, out_count};
            unsigned stride = (out_count + 31) / 32;
            matrix.resize(2 + in_count * stride);
            for (const auto &[output, mask] : output_dependencies)
                for (unsigned i = 0; i < in_count; ++i)
                    if (mask.test(i))
                        matrix.at(2 + i * stride + output / 32) |= 1u << (output % 32);
            if (stage == "gs")
                matrix.resize(matrix.size() + 3); // no outputs on streams 1 to 3
            Text values;
            for (auto word : matrix) {
                if (!values.empty())
                    values += ", ";
                values += "i32 " + std::to_string(word);
            }
            auto id = "!" + std::to_string(metadata.size());
            metadata.push_back("[" + std::to_string(matrix.size()) + " x i32] [" + values + "]");
            view_state = "!dx.viewIdState = !{" + id + "}\n";
        }
        const auto entry = node("void ()* @main, !\"main\", " + signature + ", " + res + ", " + properties);
        out << view_state;
        out << globals.str() << "define void @main() {\nentry:\n" << setup.str() << body.str() << "}\n";
        for (auto &[name, decl] : declarations)
            out << decl << "\n";
        if (heap_used) {
            out << "declare %dx.types.Handle @dx.op.createHandleFromHeap(i32,i32,i1,i1) nounwind readnone\n";
            if (resources.empty())
                out << "declare %dx.types.Handle "
                       "@dx.op.annotateHandle(i32,%dx.types.Handle,%dx.types.ResourceProperties) nounwind readnone\n";
        }
        if (!resources.empty())
            out << "declare %dx.types.Handle @dx.op.createHandleFromBinding(i32,%dx.types.ResBind,i32,i1) nounwind "
                   "readnone\n"
                   "declare %dx.types.Handle "
                   "@dx.op.annotateHandle(i32,%dx.types.Handle,%dx.types.ResourceProperties) nounwind readnone\n";
        out << "!llvm.ident = !{" << ident << "}\n!dx.version = !{" << version << "}\n!dx.valver = !{" << version
            << "}\n!dx.shaderModel = !{" << model << "}\n";
        if (res != "null")
            out << "!dx.resources = !{" << res << "}\n";
        out << "!dx.entryPoints = !{" << entry << "}\n";
        for (size_t i = 0; i < metadata.size(); ++i)
            out << "!" << i << " = !{" << metadata[i] << "}\n";
        return out.str();
    }

  private:
    unsigned sequence = 0;
    std::vector<Resource> resources;
    std::set<Text> static_handles;
    std::map<Text, Text> declarations;
};
} // namespace x2w
