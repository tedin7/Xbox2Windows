#pragma once

#include "native_buffer_stride.h"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <d3d12.h>
#include <sstream>
#include <string>
#include <vector>

// The view type of a global (descriptor heap) SRV slot that a dispatch can select from GPU data.
//
// A shader that picks its slot at run time can only be recorded and replayed if the type of every slot it may pick is
// known: the capture keeps {slot, type, stride} for each slot of the heap, and the dispatch is refused, before any
// shader runs, when the selected slot holds another type than the one the shader reads. No heap-indexed data read is
// translated here; this header only classifies, serializes and checks.
//
// A .views file describes the one dispatch it was captured from. It is not part of the .cdx/.root content key of the
// request it sits next to.
namespace x2w_startup {
enum class GlobalSrvType : unsigned { none, raw_buffer, structured_buffer, typed_buffer, texture };

inline const char *global_srv_type_name(GlobalSrvType type) {
    switch (type) {
    case GlobalSrvType::raw_buffer: return "raw_buffer";
    case GlobalSrvType::structured_buffer: return "structured_buffer";
    case GlobalSrvType::typed_buffer: return "typed_buffer";
    case GlobalSrvType::texture: return "texture";
    default: return "none";
    }
}

inline GlobalSrvType global_srv_type_from_name(const std::string &name) {
    for (auto type : {GlobalSrvType::raw_buffer, GlobalSrvType::structured_buffer, GlobalSrvType::typed_buffer,
                      GlobalSrvType::texture})
        if (name == global_srv_type_name(type)) return type;
    return GlobalSrvType::none;
}

struct GlobalSrvView {
    unsigned slot = 0;
    GlobalSrvType type = GlobalSrvType::none;
    unsigned stride = 0; // Bytes per element for a buffer view (the value native_view_stride gives), 0 for a texture.
    std::uint64_t bytes = 0; // Size of the buffer in bytes, 0 for a texture. Set by the caller from the resource description.
};

// Classifies the description a title passed for an SRV (slot 18 of the device interface). False: not a view this
// path can name (an unknown dimension, or a buffer view without a usable stride).
inline bool classify_global_srv_view(unsigned heap_slot, const void *bytes, std::size_t size, GlobalSrvView &view) {
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    if (!bytes || size < sizeof(desc)) return false;
    std::memcpy(&desc, bytes, sizeof(desc));
    view = {};
    view.slot = heap_slot;
    if (desc.ViewDimension == D3D12_SRV_DIMENSION_UNKNOWN) return false;
    if (desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER) {
        view.type = GlobalSrvType::texture;
        return true;
    }
    if (!native_view_stride(18, bytes, size, view.stride)) return false;
    view.type = desc.Buffer.StructureByteStride ? GlobalSrvType::structured_buffer
                : (desc.Buffer.Flags & D3D12_BUFFER_SRV_FLAG_RAW) ? GlobalSrvType::raw_buffer
                                                                   : GlobalSrvType::typed_buffer;
    return true;
}

// One request's view metadata, the text that travels with a captured request (req_<key>.views).
struct GlobalSrvRequest {
    unsigned selected = 0;
    GlobalSrvType needed = GlobalSrvType::none;
    std::vector<GlobalSrvView> views;
    bool has_selected = false;
};

inline std::string encode_global_srv_request(const GlobalSrvRequest &request) {
    std::ostringstream out;
    out << "x2w-global-srv-views 2\nselect " << request.selected << "\nneed " << global_srv_type_name(request.needed) << '\n';
    for (const auto &view : request.views)
        out << "slot " << view.slot << ' ' << global_srv_type_name(view.type) << ' ' << view.stride << ' ' << view.bytes << '\n';
    return out.str();
}

// False with a diagnostic in `error` for a text that cannot reconstruct the binding.
inline bool decode_global_srv_request(const std::string &text, GlobalSrvRequest &request, std::string &error) {
    request = {};
    std::istringstream in(text);
    std::string line;
    if (!std::getline(in, line) || line != "x2w-global-srv-views 2") {
        error = "global SRV view metadata has no 'x2w-global-srv-views 2' header";
        return false;
    }
    bool has_need = false;
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        std::string key;
        fields >> key;
        if (key.empty()) continue;
        if (key == "select") {
            if (!(fields >> request.selected)) { error = "global SRV view metadata: unreadable select line"; return false; }
            request.has_selected = true;
        } else if (key == "need") {
            std::string name;
            fields >> name;
            request.needed = global_srv_type_from_name(name);
            if (request.needed == GlobalSrvType::none) {
                error = "global SRV view metadata: unknown needed view type '" + name + "'";
                return false;
            }
            has_need = true;
        } else if (key == "slot") {
            GlobalSrvView view;
            std::string name;
            if (!(fields >> view.slot >> name >> view.stride >> view.bytes)) {
                error = "global SRV view metadata: unreadable slot line";
                return false;
            }
            view.type = global_srv_type_from_name(name);
            if (view.type == GlobalSrvType::none) {
                error = "global SRV slot " + std::to_string(view.slot) + " has unknown view type '" + name + "'";
                return false;
            }
            request.views.push_back(view);
        } else {
            error = "global SRV view metadata: unknown line '" + key + "'";
            return false;
        }
    }
    if (!request.has_selected || !has_need) {
        error = std::string("global SRV view metadata lacks the ") + (request.has_selected ? "needed view type" : "selected slot");
        return false;
    }
    return true;
}

// Empty when the selected slot holds the type the shader reads. Otherwise the refusal text: it names the slot and the
// view types, and is produced before the shader runs. Used by the native recording and by the replay alike.
inline std::string refuse_global_srv_selection(const std::vector<GlobalSrvView> &views, unsigned selected,
                                              GlobalSrvType needed) {
    for (const auto &view : views) {
        if (view.slot != selected) continue;
        if (view.type == needed) return {};
        return "global SRV slot " + std::to_string(selected) + " has view type " + global_srv_type_name(view.type) +
               ", the shader needs " + global_srv_type_name(needed);
    }
    return "global SRV slot " + std::to_string(selected) + " has no captured view type";
}
} // namespace x2w_startup
