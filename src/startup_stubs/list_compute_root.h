#pragma once

// Descriptor tables of the compute root of one command list: command-list slot 35 (desktop 31 + 4,
// SetComputeRootDescriptorTable). Slot 33 (SetComputeRootSignature) keeps its own desktop bind; it only tells this
// state which root signature is current. Tables are recorded for a later Dispatch handler and are not forwarded.
//
// The rules follow the graphics side (slots 34 and 36): setting the root signature that is already set keeps the
// tables, setting a different one (or none) leaves no table bound, and so does a change of the title's view heap.
// A parameter past the kept range is refused and counted.

#include <array>
#include <cstdint>

#include "startup_stubs/logical_view_table.h"

namespace x2w_startup {

// Slot 32 builds this selection afresh for each call. The graphics lookup still singles out
// oversized logical heaps; compute table invalidation also needs ordinary shader-visible heaps.
struct ViewHeapSelection {
    std::uint64_t graphics = 0;
    std::uint64_t compute = 0;

    void note(std::uint64_t wrapper, bool shader_visible, unsigned logical_count, unsigned desktop_count)
    {
        if (shader_visible) compute = wrapper;
        if (logical_count > desktop_count) graphics = wrapper;
    }
};

struct RootView {
    std::uint64_t address;
    unsigned slot;
};

class RootViews {
public:
    template<typename Params, typename Type>
    bool record(const Params* params, unsigned parameter, Type type, unsigned slot, std::uint64_t address)
    {
        if (!params || parameter >= params->size() || parameter >= views_.size() ||
            (*params)[parameter].type != type) return false;
        views_[parameter] = {address, slot};
        return true;
    }

    RootView view(unsigned parameter) const { return parameter < views_.size() ? views_[parameter] : RootView{}; }
    void clear() { views_.fill({}); }

private:
    std::array<RootView, 64> views_{};
};

class ComputeTables {
public:
    static constexpr unsigned max_parameters = 16;  // same range as the graphics tables

    // The root signature is identified by any stable pointer (the title wrapper's parameter list); null clears it.
    void set_signature(const void* signature)
    {
        if (signature != signature_) clear();
        signature_ = signature;
    }

    // Slot 32 with another view heap: the recorded addresses belong to the old one.
    void set_view_heap(std::uint64_t heap)
    {
        if (heap != view_heap_) clear();
        view_heap_ = heap;
    }

    // False when the parameter is out of range (counted).
    bool set_table(unsigned parameter, const ViewTable& table)
    {
        if (parameter >= max_parameters) {
            ++out_of_range_;
            return false;
        }
        tables_[parameter] = table;
        return true;
    }

    const void* signature() const { return signature_; }
    ViewTable table(unsigned parameter) const { return parameter < max_parameters ? tables_[parameter] : ViewTable{}; }
    std::uint32_t out_of_range_count() const { return out_of_range_; }

private:
    void clear()
    {
        for (auto& table : tables_) table = {};
    }

    const void* signature_ = nullptr;
    std::uint64_t view_heap_ = 0;
    ViewTable tables_[max_parameters] = {};
    std::uint32_t out_of_range_ = 0;
};

}  // namespace x2w_startup
