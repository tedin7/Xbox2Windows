// Checks src/startup_stubs/list_compute_root.h without a device: the descriptor tables slot 35 records for a
// compute Dispatch, resolved against a title view heap the way the graphics tables are (slot 36).
#include "startup_stubs/list_compute_root.h"

#include <cstdint>
#include <cstdio>
#include <array>

namespace {
using x2w_startup::ComputeTables;
using x2w_startup::ViewHeapSelection;
using x2w_startup::ViewTable;
using x2w_startup::resolve_view_table;
int failures;

void check(bool ok, const char* name)
{
    if (!ok) {
        std::printf("FAIL %s\n", name);
        ++failures;
    }
}

// A title heap of 8 descriptors of 32 bytes: GPU base 0x10000, CPU base 0x7000.
constexpr std::uint64_t gpu_base = 0x10000, cpu_base = 0x7000;
constexpr unsigned count = 8, stride = 32;
ViewTable at(unsigned descriptor) { return resolve_view_table(gpu_base, cpu_base, count, stride, gpu_base + std::uint64_t{descriptor} * stride); }
}  // namespace

int main()
{
    int root_a = 0, root_b = 0;

    {
        // A UAV table at a nonzero descriptor offset resolves to that offset, not to the heap start.
        ComputeTables s;
        s.set_signature(&root_a);
        check(s.set_table(2, at(3)), "table set");
        check(s.table(2).address == cpu_base + 3 * stride && s.table(2).bytes == 5 * stride, "slot 35 keeps the offset table");
        check(s.table(2).address != at(0).address, "a nonzero offset is not the default table");
        check(s.table(3).address == 0, "an unset parameter has no table");
    }
    {
        // A handle that is not in the heap resolves to nothing: the wrong binding is not turned into a default.
        check(resolve_view_table(gpu_base, cpu_base, count, stride, gpu_base + count * stride).address == 0, "past the heap");
        check(resolve_view_table(gpu_base, cpu_base, count, stride, gpu_base + 5).address == 0, "inside a descriptor");
        check(resolve_view_table(gpu_base, cpu_base, count, stride, gpu_base - stride).address == 0, "before the heap");
        ComputeTables s;
        s.set_signature(&root_a);
        s.set_table(0, at(1));
        s.set_table(0, resolve_view_table(gpu_base, cpu_base, count, stride, 0xdead0000));
        check(s.table(0).address == 0, "a wrong handle replaces the earlier binding with nothing");
    }
    {
        ComputeTables s;
        s.set_signature(&root_a);
        s.set_table(0, at(1));
        s.set_table(5, at(4));
        s.set_signature(&root_a);
        check(s.table(0).address == cpu_base + stride && s.table(5).address == cpu_base + 4 * stride, "the same signature keeps the tables");
        s.set_signature(&root_b);
        check(s.signature() == &root_b && s.table(0).address == 0 && s.table(5).address == 0, "a different signature clears the tables");
        s.set_table(1, at(2));
        s.set_signature(nullptr);
        check(s.signature() == nullptr && s.table(1).address == 0, "a null signature clears the tables");
    }
    {
        ComputeTables s;
        s.set_signature(&root_a);
        const auto select = [&](std::uint64_t heap, bool visible, unsigned logical, unsigned desktop) {
            ViewHeapSelection selection;
            selection.note(heap, visible, logical, desktop);
            s.set_view_heap(selection.compute);
            return selection;
        };
        auto selection = select(1, true, 8, 8);
        check(selection.compute == 1 && !selection.graphics, "ordinary heap is selected only for compute invalidation");
        s.set_table(7, at(6));
        select(1, true, 8, 8);
        check(s.table(7).address == at(6).address, "same ordinary heap keeps the table");
        select(2, true, 8, 8);
        check(!s.table(7).address, "different ordinary heap clears the table");
        s.set_table(7, at(6));
        selection = select(3, true, 16, 8);
        check(selection.compute == 3 && selection.graphics == 3, "oversized heap remains visible to graphics lookup");
        check(!s.table(7).address, "ordinary to oversized clears the table");
        s.set_table(7, at(6));
        select(3, true, 16, 8);
        check(s.table(7).address == at(6).address, "same oversized heap keeps the table");
        select(4, true, 16, 8);
        check(!s.table(7).address, "different oversized heap clears the table");
        s.set_table(7, at(6));
        select(1, true, 8, 8);
        check(!s.table(7).address, "oversized to ordinary clears the table");
        s.set_table(7, at(6));
        s.set_view_heap(ViewHeapSelection{}.compute);
        check(!s.table(7).address && s.signature() == &root_a, "unsetting the heap clears the table and keeps the signature");
        selection = select(5, false, 8, 8);
        check(!selection.compute && !selection.graphics, "CPU-only heap does not select a view heap");
    }
    {
        ComputeTables s;
        s.set_signature(&root_a);
        s.set_table(1, at(1));
        check(s.set_table(ComputeTables::max_parameters - 1, at(2)) && s.table(ComputeTables::max_parameters - 1).address, "the last parameter is kept");
        check(!s.set_table(ComputeTables::max_parameters, at(3)) && s.out_of_range_count() == 1, "a parameter out of range is refused and counted");
        check(s.table(ComputeTables::max_parameters).address == 0 && s.table(1).address == cpu_base + stride, "a refused call leaves the other tables alone");
    }
    {
        enum Type { srv, uav, cbv };
        struct Parameter { Type type; };
        const std::array<Parameter, 3> params{{{srv}, {uav}, {cbv}}};
        x2w_startup::RootViews views;
        check(views.record(&params, 0, srv, 43, 0x1010), "slot 43 records its SRV address");
        check(views.record(&params, 1, uav, 45, 0x2020), "slot 45 records its UAV address");
        check(views.record(&params, 2, cbv, 41, 0x3100), "slot 41 records its CBV address");
        check(!views.record(&params, 0, uav, 45, 0x4040), "a UAV slot cannot write an SRV parameter");
        check(!views.record(&params, 3, uav, 45, 0x4040), "an undeclared parameter is refused");
        check(views.view(0).slot == 43 && views.view(0).address == 0x1010 &&
            views.view(1).slot == 45 && views.view(1).address == 0x2020 &&
            views.view(2).slot == 41 && views.view(2).address == 0x3100, "only fitting views are kept");
        views.clear();
        check(!views.view(0).address && !views.view(1).address && !views.view(2).address, "clearing drops all views");
    }
    std::printf("startup_compute_root: %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
