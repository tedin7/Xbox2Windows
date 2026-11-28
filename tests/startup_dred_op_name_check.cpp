#include "startup_stubs/d3d12_dred.h"

#include <string_view>

using x2w_dred::op_name;
using namespace std::literals;

static_assert(op_name(D3D12_AUTO_BREADCRUMB_OP_SETMARKER) == "SETMARKER"sv);
static_assert(op_name(D3D12_AUTO_BREADCRUMB_OP_DRAWINSTANCED) == "DRAWINSTANCED"sv);
static_assert(op_name(D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER) == "RESOURCEBARRIER"sv);
static_assert(op_name(D3D12_AUTO_BREADCRUMB_OP_DISPATCHRAYS) == "DISPATCHRAYS"sv);
static_assert(op_name(D3D12_AUTO_BREADCRUMB_OP_DISPATCHMESH) == "DISPATCHMESH"sv);
static_assert(op_name(56) == "UNKNOWN"sv);
static_assert(op_name(255) == "UNKNOWN"sv);

int main() { return 0; }
