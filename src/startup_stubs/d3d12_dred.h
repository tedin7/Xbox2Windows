#pragma once

#include "startup_stubs/local_log.h"

#include <d3d12.h>
#include <d3d12sdklayers.h>

namespace x2w_dred {

constexpr const char* op_name(unsigned index) noexcept
{
    // D3D12_AUTO_BREADCRUMB_OP values in SDK order. Use the stable numeric values
    // so older SDKs can still name operations returned by a newer runtime.
    constexpr const char* names[] = {
        "SETMARKER", "BEGINEVENT", "ENDEVENT", "DRAWINSTANCED", "DRAWINDEXEDINSTANCED",
        "EXECUTEINDIRECT", "DISPATCH", "COPYBUFFERREGION", "COPYTEXTUREREGION", "COPYRESOURCE",
        "COPYTILES", "RESOLVESUBRESOURCE", "CLEARRENDERTARGETVIEW", "CLEARUNORDEREDACCESSVIEW",
        "CLEARDEPTHSTENCILVIEW", "RESOURCEBARRIER", "EXECUTEBUNDLE", "PRESENT", "RESOLVEQUERYDATA",
        "BEGINSUBMISSION", "ENDSUBMISSION", "DECODEFRAME", "PROCESSFRAMES", "ATOMICCOPYBUFFERUINT",
        "ATOMICCOPYBUFFERUINT64", "RESOLVESUBRESOURCEREGION", "WRITEBUFFERIMMEDIATE", "DECODEFRAME1",
        "SETPROTECTEDRESOURCESESSION", "DECODEFRAME2", "PROCESSFRAMES1", "BUILDRAYTRACINGACCELERATIONSTRUCTURE",
        "EMITRAYTRACINGACCELERATIONSTRUCTUREPOSTBUILDINFO", "COPYRAYTRACINGACCELERATIONSTRUCTURE",
        "DISPATCHRAYS", "INITIALIZEMETACOMMAND", "EXECUTEMETACOMMAND", "ESTIMATEMOTION",
        "RESOLVEMOTIONVECTORHEAP", "SETPIPELINESTATE1", "INITIALIZEEXTENSIONCOMMAND",
        "EXECUTEEXTENSIONCOMMAND", "DISPATCHMESH", "ENCODEFRAME", "RESOLVEENCODEROUTPUTMETADATA",
        "BARRIER", "BEGIN_COMMAND_LIST", "DISPATCHGRAPH", "SETPROGRAM", "ENCODEFRAME1",
        "RESOLVEENCODEROUTPUTMETADATA1", "RESOLVEINPUTPARAMLAYOUT", "PROCESSFRAMES2",
        "SET_WORK_GRAPH_MAXIMUM_GPU_INPUT_RECORDS", "WAITBARRIER", "SIGNALBARRIER", "UNKNOWN",
        "COPYBUFFERREGIONS", "COPYTEXTUREREGIONS", "COPYRESOURCES", "ASYNCCOPYTILES",
        "ASYNCRESOLVESUBRESOURCEREGION", "ASYNCRESOLVEQUERYDATA", "CLEARBOUNDDEPTHSTENCILVIEW",
        "CLEARBOUNDRENDERTARGETVIEWS", "CLEARTEXTURESUBRESOURCES", "FILLBUFFERS",
    };
    return index < sizeof(names) / sizeof(names[0]) ? names[index] : "UNKNOWN";
}

inline bool enabled() noexcept
{
    static const bool on = x2w_local::env_on("X2W_DRED");
    return on;
}

inline void configure_before_device() noexcept
{
    if (!enabled()) return;
    static const bool configured = [] {
        ID3D12DeviceRemovedExtendedDataSettings* settings = nullptr;
        const HRESULT hr = D3D12GetDebugInterface(IID_PPV_ARGS(&settings));
        if (FAILED(hr)) {
            x2w_local::logf("x2w-dred: settings unavailable (0x%08lx)", static_cast<unsigned long>(hr));
            return false;
        }
        settings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        settings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        settings->Release();
        x2w_local::logf("x2w-dred: auto-breadcrumbs and page faults on");
        return true;
    }();
    (void)configured;
}

inline void debug_name(char (&name)[256], const wchar_t* wide, const char* narrow) noexcept
{
    if (wide && *wide) {
        int length = 0;
        while (length < 80 && wide[length]) ++length;
        const int bytes = WideCharToMultiByte(CP_UTF8, 0, wide, length, name, sizeof(name) - 1, nullptr, nullptr);
        name[bytes] = '\0';
    } else if (narrow) {
        std::snprintf(name, sizeof(name), "%.255s", narrow);
    } else {
        name[0] = '\0';
    }
    for (char* ch = name; *ch; ++ch)
        if (*ch == '\r' || *ch == '\n') *ch = ' ';
}

template<class Node>
void log_breadcrumbs(const Node* node) noexcept
{
    for (unsigned i = 0; node && i < 16; ++i, node = node->pNext) {
        char list[256], queue[256];
        debug_name(list, node->pCommandListDebugNameW, node->pCommandListDebugNameA);
        debug_name(queue, node->pCommandQueueDebugNameW, node->pCommandQueueDebugNameA);
        const UINT completed = node->pLastBreadcrumbValue ? *node->pLastBreadcrumbValue : 0;
        constexpr UINT history_size = 65536;
        const char* op = "UNKNOWN";
        if (completed == 0) op = "NONE";
        else if (node->pCommandHistory && completed <= node->BreadcrumbCount &&
                 node->BreadcrumbCount - completed < history_size)
            op = op_name(node->pCommandHistory[(completed - 1) % history_size]);
        x2w_local::logf("x2w-dred: list=\"%s\" queue=\"%s\" last completed op %u of %u: %s",
                       list, queue, completed, node->BreadcrumbCount, op);
    }
}

inline const void* allocation_object(const D3D12_DRED_ALLOCATION_NODE1* node) noexcept { return node->pObject; }
inline const void* allocation_object(const D3D12_DRED_ALLOCATION_NODE*) noexcept { return nullptr; }

template<class Node>
void log_allocations(const char* state, const Node* node) noexcept
{
    for (unsigned i = 0; node && i < 8; ++i, node = node->pNext) {
        char name[256];
        debug_name(name, node->ObjectNameW, node->ObjectNameA);
        x2w_local::logf("x2w-dred: %s allocation name=\"%s\" type=%u object=%p", state, name,
                       static_cast<unsigned>(node->AllocationType), allocation_object(node));
    }
}

template<class Output>
void log_page_fault(const Output& page) noexcept
{
    x2w_local::logf("x2w-dred: page fault VA=0x%llx", static_cast<unsigned long long>(page.PageFaultVA));
    log_allocations("existing", page.pHeadExistingAllocationNode);
    log_allocations("recently freed", page.pHeadRecentFreedAllocationNode);
}

inline void report_removal(ID3D12Device* device, HRESULT reason) noexcept
{
    // Called by the stand-in only after it claims the first removal report.
    if (!enabled()) return;
    x2w_local::logf("x2w-dred: device removed reason=0x%08lx", static_cast<unsigned long>(reason));

    // DRED2 inherits these DRED1 queries. DRED1 adds object pointers to allocation nodes.
    ID3D12DeviceRemovedExtendedData1* dred1 = nullptr;
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dred1)))) {
        D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 breadcrumbs{};
        if (SUCCEEDED(dred1->GetAutoBreadcrumbsOutput1(&breadcrumbs))) log_breadcrumbs(breadcrumbs.pHeadAutoBreadcrumbNode);
        D3D12_DRED_PAGE_FAULT_OUTPUT1 page{};
        if (SUCCEEDED(dred1->GetPageFaultAllocationOutput1(&page))) log_page_fault(page);
        dred1->Release();
        return;
    }
    ID3D12DeviceRemovedExtendedData* dred = nullptr;
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dred)))) {
        D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT breadcrumbs{};
        if (SUCCEEDED(dred->GetAutoBreadcrumbsOutput(&breadcrumbs))) log_breadcrumbs(breadcrumbs.pHeadAutoBreadcrumbNode);
        D3D12_DRED_PAGE_FAULT_OUTPUT page{};
        if (SUCCEEDED(dred->GetPageFaultAllocationOutput(&page))) log_page_fault(page);
        dred->Release();
    }
}

}  // namespace x2w_dred
