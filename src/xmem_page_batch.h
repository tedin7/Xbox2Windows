#pragma once

#include <windows.h>

struct X2WPageRange {
    UINT kind;  // 0: consecutive, 1: repeat start, 2: unmap, 3: keep the previous mapping
    UINT start; // index from the source pool base, ignored for unmap and keep
    UINT count; // 64 KiB destination pages covered by the range
};

struct X2WPageBatch {
    UINT64 destination;
    UINT count;
    const X2WPageRange* ranges;
};

struct X2WPageBatchStats {
    UINT64 pagesChanged;
    UINT64 pagesSkipped;
    UINT64 mapCalls;
    UINT64 unmapCalls;
};

static_assert(sizeof(X2WPageRange) == 12 && sizeof(X2WPageBatch) == 24);
// Flags must be zero. Earlier ranges count toward input limits even when superseded by later edits.
// On success stats counts unique destination pages and completed OS view operations (including survivors).
// On failure stats is zero. Descriptors may overlap destinations and are read before remapping.
extern "C" HRESULT WINAPI X2WMapPageBatch(UINT count, const X2WPageBatch* batches, UINT64 pool,
                                          UINT poolPages, UINT flags, X2WPageBatchStats* stats);
