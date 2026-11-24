#pragma once

// Compute root state of one command list: command-list slot 33 (desktop 29 + 4, SetComputeRootSignature) and
// slot 35 (desktop 31 + 4, SetComputeRootDescriptorTable). Both are seen in real logs next to Dispatch (slot 14).
// Only the state is recorded here and nothing is forwarded to the desktop list. A Dispatch handler reads it through
// ready() and table(): without a compute root signature set, the desktop list rejects the dispatch.
//
// The rules follow the graphics side (slot 34): setting the root signature that is already set keeps the descriptor
// tables, setting a different one (or none) leaves no table bound. A parameter index past the kept range is
// refused and counted, and a list that is not recording ignores both calls.

#include <cstdint>

namespace x2w_startup {

class ComputeRootState {
public:
    static constexpr unsigned max_parameters = 16;  // same range as the graphics tables

    // What a call did, so the caller can log it and tests can check it.
    enum class Result { recorded, ignored_closed, out_of_range };

    // The root signature is identified by any stable pointer (the title wrapper's parameter list); null clears it.
    Result set_signature(bool recording, const void* signature)
    {
        if (!recording) {
            ++closed_ignored_;
            return Result::ignored_closed;
        }
        if (signature != signature_)
            for (auto& handle : tables_) handle = 0;
        signature_ = signature;
        return Result::recorded;
    }

    // gpu_handle is the title's GPU descriptor handle of the table, kept as given (0 = nothing bound).
    Result set_table(bool recording, unsigned parameter, std::uint64_t gpu_handle)
    {
        if (!recording) {
            ++closed_ignored_;
            return Result::ignored_closed;
        }
        if (parameter >= max_parameters) {
            ++out_of_range_;
            return Result::out_of_range;
        }
        tables_[parameter] = gpu_handle;
        return Result::recorded;
    }

    // Accessors for a Dispatch handler.
    bool ready() const { return signature_ != nullptr; }
    const void* signature() const { return signature_; }
    std::uint64_t table(unsigned parameter) const { return parameter < max_parameters ? tables_[parameter] : 0; }
    std::uint32_t out_of_range_count() const { return out_of_range_; }
    std::uint32_t closed_ignored_count() const { return closed_ignored_; }

private:
    const void* signature_ = nullptr;
    std::uint64_t tables_[max_parameters] = {};
    std::uint32_t out_of_range_ = 0;
    std::uint32_t closed_ignored_ = 0;
};

}  // namespace x2w_startup
