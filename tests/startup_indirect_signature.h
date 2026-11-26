#pragma once
#include <d3d12.h>

struct ConstantDispatchSignature {
    D3D12_INDIRECT_ARGUMENT_DESC arguments[2]{};

    ConstantDispatchSignature(UINT parameter, UINT first_word, UINT count)
    {
        arguments[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
        arguments[0].Constant = {parameter, first_word, count};
        arguments[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    }

    D3D12_COMMAND_SIGNATURE_DESC desc() const
    {
        const UINT bytes = arguments[0].Constant.Num32BitValuesToSet * 4 + sizeof(D3D12_DISPATCH_ARGUMENTS);
        return {bytes, 2, arguments, 0};
    }
};
