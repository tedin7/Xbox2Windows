// Unit check for src/startup_stubs/d16_image_format.h (#2023, item D).
#include "startup_stubs/d16_image_format.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {
struct FakeSrv {
    int kind;
};

constexpr DXGI_FORMAT kFloatExact[] = {
    DXGI_FORMAT_R32G32B32A32_FLOAT,
    DXGI_FORMAT_R32G32B32_FLOAT,
    DXGI_FORMAT_R32G32_FLOAT,
    DXGI_FORMAT_R32_FLOAT,
    DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
    DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_FORMAT_R8G8_UNORM,
    DXGI_FORMAT_R8_UNORM,
    DXGI_FORMAT_A8_UNORM,
    DXGI_FORMAT_B8G8R8A8_UNORM,
    DXGI_FORMAT_B8G8R8X8_UNORM,
};

constexpr DXGI_FORMAT kUnorm16[] = {
    DXGI_FORMAT_R16G16B16A16_UNORM,
    DXGI_FORMAT_R16G16_UNORM,
    DXGI_FORMAT_R16_UNORM,
};

x2w_startup::D16SrvMode expected_mode(DXGI_FORMAT fmt)
{
    for (DXGI_FORMAT candidate : kFloatExact) {
        if (candidate == fmt) return x2w_startup::D16SrvMode::float_exact;
    }
    for (DXGI_FORMAT candidate : kUnorm16) {
        if (candidate == fmt) return x2w_startup::D16SrvMode::unorm16;
    }
    return x2w_startup::D16SrvMode::refused;
}
}  // namespace

int main(int argc, char** argv)
{
    const bool mutate_accept_uint = argc == 3 && std::strcmp(argv[1], "--mutate") == 0 &&
                                    std::strcmp(argv[2], "accept-uint") == 0;
    if (argc != 1 && !mutate_accept_uint) return 2;

    const auto classify = [mutate_accept_uint](DXGI_FORMAT fmt) {
        if (mutate_accept_uint && fmt == DXGI_FORMAT_R16_UINT)
            return x2w_startup::D16SrvMode::float_exact;
        return x2w_startup::d16_srv_mode(fmt);
    };

    for (unsigned raw = 0; raw <= 255; ++raw) {
        const auto fmt = static_cast<DXGI_FORMAT>(raw);
        const auto got = classify(fmt);
        if (got != expected_mode(fmt)) {
            std::fprintf(stderr, "format %u mode mismatch (%u)\n", raw, static_cast<unsigned>(got));
            return 1;
        }
        if (!mutate_accept_uint && x2w_startup::d16_exact_srv_format(fmt) !=
                (got != x2w_startup::D16SrvMode::refused)) {
            std::fprintf(stderr, "format %u exact flag mismatch\n", raw);
            return 1;
        }
    }

    // Explicit family checks: integer, typeless, depth-stencil-only, SNORM16, R24_UNORM, SRGB, BCn and console stand-ins must be refused.
    constexpr DXGI_FORMAT kRefusedSpotChecks[] = {
        DXGI_FORMAT_UNKNOWN,
        DXGI_FORMAT_R32G32B32A32_TYPELESS,
        DXGI_FORMAT_R32G32B32A32_UINT,
        DXGI_FORMAT_R32G32B32A32_SINT,
        DXGI_FORMAT_R16G16B16A16_TYPELESS,
        DXGI_FORMAT_R16G16B16A16_SNORM,
        DXGI_FORMAT_R16G16B16A16_UINT,
        DXGI_FORMAT_R16G16B16A16_SINT,
        DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
        DXGI_FORMAT_X32_TYPELESS_G8X24_UINT,
        DXGI_FORMAT_R10G10B10A2_TYPELESS,
        DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R10G10B10A2_UINT,
        DXGI_FORMAT_R8G8B8A8_TYPELESS,
        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        DXGI_FORMAT_R8G8B8A8_SNORM,
        DXGI_FORMAT_R8G8B8A8_UINT,
        DXGI_FORMAT_R8G8B8A8_SINT,
        DXGI_FORMAT_R16G16_SNORM,
        DXGI_FORMAT_R32_TYPELESS,
        DXGI_FORMAT_D32_FLOAT,
        DXGI_FORMAT_R32_UINT,
        DXGI_FORMAT_R32_SINT,
        DXGI_FORMAT_D24_UNORM_S8_UINT,
        DXGI_FORMAT_R24_UNORM_X8_TYPELESS,
        DXGI_FORMAT_X24_TYPELESS_G8_UINT,
        DXGI_FORMAT_R16_TYPELESS,
        DXGI_FORMAT_D16_UNORM,
        DXGI_FORMAT_R16_UINT,
        DXGI_FORMAT_R16_SNORM,
        DXGI_FORMAT_R16_SINT,
        DXGI_FORMAT_R8_TYPELESS,
        DXGI_FORMAT_R8_UINT,
        DXGI_FORMAT_R8_SINT,
        DXGI_FORMAT_BC1_UNORM,
        DXGI_FORMAT_BC6H_UF16,
        DXGI_FORMAT_BC7_UNORM,
        DXGI_FORMAT_B5G6R5_UNORM,
        static_cast<DXGI_FORMAT>(116),
        static_cast<DXGI_FORMAT>(117),
        static_cast<DXGI_FORMAT>(118),
        static_cast<DXGI_FORMAT>(119),
        static_cast<DXGI_FORMAT>(120),
    };
    for (DXGI_FORMAT fmt : kRefusedSpotChecks) {
        if (classify(fmt) != x2w_startup::D16SrvMode::refused) {
            std::fprintf(stderr, "refused format %d accepted\n", static_cast<int>(fmt));
            return 1;
        }
    }

    x2w_startup::D16ImageSrvs srvs;
    if (srvs.any() || !srvs.valid_for_srvs(std::vector<FakeSrv>{})) return 1;
    if (!srvs.read_manifest_line("d16srv 0 exact") || !srvs.read_manifest_line("d16srv 31 exact"))
        return 1;
    if (!srvs.any() || !srvs.requires_exact(0) || !srvs.requires_exact(31) ||
        srvs.requires_exact(1) || srvs.requires_exact(32))
        return 1;

    constexpr const char* kBadLines[] = {
        "d16srv 0 exact",       // duplicate index 0
        "d16srv 00 exact",      // leading zero
        "d16srv 01 exact",      // leading zero
        "d16srv 32 exact",      // out of range
        "d16srv -1 exact",      // negative
        "d16srv  exact",        // missing index
        "d16srv 2 float_norm",  // obsolete tag
        "d16srv 2 uint",        // unsupported expectation
        "d16srv 2 exact ",      // trailing whitespace
        "wmask 0 0x1",          // wrong tag
    };
    for (const char* bad : kBadLines) {
        if (srvs.read_manifest_line(bad)) {
            std::fprintf(stderr, "accepted bad manifest line: %s\n", bad);
            return 1;
        }
    }

    std::vector<FakeSrv> bindings(32, FakeSrv{1});
    if (!srvs.valid_for_srvs(bindings)) return 1;
    bindings[31].kind = 4;  // tex1d is a texture SRV
    if (!srvs.valid_for_srvs(bindings)) return 1;
    bindings[0].kind = 0;   // buffer SRV cannot satisfy d16srv
    if (srvs.valid_for_srvs(bindings)) return 1;
    bindings[0].kind = 3;   // tex3d
    bindings.pop_back();    // index 31 now out of range
    if (srvs.valid_for_srvs(bindings)) return 1;

    std::printf("d16_image_format: 14 exact formats (11 float_exact + 3 unorm16), 242 refused values and manifest parser verified\n");
    return 0;
}
