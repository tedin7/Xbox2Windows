// Device-free checks for the opt-in translated-PSO recipe format. The runtime reads the same
// serializer/parser and reconstructs the desktop pipeline description from these fields.
#include "startup_stubs/pso_recipe.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {
int failures = 0;

void expect(bool condition, const char* what)
{
    std::printf("%s: %s\n", what, condition ? "ok" : "MISMATCH");
    failures += !condition;
}

x2w_startup::PsoRecipe sample_recipe()
{
    x2w_startup::PsoRecipe recipe;
    recipe.translation_key = 0x1122334455667788ull;
    recipe.target_count = 2;
    recipe.target_formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    recipe.target_formats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
    recipe.dsv_format = DXGI_FORMAT_D32_FLOAT;
    recipe.indexed_rect = true;
    recipe.use_depth_stencil = true;
    recipe.has_rasterizer = true;
    recipe.blend = x2w_startup::default_pso_blend();
    recipe.blend.IndependentBlendEnable = TRUE;
    recipe.blend.RenderTarget[0].BlendEnable = TRUE;
    recipe.blend.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
    recipe.blend.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    recipe.blend.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
    recipe.blend.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    recipe.blend.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
    recipe.blend.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
    recipe.blend.RenderTarget[0].RenderTargetWriteMask = 0x9;
    recipe.blend.RenderTarget[1].RenderTargetWriteMask = 0x3;
    recipe.depth_stencil.DepthEnable = TRUE;
    recipe.depth_stencil.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    recipe.depth_stencil.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    recipe.depth_stencil.StencilEnable = TRUE;
    recipe.depth_stencil.StencilReadMask = 0x7f;
    recipe.depth_stencil.StencilWriteMask = 0x0f;
    recipe.depth_stencil.FrontFace = {D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_INCR, D3D12_STENCIL_OP_REPLACE,
                                      D3D12_COMPARISON_FUNC_EQUAL};
    recipe.depth_stencil.BackFace = {D3D12_STENCIL_OP_ZERO, D3D12_STENCIL_OP_DECR, D3D12_STENCIL_OP_KEEP,
                                     D3D12_COMPARISON_FUNC_ALWAYS};
    recipe.rasterizer = x2w_startup::default_pso_rasterizer();
    recipe.rasterizer.CullMode = D3D12_CULL_MODE_BACK;
    recipe.rasterizer.FrontCounterClockwise = TRUE;
    recipe.rasterizer.DepthBias = -3;
    recipe.rasterizer.DepthBiasClamp = 0.5f;
    recipe.rasterizer.SlopeScaledDepthBias = -1.25f;
    recipe.rasterizer.MultisampleEnable = TRUE;
    recipe.sample_mask = 0xf0f0u;
    recipe.sample_count = 1;
    recipe.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    recipe.state_key = x2w_startup::pso_recipe_state_key(recipe);
    return recipe;
}

std::string replace_state_key(std::string text)
{
    const std::string marker = "state 0x";
    const std::size_t at = text.find(marker);
    if (at == std::string::npos) return text;
    for (std::size_t i = 0; i < 16; ++i) text[at + marker.size() + i] = '0';
    return text;
}
}  // namespace

int main()
{
    using namespace x2w_startup;
    const PsoRecipe recipe = sample_recipe();
    const std::string text = serialize_pso_recipe(recipe);
    expect(text.rfind("x2w-pso-recipe 1\n", 0) == 0, "serialized record starts with version 1");
    expect(text.size() <= pso_recipe_file_limit, "one record stays inside the file-size bound");

    const PsoRecipeParseResult parsed = parse_pso_recipe_file(text);
    expect(parsed.rejected == 0 && parsed.recipes.size() == 1, "record parses");
    if (parsed.recipes.size() == 1) {
        expect(serialize_pso_recipe(parsed.recipes[0]) == text, "serializer/parser round-trips byte-for-byte");

        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
        desc.NumRenderTargets = 8;
        desc.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
        desc.SampleMask = 0;
        desc.SampleDesc.Count = 4;
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
        apply_pso_recipe(parsed.recipes[0], desc);
        expect(same_pso_blend(desc.BlendState, recipe.blend), "blend state recreated");
        expect(same_pso_depth(desc.DepthStencilState, recipe.depth_stencil), "depth-stencil state recreated");
        expect(same_pso_rasterizer(desc.RasterizerState, recipe.rasterizer), "rasterizer state recreated");
        expect(desc.SampleMask == recipe.sample_mask && desc.SampleDesc.Count == recipe.sample_count,
               "sample mask and count recreated");
        expect(desc.PrimitiveTopologyType == recipe.topology, "topology recreated");
        expect(desc.NumRenderTargets == recipe.target_count && desc.RTVFormats[0] == recipe.target_formats[0] &&
                   desc.RTVFormats[1] == recipe.target_formats[1] && desc.RTVFormats[2] == DXGI_FORMAT_UNKNOWN &&
                   desc.DSVFormat == recipe.dsv_format,
               "target formats and DSV recreated");
    }

    PsoRecipe second = recipe;
    second.target_count = 1;
    second.target_formats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    second.target_formats[1] = DXGI_FORMAT_UNKNOWN;
    second.dsv_format = DXGI_FORMAT_UNKNOWN;
    second.indexed_rect = false;
    second.state_key = pso_recipe_state_key(second);
    const std::string two_records = text + serialize_pso_recipe(second);
    const PsoRecipeParseResult both = parse_pso_recipe_file(two_records);
    expect(both.rejected == 0 && both.recipes.size() == 2, "two appended records parse");
    if (both.recipes.size() == 2)
        expect(serialize_pso_recipe(both.recipes[0]) + serialize_pso_recipe(both.recipes[1]) == two_records,
               "two-record file round-trips byte-for-byte");

    const PsoRecipeParseResult corrupted = parse_pso_recipe_file(replace_state_key(text));
    expect(corrupted.recipes.empty() && corrupted.rejected == 1, "corrupt state key is rejected");

    std::string unknown = text;
    unknown.replace(0, std::strlen("x2w-pso-recipe 1"), "x2w-pso-recipe 2");
    const PsoRecipeParseResult future = parse_pso_recipe_file(unknown);
    expect(future.recipes.empty() && future.rejected == 1, "unknown recipe version is rejected");

    const PsoRecipeParseResult oversized = parse_pso_recipe_file(std::string(pso_recipe_file_limit + 1, 'x'));
    expect(oversized.recipes.empty() && oversized.rejected == 1, "oversized recipe file is rejected");

    DXGI_FORMAT wrong_formats[2] = {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT};
    expect(pso_recipe_targets_match(recipe, 2, recipe.target_formats, recipe.dsv_format),
           "matching target formats and DSV accepted");
    expect(!pso_recipe_targets_match(recipe, 2, wrong_formats, recipe.dsv_format), "target-format mismatch rejected");
    expect(!pso_recipe_targets_match(recipe, 2, recipe.target_formats, DXGI_FORMAT_D24_UNORM_S8_UINT),
           "DSV-format mismatch rejected");

    // The writer stops where the parser stops. At 63 records the next append is still readable;
    // at 64 it would become the record the next process rejects, so it is refused instead.
    std::string sixty_three;
    for (unsigned i = 0; i < pso_recipe_record_limit - 1; ++i) sixty_three += text;
    const PsoRecipeParseResult at_63 = parse_pso_recipe_file(sixty_three);
    expect(at_63.rejected == 0 && at_63.recipes.size() == pso_recipe_record_limit - 1, "63 records parse");
    expect(pso_recipe_append_allowed(sixty_three, text), "63 records accept one more append");
    expect(pso_recipe_append_allowed("", text), "an empty file accepts the first record");

    const std::string sixty_four = sixty_three + text;
    const PsoRecipeParseResult at_64 = parse_pso_recipe_file(sixty_four);
    expect(at_64.rejected == 0 && at_64.recipes.size() == pso_recipe_record_limit, "64 records parse");
    expect(!pso_recipe_append_allowed(sixty_four, text), "64 records refuse a 65th append");

    const PsoRecipeParseResult at_65 = parse_pso_recipe_file(sixty_four + text);
    expect(at_65.recipes.size() == pso_recipe_record_limit && at_65.rejected == 1 &&
               at_65.error == "recipe record limit reached",
           "65 records reject the extra record");
    expect(!pso_recipe_append_allowed(replace_state_key(text), text), "a malformed file is not appended to");
    expect(!pso_recipe_append_allowed(std::string(pso_recipe_file_limit, 'x'), text),
           "an unparseable full-size file is not appended to");

    // A record without its final LF still parses, but appending would put `end` and the next
    // header on one line. The writer refuses and leaves the parseable record intact.
    const std::string no_final_lf = text.substr(0, text.size() - 1);
    const PsoRecipeParseResult parsed_no_lf = parse_pso_recipe_file(no_final_lf);
    expect(parsed_no_lf.rejected == 0 && parsed_no_lf.recipes.size() == 1,
           "a record without a final LF still parses");
    expect(!pso_recipe_append_allowed(no_final_lf, text), "a file without a final LF refuses an append");
    const PsoRecipeParseResult after_refusal = parse_pso_recipe_file(no_final_lf);
    expect(after_refusal.rejected == 0 && after_refusal.recipes.size() == 1,
           "the earlier record remains readable after the refused append");

    std::printf("startup pso recipe checks: %d failed\n", failures);
    return failures ? 1 : 0;
}
