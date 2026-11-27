#pragma once

#include <windows.h>
#include <objidl.h>
#include <oleauto.h>
#include <dxcapi.h>
#include <wrl/client.h>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace x2w {

std::string shader_digest(std::string_view bytes);
enum class ShaderSourceFormat { hlsl, dxil_ir };

struct CompiledShader {
    std::vector<unsigned char> dxil;
    bool cache_hit = false;
};

// The caller supplies a matching SDK dxcompiler.dll and dxil.dll in the same directory.
// Source must be self-contained. The backend never resolves process-relative includes.
class ShaderDxilCompiler {
public:
    explicit ShaderDxilCompiler(const std::filesystem::path& sdk_directory);
    ShaderDxilCompiler(const ShaderDxilCompiler&) = delete;
    ShaderDxilCompiler& operator=(const ShaderDxilCompiler&) = delete;
    // Independent DXC interfaces, with the same loaded and fingerprinted SDK.
    ShaderDxilCompiler new_worker() const;
    // The startup shim uses one SDK directory for the lifetime of the process.
    static ShaderDxilCompiler runtime_worker(const std::filesystem::path& sdk_directory);
    CompiledShader compile(const std::string& hlsl, const std::wstring& profile,
                           const std::filesystem::path& cache, ShaderSourceFormat format = ShaderSourceFormat::hlsl);
    HRESULT reflect(const void* bytecode, size_t bytes, REFIID iid, void** result);
private:
    struct Sdk;
    explicit ShaderDxilCompiler(std::shared_ptr<const Sdk> sdk);
    std::shared_ptr<const Sdk> sdk_;
    Microsoft::WRL::ComPtr<IDxcCompiler3> compiler_;
    Microsoft::WRL::ComPtr<IDxcUtils> utils_;
};
}
