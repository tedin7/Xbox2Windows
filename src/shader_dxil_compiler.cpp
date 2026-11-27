#include <windows.h>
#include <bcrypt.h>
#include "shader_dxil_compiler.h"
#include "gfx1013_compute_ir.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;
using Microsoft::WRL::ComPtr;
namespace {
using Bytes = std::vector<unsigned char>;
bool try_read_file(const fs::path& path, Bytes& bytes) {
    HANDLE handle = CreateFileW(path.c_str(), GENERIC_READ,
                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER size{};
    const bool sized = GetFileSizeEx(handle, &size) && size.QuadPart >= 0 && size.QuadPart <= 256ll * 1024 * 1024;
    bool ok = sized;
    if (sized) {
        bytes.resize(static_cast<size_t>(size.QuadPart));
        DWORD read_bytes = 0;
        ok = !size.QuadPart || (ReadFile(handle, bytes.data(), static_cast<DWORD>(size.QuadPart), &read_bytes, nullptr) &&
                                read_bytes == static_cast<DWORD>(size.QuadPart));
    }
    CloseHandle(handle);
    return ok;
}
Bytes read_file(const fs::path& path) {
    Bytes bytes;
    if (!try_read_file(path, bytes)) throw std::runtime_error("unreadable or oversized input");
    return bytes;
}
std::string digest(const void* data, size_t size) {
    if (size > ULONG_MAX) throw std::length_error("digest input too large");
    std::array<unsigned char,32> hash{};
    if (BCryptHash(BCRYPT_SHA256_ALG_HANDLE,nullptr,0,
                   static_cast<PUCHAR>(const_cast<void*>(data)),static_cast<ULONG>(size),
                   hash.data(),static_cast<ULONG>(hash.size())) < 0) throw std::runtime_error("SHA-256 failed");
    std::string result;
    constexpr char hex[] = "0123456789abcdef";
    for (auto byte : hash) { result += hex[byte >> 4]; result += hex[byte & 15]; }
    return result;
}
void checked(HRESULT status) {
    if (FAILED(status)) throw std::runtime_error("DXC API failure " + std::to_string(status));
}
}
namespace x2w {
std::string shader_digest(std::string_view bytes) { return digest(bytes.data(),bytes.size()); }
struct ShaderDxilCompiler::Sdk {
    struct Module {
        HMODULE handle{};
        bool process_lifetime{};
        ~Module() { if (handle && !process_lifetime) FreeLibrary(handle); }
        Module() = default;
        Module(const Module&) = delete;
        Module& operator=(const Module&) = delete;
    } validator, compiler;
    std::string identity;
    DxcCreateInstanceProc create{};
    explicit Sdk(const fs::path& directory);
};

ShaderDxilCompiler::Sdk::Sdk(const fs::path& sdk_directory) {
    auto load = [&](Module& module, const wchar_t* name) {
        auto path = fs::canonical(sdk_directory / name);
        auto file = read_file(path);
        identity += digest(file.data(), file.size());
        module.handle = LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!module.handle) throw std::runtime_error("could not load DXC SDK module");
    };
    load(validator, L"dxil.dll");
    load(compiler, L"dxcompiler.dll");
    create = reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(compiler.handle,"DxcCreateInstance"));
    if (!create) throw std::runtime_error("DXC entry missing");
}

ShaderDxilCompiler::ShaderDxilCompiler(const fs::path& directory)
    : ShaderDxilCompiler(std::make_shared<Sdk>(directory)) {}

ShaderDxilCompiler::ShaderDxilCompiler(std::shared_ptr<const Sdk> sdk) : sdk_(std::move(sdk)) {
    checked(sdk_->create(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler_)));
    checked(sdk_->create(CLSID_DxcUtils, IID_PPV_ARGS(&utils_)));
}

ShaderDxilCompiler ShaderDxilCompiler::new_worker() const {
    return ShaderDxilCompiler(sdk_);
}

ShaderDxilCompiler ShaderDxilCompiler::runtime_worker(const fs::path& directory) {
    static const auto sdk = [&] {
        auto loaded = std::make_shared<Sdk>(directory);
        // The loader releases these two runtime references at process shutdown.
        // A shim static destructor runs under loader lock, after DXC may detach.
        loaded->compiler.process_lifetime = true;
        loaded->validator.process_lifetime = true;
        return loaded;
    }();
    return ShaderDxilCompiler(sdk);
}

HRESULT ShaderDxilCompiler::reflect(const void* bytecode,size_t bytes,REFIID iid,void** result) {
    DxcBuffer buffer{bytecode,bytes,0};
    return utils_->CreateReflection(&buffer,iid,result);
}

CompiledShader ShaderDxilCompiler::compile(const std::string& source_text, const std::wstring& profile,
                                          const fs::path& cache, ShaderSourceFormat format) {
    const bool direct = format == ShaderSourceFormat::dxil_ir;
    const bool native_ir = source_text.find("#pragma x2w_") != std::string::npos;
    // Bump the native IR version when instruction semantics change. Keying
    // before expansion lets a validated hit bypass instruction lowering.
    const char* revision = direct ? "x2w-direct-dxil-1/validated/" : native_ir ? "x2w-dxil-native-ir-1/main/-Gis/-O3/" : "x2w-dxil-1/main/-Gis/-O3/";
    const std::string identity = revision + sdk_->identity + digest(profile.data(),profile.size()*sizeof(wchar_t)) +
        digest(source_text.data(),source_text.size());
    const auto key = digest(identity.data(),identity.size());
    const auto path = cache / (key + ".bin");
    Bytes cached;
    if (try_read_file(path, cached) && cached.size() > 128 && std::equal(key.begin(), key.end(), cached.begin())) {
        const auto hash = digest(cached.data() + 128, cached.size() - 128);
        if (std::equal(hash.begin(), hash.end(), cached.begin() + 64))
            return {Bytes(cached.begin() + 128, cached.end()), true};
    }
    ComPtr<IDxcBlob> blob;
    if (direct) {
        ComPtr<IDxcBlobEncoding> input;
        checked(utils_->CreateBlob(source_text.data(),static_cast<UINT32>(source_text.size()),DXC_CP_UTF8,&input));
        ComPtr<IDxcAssembler> assembler;
        checked(sdk_->create(CLSID_DxcAssembler,IID_PPV_ARGS(&assembler)));
        auto take_result = [](IDxcOperationResult *result) {
            HRESULT hr;
            checked(result->GetStatus(&hr));
            if (FAILED(hr)) {
                ComPtr<IDxcBlobEncoding> error;
                result->GetErrorBuffer(&error);
                throw std::runtime_error(error ? std::string(static_cast<const char*>(error->GetBufferPointer()),
                                                             error->GetBufferSize()) : "DXIL assembly/validation failed");
            }
            ComPtr<IDxcBlob> bytes;
            checked(result->GetResult(&bytes));
            return bytes;
        };
        ComPtr<IDxcOperationResult> assembled, validated;
        checked(assembler->AssembleToContainer(input.Get(),&assembled));
        blob=take_result(assembled.Get());
        ComPtr<IDxcValidator> validator;
        const auto create_validator = reinterpret_cast<DxcCreateInstanceProc>(
            GetProcAddress(sdk_->validator.handle,"DxcCreateInstance"));
        if (!create_validator) throw std::runtime_error("SDK DXIL validator export unavailable");
        checked(create_validator(CLSID_DxcValidator,IID_PPV_ARGS(&validator)));
        checked(validator->Validate(blob.Get(),DxcValidatorFlags_InPlaceEdit,&validated));
        blob=take_result(validated.Get());
    } else {
        std::string expanded;
        if (native_ir) {
            expanded=expand_gfx1013_compute(source_text);
            if (expanded!=source_text && profile.rfind(L"cs_",0)!=0)
                throw std::runtime_error("native register lowering requires a compute profile");
        }
        const auto& hlsl=native_ir ? expanded : source_text;
        DxcBuffer source{hlsl.data(),hlsl.size(),DXC_CP_UTF8};
        const wchar_t* args[] = {L"-E",L"main",L"-T",profile.c_str(),L"-Gis",L"-O3"};
        ComPtr<IDxcResult> result;
        checked(compiler_->Compile(&source,args,ARRAYSIZE(args),nullptr,IID_PPV_ARGS(&result)));
        HRESULT status;
        checked(result->GetStatus(&status));
        if (FAILED(status)) {
            ComPtr<IDxcBlobUtf8> errors;
            checked(result->GetOutput(DXC_OUT_ERRORS,IID_PPV_ARGS(&errors),nullptr));
            throw std::runtime_error(errors && errors->GetStringLength() ? errors->GetStringPointer() : "DXC compile failed");
        }
        checked(result->GetOutput(DXC_OUT_OBJECT,IID_PPV_ARGS(&blob),nullptr));
    }
    const auto* start = static_cast<const unsigned char*>(blob->GetBufferPointer());
    Bytes bytes(start,start+blob->GetBufferSize());
    auto hash = digest(bytes.data(),bytes.size());
    fs::create_directories(cache);
    static std::atomic<unsigned long long> serial{};
    auto temp = path;
    temp += "."+std::to_string(GetCurrentProcessId())+"."+std::to_string(serial++)+".tmp";
    try {
        std::ofstream out(temp,std::ios::binary | std::ios::trunc);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out << key << hash;
        out.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
        out.close();
        if (!MoveFileExW(temp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            // Another process may already have published and opened the same
            // entry without FILE_SHARE_DELETE. Only accept an identical winner.
            auto winner=read_file(path);
            if(winner.size()!=128+bytes.size() || !std::equal(key.begin(),key.end(),winner.begin()) ||
               !std::equal(hash.begin(),hash.end(),winner.begin()+64) ||
               !std::equal(bytes.begin(),bytes.end(),winner.begin()+128))
                throw std::runtime_error("could not publish DXIL cache entry");
            fs::remove(temp);
        }
    } catch (...) {
        std::error_code ignored;
        fs::remove(temp,ignored);
        throw;
    }
    return {std::move(bytes),false};
}
}
