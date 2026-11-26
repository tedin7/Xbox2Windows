# X2W_D3D12_LAZY_TEXTURES=1 (Refs #2764, #2379, #1199): slot 133 read-only textures defer desktop
# CreateCommittedResource until first real GPU/copy/CPU-real use, refresh pre-created slot-18 SRVs on late
# materialization, and retire untouched textures cleanly at guest Release().
add_executable(startup_slot133_lazy_textures_check tests/startup_slot133_lazy_textures_check.cpp)
target_include_directories(startup_slot133_lazy_textures_check PRIVATE src)
target_compile_features(startup_slot133_lazy_textures_check PRIVATE cxx_std_17)
target_link_libraries(startup_slot133_lazy_textures_check PRIVATE d3d12_xs_shim d3d12 dxgi d3dcompiler psapi)
foreach(backend IN ITEMS warp hardware)
    foreach(debug IN ITEMS 0 1)
        set(lazy_tex_env "X2W_D3D12_BUDGET=1;X2W_D3D12_PERMISSIVE=1;X2W_D3D12_LIST_NOOP=1;X2W_D3D12_RENDER=1;X2W_RESOURCE_INDEX_CHECK=1;X2W_D3D12_DEBUG=${debug}")
        add_test(NAME startup_slot133_lazy_textures_${backend}_debug${debug} COMMAND startup_slot133_lazy_textures_check ${backend})
        set_tests_properties(startup_slot133_lazy_textures_${backend}_debug${debug} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77
            ENVIRONMENT "${lazy_tex_env};X2W_D3D12_RELEASE_RETIRED=1;X2W_D3D12_LAZY_TEXTURES=1")
    endforeach()
    set(lazy_tex_base "X2W_D3D12_BUDGET=1;X2W_D3D12_PERMISSIVE=1;X2W_D3D12_LIST_NOOP=1;X2W_D3D12_RENDER=1;X2W_RESOURCE_INDEX_CHECK=1;X2W_D3D12_DEBUG=0")
    add_test(NAME startup_slot133_lazy_textures_noretire_${backend} COMMAND startup_slot133_lazy_textures_check ${backend} no-retire)
    set_tests_properties(startup_slot133_lazy_textures_noretire_${backend} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77
        ENVIRONMENT "${lazy_tex_base};X2W_D3D12_LAZY_TEXTURES=1")
    add_test(NAME startup_slot133_lazy_textures_off_${backend} COMMAND startup_slot133_lazy_textures_check ${backend} off)
    set_tests_properties(startup_slot133_lazy_textures_off_${backend} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77
        ENVIRONMENT "${lazy_tex_base};X2W_D3D12_RELEASE_RETIRED=1")
endforeach()
