# Samplers read from the directly indexed sampler heap at a fixed index, translated natively (issue #3076).
# The fixture builds source-written compute requests, runs them on a sampler heap of decoys and compares with a CPU model;
# the check script also covers the compile refusals. _warp skips (77): WARP has no 32-lane waves.
if(TARGET compile_dxil AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll" AND EXISTS "${INTAKE_PYTHON}")
    add_executable(native_sampler_heap tests/native_sampler_heap.cpp tests/desktop_gpu.cpp)
    target_include_directories(native_sampler_heap PRIVATE src tests)
    target_link_libraries(native_sampler_heap PRIVATE x2w_shader_compiler d3d12 dxgi d3dcompiler)
    set_property(TARGET native_sampler_heap PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME native_sampler_heap_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_sampler_heap_check.py"
            "$<TARGET_FILE:native_sampler_heap>" "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}" ${adapter})
        set_tests_properties(native_sampler_heap_${adapter} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
    endforeach()
endif()
