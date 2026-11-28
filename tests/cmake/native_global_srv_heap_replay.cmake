# GPU-selected global SRV heap slot: record and offline request replay with captured view types (issue #2908).
# The fixture runs the native recording path (NativeRecording and NativeDispatchStates) and the compile_dxil request
# replay on the same accepted and refused selections. _warp skips (77): WARP has no 32-lane waves.
if(TARGET compile_dxil AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll" AND EXISTS "${INTAKE_PYTHON}")
    add_executable(native_global_srv_heap_replay tests/native_global_srv_heap_replay.cpp tests/desktop_gpu.cpp)
    target_include_directories(native_global_srv_heap_replay PRIVATE src tests)
    target_link_libraries(native_global_srv_heap_replay PRIVATE x2w_shader_compiler d3d12 dxgi d3dcompiler)
    set_property(TARGET native_global_srv_heap_replay PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME native_global_srv_heap_replay_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_global_srv_heap_replay_check.py"
            "$<TARGET_FILE:native_global_srv_heap_replay>" "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}" ${adapter})
        set_tests_properties(native_global_srv_heap_replay_${adapter} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    endforeach()
    add_test(NAME native_raw_load_cache_bits_hardware COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_raw_load_cache_bits_check.py"
            "$<TARGET_FILE:native_global_srv_heap_replay>" "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_raw_load_cache_bits_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()
