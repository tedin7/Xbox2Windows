# Named reasons for compute programs that compile but get no runtime manifest (#3224).
# native_compute_manifest_reasons calls the formatter once per reason; the _requests check runs one reachable
# reason through compile_dxil --requests, --refusals, --watch and triage_run.py.
if(TARGET compile_dxil)
    add_executable(native_compute_manifest_reasons tests/native_compute_manifest_reasons.cpp)
    target_link_libraries(native_compute_manifest_reasons PRIVATE x2w_shader_compiler)
    set_property(TARGET native_compute_manifest_reasons PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_test(NAME native_compute_manifest_reasons COMMAND native_compute_manifest_reasons)
    if(EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll" AND EXISTS "${INTAKE_PYTHON}")
        add_test(NAME native_compute_manifest_reasons_requests COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_compute_manifest_reasons_check.py"
            "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
        set_tests_properties(native_compute_manifest_reasons_requests PROPERTIES TIMEOUT 60)
    endif()
endif()
