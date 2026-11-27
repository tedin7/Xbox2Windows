# A slot 67 request translated in-process by the C++ compiler (X2W_NATIVE_SHADERS) and dispatched through the
# stand-in. Without 32-lane waves the check verifies its refusals and exits 77.
if(TARGET x2w_shader_compiler AND TARGET startup_stream_consumer AND EXISTS "${INTAKE_PYTHON}"
        AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_compute_dispatch_native_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_compute_dispatch_native_check.py"
            "$<TARGET_FILE:startup_stream_consumer>" "${DXC_BIN_DIR}" ${adapter})
        set_tests_properties(startup_compute_dispatch_native_${adapter} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
endif()
