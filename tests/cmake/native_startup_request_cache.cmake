# Startup request consumption and runtime cache publication for compile_dxil (#2933).
if(TARGET compile_dxil AND TARGET startup_stream_consumer AND TARGET offscreen_check
   AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    add_test(NAME native_startup_request_cache_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/native_startup_request_cache_check.py" controls
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_startup_request_cache_controls PROPERTIES TIMEOUT 60)
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME native_startup_request_cache_${backend} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/native_startup_request_cache_check.py" e2e
            "$<TARGET_FILE:compile_dxil>" "$<TARGET_FILE:startup_stream_consumer>"
            "$<TARGET_FILE:offscreen_check>" "${DXC_BIN_DIR}" ${backend})
        set_tests_properties(native_startup_request_cache_${backend} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 90)
    endforeach()
endif()
