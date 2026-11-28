# Native raw DWORD atomic add through the graphics runtime's published cache.
if(TARGET compile_dxil AND TARGET startup_stream_consumer
   AND EXISTS "${INTAKE_PYTHON}" AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME native_pixel_atomic_${backend} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/native_pixel_atomic_check.py"
            "$<TARGET_FILE:compile_dxil>" "$<TARGET_FILE:startup_stream_consumer>" "${DXC_BIN_DIR}" ${backend})
        set_tests_properties(native_pixel_atomic_${backend} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    endforeach()
endif()
