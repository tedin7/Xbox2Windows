# Source-built native pixel buffer writes through the runtime-published graphics cache.
add_executable(native_raw_buffer_view_check tests/native_raw_buffer_view_check.cpp)
target_include_directories(native_raw_buffer_view_check PRIVATE src)
target_compile_features(native_raw_buffer_view_check PRIVATE cxx_std_17)
add_test(NAME native_raw_buffer_view_controls COMMAND native_raw_buffer_view_check)
if(TARGET compile_dxil AND TARGET startup_stream_consumer
   AND EXISTS "${INTAKE_PYTHON}" AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME native_pixel_raw_uav_${backend} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/native_pixel_raw_uav_check.py"
            "$<TARGET_FILE:compile_dxil>" "$<TARGET_FILE:startup_stream_consumer>" "${DXC_BIN_DIR}" ${backend})
        set_tests_properties(native_pixel_raw_uav_${backend} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    endforeach()
endif()
