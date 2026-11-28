# RTV variant requests (req_<key>.rtv) resolved and cached by compile_dxil --requests (#2980). The controls need no
# device; the warp and hardware entries draw each variant through the stand-in and read the target back.
if(TARGET compile_dxil AND TARGET startup_stream_consumer AND EXISTS "${INTAKE_PYTHON}"
        AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll" AND EXISTS "${DXC_BIN_DIR}/dxc.exe")
    add_test(NAME native_rtv_variant_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_rtv_variant_check.py" controls
        "$<TARGET_FILE:compile_dxil>" "${DXC_BIN_DIR}")
    set_tests_properties(native_rtv_variant_controls PROPERTIES TIMEOUT 60)
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME native_rtv_variant_${backend} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_rtv_variant_check.py" e2e
            "$<TARGET_FILE:compile_dxil>" "$<TARGET_FILE:startup_stream_consumer>" "${DXC_BIN_DIR}" ${backend})
        set_tests_properties(native_rtv_variant_${backend} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
endif()
