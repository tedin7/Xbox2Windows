# A per-vertex VCC crosses a two-arm join in a translated vertex stage (#2906).
if(TARGET compile_dxil AND EXISTS "${DXC_BIN_DIR}/dxcompiler.dll" AND EXISTS "${DXC_BIN_DIR}/dxil.dll")
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME native_graphics_vertex_vcc_join_${backend} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/native_graphics_vertex_vcc_join_check.py" "$<TARGET_FILE:compile_dxil>"
            "$<TARGET_FILE:offscreen_check>" "${DXC_BIN_DIR}" ${backend})
        set_tests_properties(native_graphics_vertex_vcc_join_${backend} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
    endforeach()
endif()
