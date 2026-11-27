if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_gfx2hlsl_reaching_defs_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_reaching_defs_check.py")
    if(TARGET offscreen_check)
        add_test(NAME pixel_aot_reaching_defs_warp COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_reaching_defs_check.py" draw
            "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>")
        set_tests_properties(pixel_aot_reaching_defs_warp PROPERTIES TIMEOUT 90 SKIP_RETURN_CODE 77)
    endif()
endif()
