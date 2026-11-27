if(INTAKE_PYTHON AND EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_gfx2hlsl_placeholder_ps COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_placeholder_check.py")
    if(DXC_EXECUTABLE)
        add_test(NAME startup_gfx2hlsl_placeholder_ps_warp COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_placeholder_check.py" draw
            "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>")
        set_tests_properties(startup_gfx2hlsl_placeholder_ps_warp PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()
endif()
