if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_gfx2hlsl_dpp_derivative_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_dpp_derivative_check.py")
    if(DXC_EXECUTABLE AND TARGET offscreen_check)
        add_test(NAME startup_gfx2hlsl_dpp_derivative_warp COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_dpp_derivative_check.py"
            "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>")
        set_tests_properties(startup_gfx2hlsl_dpp_derivative_warp PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    endif()
endif()
