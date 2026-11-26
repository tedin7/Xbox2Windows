if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_gfx2hlsl_execz_arms_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_execz_arms_check.py")
    if(DXC_EXECUTABLE AND TARGET offscreen_check)
        add_test(NAME startup_gfx2hlsl_execz_arms_warp COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_execz_arms_check.py"
            dispatch "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>")
        set_tests_properties(startup_gfx2hlsl_execz_arms_warp PROPERTIES TIMEOUT 120)
    endif()
endif()
