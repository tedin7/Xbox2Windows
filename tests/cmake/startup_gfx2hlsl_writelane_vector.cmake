if(EXISTS "${INTAKE_PYTHON}" AND TARGET offscreen_check)
    set(writelane_vector_check "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_writelane_vector_check.py")
    add_test(NAME startup_gfx2hlsl_writelane_vector_controls COMMAND "${INTAKE_PYTHON}" -B "${writelane_vector_check}")
    add_test(NAME startup_gfx2hlsl_writelane_vector_warp COMMAND "${INTAKE_PYTHON}" -B "${writelane_vector_check}"
        "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>")
    set_tests_properties(startup_gfx2hlsl_writelane_vector_warp PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    unset(writelane_vector_check)
endif()
