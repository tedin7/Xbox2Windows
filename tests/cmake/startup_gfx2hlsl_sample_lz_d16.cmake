if(EXISTS "${INTAKE_PYTHON}" AND TARGET offscreen_check)
    add_test(NAME startup_gfx2hlsl_sample_lz_d16_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_sample_lz_d16_check.py")
    add_test(NAME startup_gfx2hlsl_sample_lz_d16_warp COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_sample_lz_d16_check.py" draw "${DXC_EXECUTABLE}"
        "$<TARGET_FILE:offscreen_check>")
    set_tests_properties(startup_gfx2hlsl_sample_lz_d16_warp PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
endif()
