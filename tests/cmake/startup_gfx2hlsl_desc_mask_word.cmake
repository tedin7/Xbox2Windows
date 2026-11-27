if(EXISTS "${INTAKE_PYTHON}" AND TARGET offscreen_check)
    set(desc_mask_word_check "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_desc_mask_word_check.py")
    add_test(NAME startup_gfx2hlsl_desc_mask_word_controls COMMAND "${INTAKE_PYTHON}" -B
        "${desc_mask_word_check}")
    add_test(NAME startup_gfx2hlsl_desc_mask_word_warp COMMAND "${INTAKE_PYTHON}" -B
        "${desc_mask_word_check}" "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>")
    set_tests_properties(startup_gfx2hlsl_desc_mask_word_warp PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
endif()
