if(EXISTS "${INTAKE_PYTHON}" AND TARGET offscreen_check)
    add_test(NAME startup_gfx2hlsl_graphics_spill_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_graphics_spill_check.py")
    if(DXC_EXECUTABLE)
        add_test(NAME startup_gfx2hlsl_graphics_spill_warp COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_graphics_spill_check.py" "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>")
        set_tests_properties(startup_gfx2hlsl_graphics_spill_warp PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()
endif()
