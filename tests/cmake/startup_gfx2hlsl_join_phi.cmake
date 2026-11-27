if(EXISTS "${INTAKE_PYTHON}" AND TARGET offscreen_check)
    add_test(NAME startup_gfx2hlsl_join_phi_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_join_phi_check.py")
    if(DXC_EXECUTABLE)
        add_test(NAME startup_gfx2hlsl_join_phi_warp COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_join_phi_check.py" draw "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" warp)
        set_tests_properties(startup_gfx2hlsl_join_phi_warp PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endif()
endif()
