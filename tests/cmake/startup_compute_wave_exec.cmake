if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_compute_wave_exec_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_compute_wave_exec_check.py")
    if(DXC_EXECUTABLE AND TARGET offscreen_check)
        add_test(NAME startup_compute_wave_exec_warp COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_compute_wave_exec_check.py"
            --dispatch "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>" warp)
        set_tests_properties(startup_compute_wave_exec_warp PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77)
    endif()
endif()
