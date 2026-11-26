if(INTAKE_PYTHON AND EXISTS "${INTAKE_PYTHON}")
    set(packed_formats_check "${CMAKE_SOURCE_DIR}/tests/startup_compute_packed_formats_check.py")
    add_test(NAME startup_compute_packed_formats_controls
        COMMAND "${INTAKE_PYTHON}" -B "${packed_formats_check}")
    add_test(NAME startup_compute_packed_formats_warp
        COMMAND "${INTAKE_PYTHON}" -B "${packed_formats_check}" "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>")
    set_tests_properties(startup_compute_packed_formats_warp PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
    foreach(control IN ITEMS store register memory heap)
        add_test(NAME startup_compute_packed_formats_${control}_mutant_warp
            COMMAND "${INTAKE_PYTHON}" -B "${packed_formats_check}" "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>" "${control}")
        set_tests_properties(startup_compute_packed_formats_${control}_mutant_warp
            PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
    endforeach()
endif()
