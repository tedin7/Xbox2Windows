# Six synchronized LDS reduction phases and packed/partial half operations.
if(EXISTS "${INTAKE_PYTHON}" AND TARGET offscreen_check AND DXC_EXECUTABLE)
    set(lds_reduction_check "${CMAKE_SOURCE_DIR}/tests/startup_compute_lds_reduction_check.py")
    add_test(NAME startup_compute_lds_reduction_proof COMMAND "${INTAKE_PYTHON}" -B "${lds_reduction_check}")
    add_test(NAME startup_compute_lds_reduction_warp COMMAND "${INTAKE_PYTHON}" -B
        "${lds_reduction_check}" dispatch "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>" warp)
    set_tests_properties(startup_compute_lds_reduction_warp PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    foreach(control IN ITEMS exchange fma-select literal-select half-address half-preserve)
        add_test(NAME startup_compute_lds_reduction_${control}_warp COMMAND "${INTAKE_PYTHON}" -B
            "${lds_reduction_check}" control ${control} "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>" warp)
        set_tests_properties(startup_compute_lds_reduction_${control}_warp PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77
            PASS_REGULAR_EXPRESSION "control ${control}: rejected at word=[0-9]+ ")
    endforeach()
    unset(lds_reduction_check)
endif()
