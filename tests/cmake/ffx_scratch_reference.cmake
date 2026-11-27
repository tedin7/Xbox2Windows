if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME ffx_scratch_reference_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/ffx_scratch_reference_check.py")
    set_tests_properties(ffx_scratch_reference_check PROPERTIES TIMEOUT 60)
endif()
