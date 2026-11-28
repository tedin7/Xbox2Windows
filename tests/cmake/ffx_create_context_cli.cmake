if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME ffx_create_context_cli_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/ffx_create_context_cli_check.py")
    set_tests_properties(ffx_create_context_cli_check PROPERTIES TIMEOUT 60)
endif()
