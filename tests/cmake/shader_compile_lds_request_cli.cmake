if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME shader_compile_lds_request_cli_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/startup_compile_lds_request_cli_check.py")
    set_tests_properties(shader_compile_lds_request_cli_check PROPERTIES TIMEOUT 60)
endif()
