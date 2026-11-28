if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME cpu_call_aot_cli_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/cpu_call_aot_cli_check.py")
    set_tests_properties(cpu_call_aot_cli_check PROPERTIES TIMEOUT 60)
endif()
