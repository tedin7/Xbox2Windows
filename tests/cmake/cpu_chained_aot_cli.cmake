if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME cpu_chained_aot_cli_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/cpu_chained_aot_cli_check.py")
    set_tests_properties(cpu_chained_aot_cli_check PROPERTIES TIMEOUT 60)
endif()
