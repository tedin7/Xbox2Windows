if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME tooling_cpu_leaf_aot_cli_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/cpu_leaf_aot_cli_check.py")
    set_tests_properties(tooling_cpu_leaf_aot_cli_check PROPERTIES TIMEOUT 60)
endif()
