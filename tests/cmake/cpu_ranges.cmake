if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME cpu_ranges_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/cpu_ranges_check.py")
    set_tests_properties(cpu_ranges_check PROPERTIES TIMEOUT 60)
endif()
