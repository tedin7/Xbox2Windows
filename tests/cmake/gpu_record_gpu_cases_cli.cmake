# Mocked CLI and mutation unit suite for tools/record_gpu_cases.py.
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME gpu_record_gpu_cases_cli_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/record_gpu_cases_cli_check.py")
    set_tests_properties(gpu_record_gpu_cases_cli_check PROPERTIES TIMEOUT 60)
endif()
