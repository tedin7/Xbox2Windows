# Pure-Python unit, boundary-validation, and temp-copy mutation checks for tools/host_commit.py.
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME runtime_host_commit_unit_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/host_commit_unit_check.py")
    set_tests_properties(runtime_host_commit_unit_check PROPERTIES TIMEOUT 60)
endif()
