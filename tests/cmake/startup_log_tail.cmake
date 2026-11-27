# tools/startup/log_tail.py: the stderr tail debug_launch.py prints into debugger_summary.out keeps the newest complete
# storage heartbeat whole, even when draw output after it fills the tail.
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_log_tail COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_log_tail_check.py")
    set_tests_properties(startup_log_tail PROPERTIES TIMEOUT 60)
endif()
