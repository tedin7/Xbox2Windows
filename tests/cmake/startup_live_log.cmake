# The live debug-log window (tools/startup/live_log.py) starts minimized without taking the focus; X2W_LIVE_LOG=normal
# restores the old window in front, off skips it. On Windows with an interactive desktop the game window must keep the
# foreground; otherwise that part is skipped (77).
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_live_log COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/startup_live_log_check.py")
    set_tests_properties(startup_live_log PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
endif()
