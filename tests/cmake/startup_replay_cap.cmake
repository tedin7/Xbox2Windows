# Recorder regressions: reserved argument memory, bounded copies, and no recorder work after a prefix hits the file cap.
# The scene draws 300 presents after the cap. The check also reports time with the window closed and open.
if(TARGET startup_replay_quad AND EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_replay_cap_warp COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
        "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_cap_check.py"
        "$<TARGET_FILE:startup_replay_quad>" warp)
    set_tests_properties(startup_replay_cap_warp PROPERTIES TIMEOUT 600)
endif()
