# x2w_replay restores the title memory a recorded draw reads: the page the title rewrites between presents comes back at its
# address and every replayed picture equals the recorded one (tests/startup_replay_pages_check.py).
if(TARGET startup_replay_pages AND TARGET x2w_replay AND EXISTS "${INTAKE_PYTHON}")
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME startup_replay_pages_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_pages_check.py"
            "$<TARGET_FILE:startup_replay_pages>" "$<TARGET_FILE:x2w_replay>" "${backend}")
        set_tests_properties(startup_replay_pages_${backend} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    endforeach()
endif()
