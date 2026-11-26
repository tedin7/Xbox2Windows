# x2w_replay: a recording of startup_replay_quad replays to the same pictures, and a recording with one clear colour
# changed replays to a different last picture only (tests/startup_replay_check.py).
if(TARGET startup_replay_quad AND TARGET x2w_replay AND EXISTS "${INTAKE_PYTHON}")
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME startup_replay_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_check.py"
            "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>" "${backend}")
        set_tests_properties(startup_replay_${backend} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    endforeach()
endif()
