# x2w_replay is deterministic: one recording replayed twice (one after the other and at the same time) writes the same
# pictures and the same counters (tests/startup_replay_determinism_check.py).
if(TARGET startup_replay_quad AND TARGET x2w_replay AND EXISTS "${INTAKE_PYTHON}")
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME startup_replay_determinism_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_determinism_check.py"
            "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>" "${backend}")
        set_tests_properties(startup_replay_determinism_${backend} PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
    endforeach()
endif()
