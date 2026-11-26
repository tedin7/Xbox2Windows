# x2w_replay: a recording of startup_replay_quad replays to the same pictures, and a recording with one clear colour
# changed replays to a different last picture only (tests/startup_replay_check.py).
if(TARGET startup_replay_quad AND TARGET x2w_replay AND EXISTS "${INTAKE_PYTHON}")
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME startup_replay_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_check.py"
            "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>" "${backend}")
        set_tests_properties(startup_replay_${backend} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    endforeach()
    add_test(NAME startup_replay_title_ranges_warp COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
        "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_title_ranges_check.py"
        "$<TARGET_FILE:startup_replay_quad>" "$<TARGET_FILE:x2w_replay>")
    set_tests_properties(startup_replay_title_ranges_warp PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
endif()

# x2w_replay writes a recording's pages (PAGE/MAP) at their recorded addresses, in sequence order; a range that cannot be
# had stops it with exit 3 (tests/startup_replay_pages_play_check.py).
if(TARGET x2w_replay AND EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_replay_pages_play COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_pages_play_check.py"
        "$<TARGET_FILE:x2w_replay>")
    set_tests_properties(startup_replay_pages_play PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
endif()

add_executable(startup_replay_submit_pages tests/startup_replay_submit_pages.cpp)
target_compile_features(startup_replay_submit_pages PRIVATE cxx_std_17)
target_link_libraries(startup_replay_submit_pages PRIVATE d3d12_xs_shim d3d12 dxgi)
if(EXISTS "${INTAKE_PYTHON}")
    foreach(backend warp hardware)
        add_test(NAME startup_replay_submit_pages_${backend}
            COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_submit_pages_check.py"
                "$<TARGET_FILE:startup_replay_submit_pages>" "$<TARGET_FILE:x2w_replay>" "${backend}")
        set_tests_properties(startup_replay_submit_pages_${backend} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    endforeach()
endif()

# A synthetic compute dispatch writes to a placed UAV that aliases title memory.
if(TARGET startup_stream_consumer AND TARGET x2w_replay AND EXISTS "${INTAKE_PYTHON}" AND DXC_EXECUTABLE)
    foreach(backend warp hardware)
        add_test(NAME startup_replay_compute_${backend} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/startup_replay_compute_check.py" "$<TARGET_FILE:startup_stream_consumer>"
            "$<TARGET_FILE:x2w_replay>" "${DXC_EXECUTABLE}" "${backend}")
        set_tests_properties(startup_replay_compute_${backend} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    endforeach()
endif()
