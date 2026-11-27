# The gray-world log lines of the console present (render mode, WARP): the presented picture's mean level and spread for a flat
# fill and for a black picture with a white half, and the per-interval counter line at frames 300 and 600.
if(EXISTS "${INTAKE_PYTHON}" AND TARGET startup_stream_consumer)
    add_test(NAME startup_present_gray COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_present_gray_check.py" "$<TARGET_FILE:startup_stream_consumer>")
    set_tests_properties(startup_present_gray PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    add_test(NAME startup_present_recent_fps_warp COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_present_gray_check.py" "$<TARGET_FILE:startup_stream_consumer>" --recent-fps)
    set_tests_properties(startup_present_recent_fps_warp PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
endif()
