if(STARTUP_FXC AND EXISTS "${INTAKE_PYTHON}" AND TARGET startup_stream_consumer)
    add_test(NAME startup_frame_copy_mapped_spans_warp COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
        "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_frame_copy_check.py"
        "$<TARGET_FILE:startup_stream_consumer>" warp mapped-spans)
    set_tests_properties(startup_frame_copy_mapped_spans_warp PROPERTIES TIMEOUT 240 SKIP_RETURN_CODE 77)
endif()
