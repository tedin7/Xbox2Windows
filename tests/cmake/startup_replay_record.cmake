# X2W_REPLAY_RECORD: the stream draw fixture recorded draws the same pixels, and the recording is well formed (every call
# with its sequence number, the creations' returned wrappers, argument memory with its blobs).
if(TARGET startup_stream_consumer AND EXISTS "${INTAKE_PYTHON}")
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME startup_replay_record_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_replay_record_check.py"
            "$<TARGET_FILE:startup_stream_consumer>" "${backend}")
        set_tests_properties(startup_replay_record_${backend} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
endif()
