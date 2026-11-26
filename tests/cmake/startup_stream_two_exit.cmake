if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_stream_two_exit_translate COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/startup_stream_two_exit_check.py")
    if(TARGET startup_stream_consumer)
        foreach(backend IN ITEMS warp hardware)
            add_test(NAME startup_stream_two_exit_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
                "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_stream_two_exit_check.py"
                "$<TARGET_FILE:startup_stream_consumer>" "${backend}")
            set_tests_properties(startup_stream_two_exit_${backend} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
        endforeach()
    endif()
endif()
