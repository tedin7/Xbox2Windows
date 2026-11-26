if(STARTUP_FXC AND EXISTS "${INTAKE_PYTHON}" AND TARGET startup_stream_consumer)
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME startup_pixel_history_draw_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_pixel_history_draw_check.py"
            "$<TARGET_FILE:startup_stream_consumer>" "${backend}")
        set_tests_properties(startup_pixel_history_draw_${backend} PROPERTIES TIMEOUT 90 SKIP_RETURN_CODE 77)
    endforeach()
endif()
