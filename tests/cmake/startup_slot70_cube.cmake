# Cube views of console-layout textures published by address (device slot 70). The cube draws run inside
# startup_address_console_{warp,hardware}; this is their negative control: the faces written in reverse slice
# order must not draw exactly.
if(STARTUP_FXC AND EXISTS "${INTAKE_PYTHON}" AND TARGET startup_stream_consumer)
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME startup_address_console_cube_order_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_address_console_check.py"
            "$<TARGET_FILE:startup_stream_consumer>" "${backend}" cube-order)
        set_tests_properties(startup_address_console_cube_order_${backend} PROPERTIES TIMEOUT 90 SKIP_RETURN_CODE 77)
    endforeach()
endif()
