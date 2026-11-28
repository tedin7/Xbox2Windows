# The console linear mode (0x100) in src/startup_stubs/gfx10_layout.h. The mutated run expects the largest
# mip first and is a negative control; the addrlib run compares every element address of several linear
# textures with addrlib's SW_LINEAR (Mesa 25.1.0, src/amd/addrlib, MIT) and skips (77) without addrlib.
add_test(NAME startup_gfx10_layout_linear_order_control COMMAND startup_gfx10_layout_check --mutate linear-largest-first)
set_tests_properties(startup_gfx10_layout_linear_order_control PROPERTIES WILL_FAIL TRUE)
if(XBOX2WINDOWS_ADDRLIB_DIR)
    target_link_libraries(startup_gfx10_layout_check PRIVATE xbw_addrlib)
    target_compile_definitions(startup_gfx10_layout_check PRIVATE XBOX2WINDOWS_HAVE_ADDRLIB)
endif()
add_test(NAME startup_gfx10_layout_linear_addrlib COMMAND startup_gfx10_layout_check --addrlib)
set_tests_properties(startup_gfx10_layout_linear_addrlib PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)
# The negative control for the linear cases of startup_address_console: a 2D texture and a 1D chain written
# without the 256-byte padding must not draw exactly.
if(STARTUP_FXC AND EXISTS "${INTAKE_PYTHON}" AND TARGET startup_stream_consumer)
    foreach(backend IN ITEMS warp hardware)
        add_test(NAME startup_address_console_linear_unpadded_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
            "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_address_console_check.py"
            "$<TARGET_FILE:startup_stream_consumer>" "${backend}" linear-unpadded)
        set_tests_properties(startup_address_console_linear_unpadded_${backend} PROPERTIES TIMEOUT 90 SKIP_RETURN_CODE 77)
    endforeach()
endif()
