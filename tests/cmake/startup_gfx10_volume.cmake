# Volumes in the 4 KiB standard console layout (0x105, one mip; src/startup_stubs/gfx10_layout.h) and the
# slot-70 descriptions and views that use them (src/startup_stubs/address_texture.h), without a device:
# address equations against addrlib's GFX10_SW_4K_S3 rows (Mesa 25.1.0, src/amd/addrlib, MIT), block
# order, distinct addresses, round trips and refusals. The thin run lays each volume out as a 2D array of
# its layers and is the negative control; the addrlib run compares every element address with addrlib
# and skips (77) without it.
add_executable(startup_gfx10_volume_check tests/startup_gfx10_volume_check.cpp)
target_compile_features(startup_gfx10_volume_check PRIVATE cxx_std_17)
target_include_directories(startup_gfx10_volume_check PRIVATE src)
if(XBOX2WINDOWS_ADDRLIB_DIR)
    target_link_libraries(startup_gfx10_volume_check PRIVATE xbw_addrlib)
    target_compile_definitions(startup_gfx10_volume_check PRIVATE XBOX2WINDOWS_HAVE_ADDRLIB)
endif()
add_test(NAME startup_gfx10_volume COMMAND startup_gfx10_volume_check)
add_test(NAME startup_gfx10_volume_thin_control COMMAND startup_gfx10_volume_check --mutate thin)
set_tests_properties(startup_gfx10_volume_thin_control PROPERTIES WILL_FAIL TRUE)
add_test(NAME startup_gfx10_volume_addrlib COMMAND startup_gfx10_volume_check --addrlib)
set_tests_properties(startup_gfx10_volume_addrlib PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
