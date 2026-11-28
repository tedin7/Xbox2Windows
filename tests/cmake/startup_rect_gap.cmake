# The eight-vertex quad-and-edge-ring draw with 12-byte vertex records (#3131): the shared indexed rectangle
# expansion with back-face culling on, and the same draw through the startup runtime entry.
add_test(NAME startup_rect_geometry_warp COMMAND startup_rect_geometry)
add_test(NAME startup_rect_geometry_hardware COMMAND startup_rect_geometry hardware)
set_tests_properties(startup_rect_geometry_hardware PROPERTIES SKIP_RETURN_CODE 77)

foreach(backend IN ITEMS warp hardware)
    add_test(NAME startup_rect_gap_${backend} COMMAND "${CMAKE_COMMAND}" -E env "X2W_FXC=${STARTUP_FXC}"
        "${INTAKE_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_rect_gap_check.py"
        "$<TARGET_FILE:startup_stream_consumer>" "${backend}")
    set_tests_properties(startup_rect_gap_${backend} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
endforeach()
