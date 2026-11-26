# gfx2hlsl first active lane of a pixel stage (#2618, the F rows): a 32-lane model of s_ff1_i32_b32 sD, exec_lo with
# v_readlane_b32 sX, vY, sD, refusals by name, then a triangle with partly covered quads compared per desktop wave with
# the input of the wave's lowest covered lane.
if(EXISTS "${INTAKE_PYTHON}" AND TARGET offscreen_check)
    add_test(NAME startup_gfx2hlsl_pixel_first_lane_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_pixel_first_lane_check.py")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_gfx2hlsl_pixel_first_lane_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_pixel_first_lane_check.py" draw "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" ${adapter} $<$<BOOL:${STARTUP_FXC}>:${STARTUP_FXC}>)
        set_tests_properties(startup_gfx2hlsl_pixel_first_lane_${adapter} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    endforeach()
endif()
