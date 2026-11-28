# Signed 32-bit add/sub (v_add_nc_i32, v_sub_nc_i32) lowering in tools/startup/gfx2hlsl.py.
if(EXISTS "${INTAKE_PYTHON}")
    add_test(
        NAME startup_gfx2hlsl_v_add_sub_nc_i32_controls
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_gfx2hlsl_v_add_sub_nc_i32_check.py"
    )
    if(WIN32 AND DXC_EXECUTABLE AND TARGET offscreen_check)
        set(_x2w_v_add_sub_nc_i32_fxc "")
        if(FXC_EXECUTABLE)
            set(_x2w_v_add_sub_nc_i32_fxc "${FXC_EXECUTABLE}")
        endif()
        add_test(
            NAME startup_gfx2hlsl_v_add_sub_nc_i32_warp
            COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_gfx2hlsl_v_add_sub_nc_i32_check.py"
                    draw "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>" warp ${_x2w_v_add_sub_nc_i32_fxc}
        )
        set_tests_properties(startup_gfx2hlsl_v_add_sub_nc_i32_warp PROPERTIES TIMEOUT 120)
        add_test(
            NAME startup_gfx2hlsl_v_add_sub_nc_i32_hardware
            COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_gfx2hlsl_v_add_sub_nc_i32_check.py"
                    draw "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>" hardware ${_x2w_v_add_sub_nc_i32_fxc}
        )
        set_tests_properties(startup_gfx2hlsl_v_add_sub_nc_i32_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    endif()
endif()
