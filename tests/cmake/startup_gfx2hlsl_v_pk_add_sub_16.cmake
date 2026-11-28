# Packed 16-bit integer add/sub (v_pk_add/sub_i16, v_pk_add/sub_u16) lowering in tools/startup/gfx2hlsl.py.
if(EXISTS "${INTAKE_PYTHON}")
    add_test(
        NAME startup_gfx2hlsl_v_pk_add_sub_16_controls
        COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_gfx2hlsl_v_pk_add_sub_16_check.py"
    )
    if(WIN32 AND DXC_EXECUTABLE AND TARGET offscreen_check)
        set(_x2w_v_pk_add_sub_16_fxc "")
        if(FXC_EXECUTABLE)
            set(_x2w_v_pk_add_sub_16_fxc "${FXC_EXECUTABLE}")
        endif()
        add_test(
            NAME startup_gfx2hlsl_v_pk_add_sub_16_warp
            COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_gfx2hlsl_v_pk_add_sub_16_check.py"
                    draw "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>" warp ${_x2w_v_pk_add_sub_16_fxc}
        )
        set_tests_properties(startup_gfx2hlsl_v_pk_add_sub_16_warp PROPERTIES TIMEOUT 120)
        add_test(
            NAME startup_gfx2hlsl_v_pk_add_sub_16_hardware
            COMMAND "${INTAKE_PYTHON}" -B "${CMAKE_SOURCE_DIR}/tests/startup_gfx2hlsl_v_pk_add_sub_16_check.py"
                    draw "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>" hardware ${_x2w_v_pk_add_sub_16_fxc}
        )
        set_tests_properties(startup_gfx2hlsl_v_pk_add_sub_16_hardware PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    endif()
endif()
