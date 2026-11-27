# VOP3P 8-nibble integer dot products (v_dot8_i32_i4, v_dot8_u32_u4) in tools/startup/gfx2hlsl.py.
if(EXISTS "${INTAKE_PYTHON}" AND TARGET offscreen_check)
    add_test(NAME startup_gfx2hlsl_v_dot8_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_v_dot8_check.py")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_gfx2hlsl_v_dot8_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_v_dot8_check.py" draw "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" ${adapter} $<$<BOOL:${STARTUP_FXC}>:${STARTUP_FXC}>)
        set_tests_properties(startup_gfx2hlsl_v_dot8_${adapter} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
endif()
