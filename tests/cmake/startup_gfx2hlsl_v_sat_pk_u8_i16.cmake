# Saturating int16x2-to-uint8x2 pack (v_sat_pk_u8_i16) in tools/startup/gfx2hlsl.py.
if(EXISTS "${INTAKE_PYTHON}" AND TARGET offscreen_check)
    add_test(NAME startup_gfx2hlsl_v_sat_pk_u8_i16_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_v_sat_pk_u8_i16_check.py")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_gfx2hlsl_v_sat_pk_u8_i16_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_v_sat_pk_u8_i16_check.py" draw "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" ${adapter} $<$<BOOL:${STARTUP_FXC}>:${STARTUP_FXC}>)
        set_tests_properties(startup_gfx2hlsl_v_sat_pk_u8_i16_${adapter} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
endif()
