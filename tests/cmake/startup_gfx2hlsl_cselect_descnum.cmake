# gfx2hlsl compute s_cselect_b32 on descriptor-derived data (#2618, the C500 rows): the select keeps both arms as an
# opaque symbol, every later use that needs its value is refused by name, and three mutated translators must fail;
# then case A of the compute check with the dead select is dispatched and all 320 words compared.
if(EXISTS "${INTAKE_PYTHON}" AND TARGET offscreen_check)
    add_test(NAME startup_gfx2hlsl_cselect_descnum_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_cselect_descnum_check.py")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_gfx2hlsl_cselect_descnum_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_cselect_descnum_check.py" dispatch "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" ${adapter})
        set_tests_properties(startup_gfx2hlsl_cselect_descnum_${adapter} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
endif()
