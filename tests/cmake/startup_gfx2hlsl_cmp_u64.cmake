# gfx2hlsl unsigned 64-bit VOPC compares (v_cmp/v_cmpx _u64): a CPU model of every form over values whose high and
# low halves disagree, refusals by name, then a pixel draw of the kill form and of the six relations in four operand
# forms, compared with a plain-integer reference per desktop adapter, with three mutated lowerings that must fail.
if(EXISTS "${INTAKE_PYTHON}" AND TARGET offscreen_check)
    add_test(NAME startup_gfx2hlsl_cmp_u64_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_cmp_u64_check.py")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_gfx2hlsl_cmp_u64_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_cmp_u64_check.py" draw "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" ${adapter} $<$<BOOL:${STARTUP_FXC}>:${STARTUP_FXC}>)
        set_tests_properties(startup_gfx2hlsl_cmp_u64_${adapter} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    endforeach()
endif()
