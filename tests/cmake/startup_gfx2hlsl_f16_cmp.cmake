if(EXISTS "${INTAKE_PYTHON}")
    # gfx2hlsl binary16 VOPC compares (#1723): fifteen relations over NaN, infinity, signed zero and denormal halves,
    # drawn on WARP and hardware (FXC 5_1 and DXC 6_0) against the IEEE relation of the exact binary32 images.
    add_test(NAME startup_gfx2hlsl_f16_cmp_draw_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/startup_gfx2hlsl_f16_cmp_draw_check.py")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_gfx2hlsl_f16_cmp_draw_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/startup_gfx2hlsl_f16_cmp_draw_check.py" draw "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" ${adapter} $<$<BOOL:${STARTUP_FXC}>:${STARTUP_FXC}>)
        set_tests_properties(startup_gfx2hlsl_f16_cmp_draw_${adapter} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    endforeach()
endif()
