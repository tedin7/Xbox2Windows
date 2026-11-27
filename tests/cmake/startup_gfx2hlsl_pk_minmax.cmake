# Packed binary16 min/max translation controls and D3D12 readback (#2022).
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_gfx2hlsl_pk_minmax_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_pk_minmax_check.py")
    if(TARGET offscreen_check)
        foreach(adapter IN ITEMS warp hardware)
            add_test(NAME startup_gfx2hlsl_pk_minmax_${adapter} COMMAND "${INTAKE_PYTHON}" -B
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_gfx2hlsl_pk_minmax_check.py" draw
                "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>" ${adapter})
            set_tests_properties(startup_gfx2hlsl_pk_minmax_${adapter} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
        endforeach()
    endif()
endif()
