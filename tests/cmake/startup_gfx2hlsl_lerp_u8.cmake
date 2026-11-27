if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_gfx2hlsl_lerp_u8_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/startup_gfx2hlsl_lerp_u8_check.py")
    if(DXC_EXECUTABLE AND TARGET offscreen_check)
        foreach(adapter IN ITEMS warp hardware)
            add_test(NAME startup_gfx2hlsl_lerp_u8_${adapter} COMMAND "${INTAKE_PYTHON}" -B
                "${CMAKE_SOURCE_DIR}/tests/startup_gfx2hlsl_lerp_u8_check.py" draw "${DXC_EXECUTABLE}"
                "$<TARGET_FILE:offscreen_check>" "${adapter}" "${STARTUP_FXC}")
            set_tests_properties(startup_gfx2hlsl_lerp_u8_${adapter} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
        endforeach()
    endif()
endif()
