if(EXISTS "${INTAKE_PYTHON}" AND TARGET offscreen_check)
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_gather_implicit_mip_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/startup_gather_implicit_mip_check.py" "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" ${adapter})
        set_tests_properties(startup_gather_implicit_mip_${adapter} PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)
    endforeach()
endif()
