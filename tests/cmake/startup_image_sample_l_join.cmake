if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_image_sample_l_join_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_image_sample_l_join_check.py")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_image_sample_l_join_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_SOURCE_DIR}/tests/startup_image_sample_l_join_check.py" draw "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" ${adapter})
        set_tests_properties(startup_image_sample_l_join_${adapter} PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
    endforeach()
endif()
