if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_mesh57_waveloop COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_mesh57_waveloop_check.py")
    foreach(adapter IN ITEMS hardware warp)
        add_test(NAME startup_mesh57_waveloop_draw_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_mesh57_waveloop_check.py" draw "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" ${adapter})
        set_tests_properties(startup_mesh57_waveloop_draw_${adapter} PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77)
    endforeach()
endif()
