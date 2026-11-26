if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME startup_mesh57_variants_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_mesh57_variants_check.py")
    if(TARGET offscreen_check AND DXC_EXECUTABLE)
        foreach(variant IN ITEMS loop group_index count count_hoisted)
            add_test(NAME startup_mesh57_variants_${variant}_warp COMMAND "${INTAKE_PYTHON}" -B
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_mesh57_variants_check.py" draw ${variant}
                "${DXC_EXECUTABLE}" "$<TARGET_FILE:offscreen_check>")
            set_tests_properties(startup_mesh57_variants_${variant}_warp PROPERTIES TIMEOUT 60 SKIP_RETURN_CODE 77)
        endforeach()
    endif()
endif()
