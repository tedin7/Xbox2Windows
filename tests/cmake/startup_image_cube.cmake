# gfx2hlsl cube image sampling (#2943): translation controls, then a six-face TextureCube drawn from literal face
# addresses, translator mutants and the v_cube chain, compared with the address meaning and a desktop TextureCube draw.
if(EXISTS "${INTAKE_PYTHON}" AND TARGET offscreen_check)
    add_test(NAME startup_image_cube_controls COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_image_cube_check.py")
    foreach(adapter IN ITEMS warp hardware)
        add_test(NAME startup_image_cube_${adapter} COMMAND "${INTAKE_PYTHON}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/startup_image_cube_check.py" draw "${DXC_EXECUTABLE}"
            "$<TARGET_FILE:offscreen_check>" ${adapter})
        set_tests_properties(startup_image_cube_${adapter} PROPERTIES TIMEOUT 300 SKIP_RETURN_CODE 77)
    endforeach()
endif()
