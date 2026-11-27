if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME textures_texture_blocks_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/texture_blocks_check.py")
    set_tests_properties(textures_texture_blocks_check PROPERTIES TIMEOUT 60)
endif()
