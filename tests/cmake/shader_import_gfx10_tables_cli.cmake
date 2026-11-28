if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME shader_import_gfx10_tables_cli_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/startup_import_gfx10_tables_cli_check.py")
    set_tests_properties(shader_import_gfx10_tables_cli_check PROPERTIES TIMEOUT 60)
endif()
