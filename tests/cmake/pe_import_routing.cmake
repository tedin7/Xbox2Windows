# Unit, boundary-validation, and temp-copy mutation checks for tools/import_routing.py.
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME pe_import_routing_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/import_routing_check.py")
    set_tests_properties(pe_import_routing_check PROPERTIES TIMEOUT 60)
endif()
