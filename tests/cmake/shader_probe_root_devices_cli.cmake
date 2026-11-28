# CLI, upfront path/symlink/hardlink guard, atomic output, and temp-copy mutation checks for tools/probe_root_devices.py.
if(EXISTS "${INTAKE_PYTHON}")
    add_test(NAME shader_probe_root_devices_cli_check COMMAND "${INTAKE_PYTHON}" -B
        "${CMAKE_SOURCE_DIR}/tests/probe_root_devices_cli_check.py")
    set_tests_properties(shader_probe_root_devices_cli_check PROPERTIES TIMEOUT 60)
endif()
